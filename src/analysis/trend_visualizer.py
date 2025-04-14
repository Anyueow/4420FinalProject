import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

def load_feature_analysis():
    """Load the feature analysis data."""
    analysis_path = Path('data/processed/feature_analysis.json')
    with open(analysis_path, 'r') as f:
        return json.load(f)

def analyze_super_category_trends(data):
    """Analyze trends in super categories across seasons."""
    trends = []
    
    # Extract super category data for each season
    for season, season_data in data['categories'].items():
        for super_cat, count in season_data['super_categories'].items():
            trends.append({
                'season': season,
                'super_category': super_cat,
                'count': count,
                'percentage': (count / season_data['total_items']) * 100
            })
    
    return pd.DataFrame(trends)

def create_trend_visualizations(df):
    """Create interactive visualizations for trend analysis."""
    # Sort seasons chronologically
    season_order = ['Fall24', 'Spring25', 'Fall25']
    df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
    
    # Create bar chart for super category counts
    fig_counts = px.bar(
        df,
        x='season',
        y='count',
        color='super_category',
        title='Super Category Counts by Season',
        labels={'count': 'Number of Items', 'season': 'Season'},
        barmode='group'
    )
    
    # Create line chart for percentage trends
    fig_percentages = px.line(
        df,
        x='season',
        y='percentage',
        color='super_category',
        title='Super Category Percentage Trends',
        labels={'percentage': 'Percentage of Total Items (%)', 'season': 'Season'},
        markers=True
    )
    
    # Calculate trend changes
    trend_changes = []
    for super_cat in df['super_category'].unique():
        cat_data = df[df['super_category'] == super_cat].sort_values('season')
        if len(cat_data) > 1:
            latest = cat_data.iloc[-1]['percentage']
            previous = cat_data.iloc[-2]['percentage']
            change = latest - previous
            trend_changes.append({
                'super_category': super_cat,
                'change': change,
                'trend': 'Increasing' if change > 0 else 'Decreasing'
            })
    
    trend_df = pd.DataFrame(trend_changes)
    
    # Create bar chart for trend changes
    fig_changes = px.bar(
        trend_df,
        x='super_category',
        y='change',
        color='trend',
        title='Recent Super Category Trend Changes',
        labels={'change': 'Percentage Point Change', 'super_category': 'Super Category'},
        color_discrete_map={'Increasing': 'green', 'Decreasing': 'red'}
    )
    
    return fig_counts, fig_percentages, fig_changes, trend_df

def main():
    st.title("Fashion Trend Analysis Dashboard")
    
    # Load and analyze data
    data = load_feature_analysis()
    trends_df = analyze_super_category_trends(data)
    
    # Create visualizations
    fig_counts, fig_percentages, fig_changes, trend_df = create_trend_visualizations(trends_df)
    
    # Display visualizations
    st.plotly_chart(fig_counts, use_container_width=True)
    st.plotly_chart(fig_percentages, use_container_width=True)
    st.plotly_chart(fig_changes, use_container_width=True)
    
    # Display trend summary
    st.subheader("Trend Summary")
    st.write("Recent changes in super category popularity:")
    
    # Create a styled table for trend changes
    st.dataframe(
        trend_df.style.apply(
            lambda x: ['background-color: lightgreen' if v > 0 else 'background-color: lightcoral' 
                      for v in x['change']],
            subset=['change']
        )
    )
    
    # Add detailed analysis
    st.subheader("Detailed Analysis")
    
    # Get the latest season's data
    latest_season = trends_df['season'].max()
    latest_data = trends_df[trends_df['season'] == latest_season].sort_values('percentage', ascending=False)
    
    st.write(f"Current Season ({latest_season}) Distribution:")
    st.dataframe(latest_data[['super_category', 'count', 'percentage']].style.format({'percentage': '{:.1f}%'}))
    
    # Highlight significant changes
    significant_changes = trend_df[abs(trend_df['change']) > 5]
    if not significant_changes.empty:
        st.write("Significant Trend Changes:")
        for _, row in significant_changes.iterrows():
            st.write(f"- {row['super_category']}: {'Increased' if row['change'] > 0 else 'Decreased'} by {abs(row['change']):.1f} percentage points")

if __name__ == "__main__":
    main() 