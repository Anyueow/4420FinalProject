import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Set page config with light mode
st.set_page_config(
    page_title="Fashion Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Title
st.title("📊 Fashion Trend Analysis")


# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_data():
    """Load all the trend data files"""
    base_path = Path(PROJECT_ROOT / "data" / "processed")
    try:
        # Load CSV files
        style_freq = pd.read_csv(base_path / "style_frequencies.csv")
        style_pred = pd.read_csv(base_path / "style_predictions.csv")
        category_freq = pd.read_csv(base_path / "category_frequencies.csv")
        category_pred = pd.read_csv(base_path / "category_predictions.csv")
        super_cat_freq = pd.read_csv(base_path / "super_category_frequencies.csv")
        super_cat_pred = pd.read_csv(base_path / "super_category_predictions.csv")
        pattern_freq = pd.read_csv(base_path / "pattern_frequencies.csv")
        pattern_pred = pd.read_csv(base_path / "pattern_predictions.csv")
        
        # Load JSON data
        with open(base_path / "feature_analysis.json", 'r') as f:
            feature_analysis = json.load(f)
            
        # Ensure correct chronological order
        season_order = ["Fall24", "Spring25", "Fall25"]
        style_freq['season'] = pd.Categorical(style_freq['season'], categories=season_order, ordered=True)
        category_freq['season'] = pd.Categorical(category_freq['season'], categories=season_order, ordered=True)
        super_cat_freq['season'] = pd.Categorical(super_cat_freq['season'], categories=season_order, ordered=True)
        pattern_freq['season'] = pd.Categorical(pattern_freq['season'], categories=season_order, ordered=True)
        
        return {
            'style_frequencies': style_freq,
            'style_predictions': style_pred,
            'category_frequencies': category_freq,
            'category_predictions': category_pred,
            'super_category_frequencies': super_cat_freq,
            'super_category_predictions': super_cat_pred,
            'pattern_frequencies': pattern_freq,
            'pattern_predictions': pattern_pred,
            'feature_analysis': feature_analysis
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def plot_trend_over_time(df, feature_col, title):
    """Create a line plot showing trend evolution over time"""
    fig = px.line(df.sort_values('season'), 
                  x='season', 
                  y='percentage', 
                  color=feature_col,
                  title=title,
                  labels={'percentage': 'Percentage (%)', 'season': 'Season'},
                  markers=True)
    
    # Set transparent background and improved styling
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_x=0.5,
        title_font_size=20,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        margin=dict(t=50, l=50, r=50, b=50)
    )
    # Update grid and axes for better visibility
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     title_font=dict(size=14), tickfont=dict(size=12))
    return fig

def plot_predictions(df, feature_col, title):
    """Create a bar plot for predictions with Spring26 predictions"""
    fig = px.bar(df.head(10), 
                 x=feature_col, 
                 y='predicted',
                 color='confidence',
                 title=f"{title} (Spring26 Predictions)",
                 labels={'predicted': 'Predicted Percentage (%)'},
                 color_discrete_map={
                     'High': '#2ecc71',
                     'Medium': '#f1c40f',
                     'Low': '#e74c3c'
                 })
    
    # Set transparent background and improved styling
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_x=0.5,
        title_font_size=20,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        margin=dict(t=50, l=50, r=50, b=50)
    )
    # Update grid and axes for better visibility
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     title_font=dict(size=14), tickfont=dict(size=12))
    return fig

def create_trend_evolution_plot(evolution_data, feature_col, analysis_type):
    """Create a grouped bar plot for trend evolution"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Custom colors for seasons
    
    for idx, season in enumerate(['Fall24', 'Spring25', 'Fall25']):
        fig.add_trace(go.Bar(
            name=season,
            x=evolution_data[feature_col],
            y=evolution_data[season],
            text=evolution_data[season].round(1),
            textposition='auto',
            marker_color=colors[idx]
        ))
    
    fig.update_layout(
        title=f"Top 10 {analysis_type} Evolution Across Seasons",
        barmode='group',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_x=0.5,
        title_font_size=20,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, l=50, r=50, b=50)
    )
    
    # Update grid and axes for better visibility
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     title_font=dict(size=14), tickfont=dict(size=12))
    return fig

def main():
    st.header("Trend Analysis Dashboard")
    st.write("Analyze and visualize fashion trends across seasons (Fall24 → Spring25 → Fall25)")
    
    # Load data
    data = load_data()
    if not data:
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Styles", "Categories", "Super Categories", "Patterns"]
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    if analysis_type == "Styles":
        freq_data = data['style_frequencies']
        pred_data = data['style_predictions']
        feature_col = 'style'
    elif analysis_type == "Categories":
        freq_data = data['category_frequencies']
        pred_data = data['category_predictions']
        feature_col = 'category'
    elif analysis_type == "Super Categories":
        freq_data = data['super_category_frequencies']
        pred_data = data['super_category_predictions']
        feature_col = 'super_category'
    else:  # Patterns
        freq_data = data['pattern_frequencies']
        pred_data = data['pattern_predictions']
        feature_col = 'pattern'
    
    # Historical Trends
    with col1:
        st.subheader("Historical Trends")
        fig1 = plot_trend_over_time(
            freq_data,
            feature_col,
            f"{analysis_type} Trends Over Time"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # Predictions
    with col2:
        st.subheader("Future Predictions (Spring26)")
        fig2 = plot_predictions(
            pred_data,
            feature_col,
            f"Top Predicted {analysis_type}"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Trend Statistics
    st.subheader("Trend Statistics")
    stats_cols = st.columns(2)
    
    with stats_cols[0]:
        st.write("Current Season (Fall25) Top Items")
        current_season = freq_data[freq_data['season'] == "Fall25"]
        current_season = current_season.sort_values('percentage', ascending=False).head(5)
        st.dataframe(
            current_season[[feature_col, 'percentage']],
            column_config={
                'percentage': st.column_config.NumberColumn(
                    'Percentage (%)',
                    format="%.2f"
                )
            }
        )
    
    with stats_cols[1]:
        st.write("Spring26 Predictions with Confidence")
        momentum_data = pred_data.sort_values('predicted', ascending=False).head(5)
        st.dataframe(
            momentum_data[[feature_col, 'predicted', 'confidence']],
            column_config={
                'predicted': st.column_config.NumberColumn(
                    'Predicted (%)',
                    format="%.2f"
                )
            }
        )
    
    # Trend Evolution
    st.subheader("Trend Evolution")
    evolution_data = freq_data.pivot(index=feature_col, columns='season', values='percentage').reset_index()
    evolution_data = evolution_data.fillna(0)
    evolution_data = evolution_data.sort_values('Fall25', ascending=False).head(10)
    
    fig3 = create_trend_evolution_plot(evolution_data, feature_col, analysis_type)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Feature Analysis Details
    st.subheader("Detailed Analysis")
    if st.checkbox("Show Feature Analysis Details"):
        st.json(data['feature_analysis'])

if __name__ == "__main__":
    main()

# Download buttons with improved styling
st.sidebar.header("Export Data")
if st.sidebar.download_button(
    "Download Current Trends",
    data=freq_data.to_csv(index=False),
    file_name="current_trends.csv",
    mime="text/csv"
):
    st.sidebar.success("Downloaded trends!")

if st.sidebar.download_button(
    "Download Predictions",
    data=pred_data.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv"
):
    st.sidebar.success("Downloaded predictions!") 