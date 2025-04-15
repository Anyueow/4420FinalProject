import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Category Predictions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📈 Fashion Category Predictions")

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_data():
    """Load the category predictions data"""
    try:
        # Load category predictions
        df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "category_predictions.csv")
        
        # Clean up data - remove duplicates and get average prediction for each category
        df_agg = df.groupby(['category', 'confidence'])['predicted'].mean().reset_index()
        
        # Sort by prediction value in descending order
        df_agg = df_agg.sort_values('predicted', ascending=False)
        
        return df_agg
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_prediction_chart(df_agg):
    """Create the prediction chart with custom styling"""
    # Map confidence levels to color shades of C00000 (dark red)
    color_map = {
        'High': '#C00000',    # Base dark red
        'Medium': '#D83B3B',  # Medium red
        'Low': '#E67373'      # Light red
    }

    # Create a color array based on confidence level
    colors = [color_map[conf] for conf in df_agg['confidence']]

    # Create the bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_agg['category'],
        y=df_agg['predicted'],
        marker_color=colors,
        text=df_agg['confidence'],
        hovertemplate='%{x}<br>Prediction: %{y:.2f}%<br>Confidence: %{text}<extra></extra>'
    ))

    # Update layout for transparent background and other styling
    fig.update_layout(
        title='Fashion Category Trend Predictions (% Growth)',
        title_font=dict(size=20, color='#333333'),
        xaxis_title='Category',
        yaxis_title='Predicted Percentage Growth',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(gridcolor='#E0E0E0'),
        margin=dict(t=80, b=120, l=70, r=40),
        height=600
    )

    # Add a faint horizontal grid
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')

    # Show top 10 categories for better readability
    top_categories = 10
    if len(df_agg) > top_categories:
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=df_agg['category'][:top_categories],
                ticktext=df_agg['category'][:top_categories],
                tickangle=45
            )
        )
    
    return fig

def main():
    # Load data
    df_agg = load_data()
    if df_agg is None:
        return

    # Create and display the chart
    fig = create_prediction_chart(df_agg)
    st.plotly_chart(fig, use_container_width=True)

    # Display the data table
    st.subheader("Detailed Predictions")
    st.dataframe(
        df_agg,
        column_config={
            'predicted': st.column_config.NumberColumn(
                'Predicted Growth (%)',
                format="%.2f"
            )
        }
    )

    # Add download button
    csv = df_agg.to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="category_predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main() 