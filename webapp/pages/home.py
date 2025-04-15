import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Fashion Trend Analysis",
    page_icon="👗",
    layout="wide"
)

def load_data():
    """Load all prediction and analysis data."""
    data_dir = Path("data/processed")
    return {
        'style': pd.read_csv(data_dir / 'style_predictions.csv'),
        'category': pd.read_csv(data_dir / 'category_predictions.csv'),
        'color': pd.read_csv(data_dir / 'color_predictions.csv'),
        'pattern': pd.read_csv(data_dir / 'pattern_predictions.csv'),
        'ensemble': {
            'style': pd.read_csv(data_dir / 'style_ensemble_predictions.csv'),
            'category': pd.read_csv(data_dir / 'category_ensemble_predictions.csv'),
            'color': pd.read_csv(data_dir / 'color_ensemble_predictions.csv'),
            'pattern': pd.read_csv(data_dir / 'pattern_ensemble_predictions.csv')
        }
    }

# Main content
st.title("🎨 Fashion Trend Analysis Dashboard")
st.markdown("""
### Welcome to the Fashion Trend Analysis Platform
This dashboard provides comprehensive analysis of fashion trends across multiple dimensions:
- 🎨 Color Trends and Combinations
- 👗 Style Evolution
- 📊 Category Analysis
- 🔄 Pattern Predictions
""")

# Load data
try:
    data = load_data()
    
    # Top trends overview
    st.header("Spring 2026 Top Trend Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Top Color Trends")
        color_fig = px.bar(
            data['color'].head(),
            x='predicted',
            y='color',
            orientation='h',
            color='confidence',
            title="Top Color Predictions",
            color_discrete_map={'High': '#2ecc71', 'Medium': '#f1c40f', 'Low': '#e74c3c'}
        )
        color_fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(color_fig, use_container_width=True)
        
    with col2:
        st.subheader("👗 Top Style Trends")
        style_fig = px.bar(
            data['style'].head(),
            x='predicted',
            y='style',
            orientation='h',
            color='confidence',
            title="Top Style Predictions",
            color_discrete_map={'High': '#2ecc71', 'Medium': '#f1c40f', 'Low': '#e74c3c'}
        )
        style_fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(style_fig, use_container_width=True)

    # Confidence Distribution
    st.header("Prediction Confidence Overview")
    
    def create_confidence_chart(data, title):
        confidence_counts = data['confidence'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=confidence_counts.index,
            values=confidence_counts.values,
            hole=.3,
            marker_colors=['#2ecc71', '#f1c40f', '#e74c3c']
        )])
        fig.update_layout(title=title)
        return fig
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(create_confidence_chart(data['color'], "Color Predictions Confidence"), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_confidence_chart(data['style'], "Style Predictions Confidence"), use_container_width=True)
    
    with col3:
        st.plotly_chart(create_confidence_chart(data['pattern'], "Pattern Predictions Confidence"), use_container_width=True)

    # Navigation guide
    st.markdown("""
    ### 🚀 Explore More
    Use the sidebar to navigate to detailed analysis pages:
    - **Trend Analysis**: Deep dive into trend predictions and patterns
    - **Trend Verification**: Visual verification of predicted trends
    - **Color Analysis**: Detailed color trend analysis and combinations
    """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure all required data files are present in the data/processed directory.") 