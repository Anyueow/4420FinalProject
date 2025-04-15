import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="LSTM Test Results",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>

    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

def load_test_data(category):
    """Load test data for a given category."""
    file_path = f"data/lstm_tests/{category}_forecasting_test_results.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def create_comparison_chart(df, category):
    """Create a comparison chart between actual and predicted values."""
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Bar(
        x=df['category'],
        y=df['actual_fall25'],
        name='Actual',
        marker_color='#2ecc71'
    ))
    
    # Add predicted values
    fig.add_trace(go.Bar(
        x=df['category'],
        y=df['predicted_fall25'],
        name='Predicted',
        marker_color='#3498db'
    ))
    
    fig.update_layout(
        title=f"Actual vs Predicted {category.capitalize()} Frequencies",
        xaxis_title=category.capitalize(),
        yaxis_title="Frequency (%)",
        barmode='group',
        height=500
    )
    
    return fig

def main():
    st.title("LSTM Model Performance")
    
    # Explanation section
    st.markdown("""
        ## How Our LSTM Model Works
        
        Our LSTM (Long Short-Term Memory) model predicts fashion trends using the following process:
        
        1. **Training**: The model learns from historical data of previous seasons
        2. **Prediction**: It forecasts trends for Fall'25
        3. **Testing**: We compare predictions with actual Fall'25 data
        
        The model uses time series data to capture patterns and trends in fashion attributes.
    """)
    
    # Categories for testing
    categories = {
        'style': 'Styles',
        'category': 'Categories',
        'super_category': 'Super Categories',
        'pattern': 'Patterns',
        'color': 'Colors'
    }
    
    # Category selector
    selected_category = st.selectbox(
        "Select Category to View Performance",
        options=list(categories.keys()),
        format_func=lambda x: categories[x]
    )
    
    # Load and display test data
    df = load_test_data(selected_category)
    if df is not None:
        # Display comparison chart
        st.plotly_chart(
            create_comparison_chart(df, selected_category),
            use_container_width=True
        )
        
        # Calculate and display metrics
        st.markdown("### Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mae = (df['actual_fall25'] - df['predicted_fall25']).abs().mean()
            st.metric("Mean Absolute Error", f"{mae:.2f}%")
        
        with col2:
            mse = ((df['actual_fall25'] - df['predicted_fall25']) ** 2).mean()
            st.metric("Mean Squared Error", f"{mse:.2f}%")
        
        with col3:
            accuracy = (1 - (df['actual_fall25'] - df['predicted_fall25']).abs().sum() / df['actual_fall25'].sum()) * 100
            st.metric("Accuracy", f"{accuracy:.2f}%")
        
        # Display detailed comparison
        st.markdown("### Detailed Comparison")
        comparison_df = df[['category', 'actual_fall25', 'predicted_fall25', 'error', 'confidence']]
        comparison_df.columns = ['Category', 'Actual', 'Predicted', 'Error', 'Confidence']
        st.dataframe(comparison_df.style.format({
            'Actual': '{:.2f}%',
            'Predicted': '{:.2f}%',
            'Error': '{:.2f}%'
        }))
    
    else:
        st.error(f"No test data available for {categories[selected_category]}")

if __name__ == "__main__":
    main() 