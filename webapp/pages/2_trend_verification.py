import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import os
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Trend Verification",
    page_icon="🔍",
    layout="wide"
)

def load_data():
    """Load prediction data and image paths."""
    data_dir = Path("data/processed")
    image_dir = Path("data/scraped/runway")
    
    data = {
        'predictions': {
            'style': pd.read_csv(data_dir / 'style_predictions.csv'),
            'category': pd.read_csv(data_dir / 'category_predictions.csv'),
            'color': pd.read_csv(data_dir / 'color_predictions.csv'),
            'pattern': pd.read_csv(data_dir / 'pattern_predictions.csv')
        },
        'images': {}
    }
    
    # Load fashion labels with image paths
    labels = pd.read_csv(data_dir / 'fashion_labels_with_colors.csv')
    data['labels'] = labels
    
    return data

def find_trend_examples(data, trend_type, trend_value, n_examples=5):
    """Find example images for a specific trend."""
    if trend_type == 'color':
        # For colors, check color_1 through color_5
        mask = False
        for i in range(1, 6):
            mask = mask | (data['labels'][f'color_{i}'] == trend_value)
    else:
        # For other features, check the specific column
        mask = data['labels'][trend_type] == trend_value
    
    examples = data['labels'][mask]['image_path'].head(n_examples).tolist()
    return examples

# Main content
st.title("🔍 Trend Verification")
st.markdown("""
This page provides visual verification of predicted trends by showing actual runway images 
that exemplify each trend.
""")

try:
    data = load_data()
    
    # Feature selector
    feature = st.selectbox(
        "Select Feature to Verify",
        ["Color", "Style", "Category", "Pattern"]
    )
    
    feature_key = feature.lower()
    
    # Get top trends for selected feature
    top_trends = data['predictions'][feature_key].head(5)
    
    st.header(f"Top {feature} Trends Visual Verification")
    
    # Display each trend with example images
    for idx, trend in top_trends.iterrows():
        trend_value = trend[feature_key]
        confidence = trend['confidence']
        predicted = trend['predicted']
        
        st.subheader(f"{trend_value} (Confidence: {confidence}, Predicted: {predicted:.1f}%)")
        
        # Find example images
        examples = find_trend_examples(data, feature_key, trend_value)
        
        if examples:
            cols = st.columns(min(len(examples), 5))
            for i, (col, img_path) in enumerate(zip(cols, examples)):
                try:
                    with col:
                        img = Image.open(img_path)
                        st.image(img, caption=f"Example {i+1}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image {img_path}: {str(e)}")
        else:
            st.info(f"No example images found for {trend_value}")
        
        st.markdown("---")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure all required data files are present in the data/processed directory.") 