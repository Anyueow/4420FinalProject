import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Set page config
st.set_page_config(
    page_title="Trend Analysis",
    page_icon="📈",
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
        },
        'color_combos': pd.read_csv(data_dir / 'color_combo_predictions.csv')
    }

def create_trend_chart(data, x_col, y_col, title, color_col='confidence'):
    fig = px.bar(
        data.head(10),
        x=x_col,
        y=y_col,
        orientation='h',
        color=color_col,
        title=title,
        color_discrete_map={'High': '#2ecc71', 'Medium': '#f1c40f', 'Low': '#e74c3c'}
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=400
    )
    return fig

def create_color_combo_chart(data):
    # Create a matrix of color combinations
    color_matrix = pd.pivot_table(
        data,
        values='predicted',
        index='color_1',
        columns='color_2',
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=color_matrix.values,
        x=color_matrix.columns,
        y=color_matrix.index,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="Color Combination Predictions",
        xaxis_title="Color 2",
        yaxis_title="Color 1",
        height=600
    )
    
    return fig

# Main content
st.title("📈 Detailed Trend Analysis")

try:
    data = load_data()
    
    # Add feature selector
    feature = st.selectbox(
        "Select Feature to Analyze",
        ["Color", "Style", "Category", "Pattern"]
    )
    
    # Show different analyses based on feature
    if feature.lower() == "color":
        st.header("🎨 Color Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistical Model Predictions")
            st.plotly_chart(
                create_trend_chart(
                    data['color'],
                    'predicted',
                    'color',
                    "Top Color Predictions"
                ),
                use_container_width=True
            )
            
        with col2:
            st.subheader("Ensemble Model Predictions")
            st.plotly_chart(
                create_trend_chart(
                    data['ensemble']['color'],
                    'predicted',
                    'color',
                    "Ensemble Color Predictions"
                ),
                use_container_width=True
            )
        
        st.subheader("Color Combination Analysis")
        st.plotly_chart(
            create_color_combo_chart(data['color_combos']),
            use_container_width=True
        )
        
        # Add color trend insights
        st.subheader("Color Trend Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Rising Colors")
            rising_colors = data['color'][data['color']['confidence'] == 'High'].head()
            for _, row in rising_colors.iterrows():
                st.markdown(f"- {row['color']}: {row['predicted']:.1f}% confidence")
                
        with col2:
            st.markdown("#### Declining Colors")
            declining_colors = data['color'][data['color']['confidence'] == 'Low'].head()
            for _, row in declining_colors.iterrows():
                st.markdown(f"- {row['color']}: {row['predicted']:.1f}% confidence")
    
    else:
        feature_key = feature.lower()
        st.header(f"{feature} Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistical Model Predictions")
            st.plotly_chart(
                create_trend_chart(
                    data[feature_key],
                    'predicted',
                    feature_key,
                    f"Top {feature} Predictions"
                ),
                use_container_width=True
            )
            
        with col2:
            st.subheader("Ensemble Model Predictions")
            st.plotly_chart(
                create_trend_chart(
                    data['ensemble'][feature_key],
                    'predicted',
                    feature_key,
                    f"Ensemble {feature} Predictions"
                ),
                use_container_width=True
            )
        
        # Add trend insights
        st.subheader(f"{feature} Trend Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### Rising {feature}s")
            rising = data[feature_key][data[feature_key]['confidence'] == 'High'].head()
            for _, row in rising.iterrows():
                st.markdown(f"- {row[feature_key]}: {row['predicted']:.1f}% confidence")
                
        with col2:
            st.markdown(f"#### Declining {feature}s")
            declining = data[feature_key][data[feature_key]['confidence'] == 'Low'].head()
            for _, row in declining.iterrows():
                st.markdown(f"- {row[feature_key]}: {row['predicted']:.1f}% confidence")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure all required data files are present in the data/processed directory.") 