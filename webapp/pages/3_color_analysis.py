import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import io

# Set page config
st.set_page_config(
    page_title="Color Analysis",
    page_icon="🎨",
    layout="wide"
)

def load_data():
    """Load color-related data."""
    data_dir = Path("data/processed")
    return {
        'color_predictions': pd.read_csv(data_dir / 'color_predictions.csv'),
        'color_combos': pd.read_csv(data_dir / 'color_combo_predictions.csv'),
        'color_stats': pd.read_csv(data_dir / 'color_trend_stats.csv'),
        'labels': pd.read_csv(data_dir / 'fashion_labels_with_colors.csv')
    }

def hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_color_swatch(hex_color, size=(100, 100)):
    """Create a color swatch image."""
    img = Image.new('RGB', size, hex_color)
    return img

def create_color_palette_chart(colors, values, title):
    """Create a horizontal bar chart with color swatches."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=values,
        y=colors,
        orientation='h',
        marker_color=colors,
        text=values.round(1).astype(str) + '%',
        textposition='auto',
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title="Color",
        xaxis_title="Predicted Percentage",
        height=400,
        showlegend=False
    )
    
    return fig

def create_color_combo_heatmap(data):
    """Create a heatmap of color combinations."""
    # Create a matrix of color combinations
    combo_matrix = pd.pivot_table(
        data,
        values='predicted',
        index='color_1',
        columns='color_2',
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=combo_matrix.values,
        x=combo_matrix.columns,
        y=combo_matrix.index,
        colorscale='Viridis',
        text=np.round(combo_matrix.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Color Combination Predictions",
        xaxis_title="Secondary Color",
        yaxis_title="Primary Color",
        height=600
    )
    
    return fig

# Main content
st.title("🎨 Color Trend Analysis")
st.markdown("""
This page provides detailed analysis of color trends, including:
- Individual color predictions
- Color combination analysis
- Seasonal color evolution
- Color palette recommendations
""")

try:
    data = load_data()
    
    # Top Colors Overview
    st.header("Top Color Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Color palette visualization
        st.plotly_chart(
            create_color_palette_chart(
                data['color_predictions']['color'].head(10),
                data['color_predictions']['predicted'].head(10),
                "Top 10 Predicted Colors"
            ),
            use_container_width=True
        )
    
    with col2:
        # Color swatches
        st.subheader("Color Swatches")
        swatch_cols = st.columns(5)
        for i, (_, color) in enumerate(data['color_predictions'].head().iterrows()):
            with swatch_cols[i % 5]:
                swatch = create_color_swatch(color['color'])
                st.image(swatch, caption=f"{color['color']}\n{color['predicted']:.1f}%")
    
    # Color Combinations
    st.header("Color Combination Analysis")
    st.plotly_chart(
        create_color_combo_heatmap(data['color_combos']),
        use_container_width=True
    )
    
    # Color Trends Over Time
    st.header("Color Evolution")
    
    # Create color trend chart
    color_trends = data['labels'].melt(
        id_vars=['season'],
        value_vars=[f'color_{i}' for i in range(1, 6)],
        var_name='color_position',
        value_name='color'
    )
    
    color_trend_counts = color_trends.groupby(['season', 'color']).size().reset_index(name='count')
    
    fig = px.line(
        color_trend_counts,
        x='season',
        y='count',
        color='color',
        title="Color Trends Across Seasons"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Color Insights
    st.header("Color Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rising Colors")
        rising_colors = data['color_predictions'][
            data['color_predictions']['confidence'] == 'High'
        ].head()
        
        for _, color in rising_colors.iterrows():
            st.markdown(f"""
            - **{color['color']}**
              - Predicted: {color['predicted']:.1f}%
              - Confidence: {color['confidence']}
            """)
    
    with col2:
        st.subheader("Declining Colors")
        declining_colors = data['color_predictions'][
            data['color_predictions']['confidence'] == 'Low'
        ].head()
        
        for _, color in declining_colors.iterrows():
            st.markdown(f"""
            - **{color['color']}**
              - Predicted: {color['predicted']:.1f}%
              - Confidence: {color['confidence']}
            """)
    
    # Color Combination Recommendations
    st.header("Color Combination Recommendations")
    top_combos = data['color_combos'].sort_values('predicted', ascending=False).head(5)
    
    for _, combo in top_combos.iterrows():
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            swatch1 = create_color_swatch(combo['color_1'])
            st.image(swatch1, caption=combo['color_1'])
        
        with col2:
            swatch2 = create_color_swatch(combo['color_2'])
            st.image(swatch2, caption=combo['color_2'])
        
        with col3:
            st.markdown(f"""
            **Prediction: {combo['predicted']:.1f}%**  
            Confidence: {combo['confidence']}
            """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure all required data files are present in the data/processed directory.") 