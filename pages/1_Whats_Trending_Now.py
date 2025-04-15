import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

st.set_page_config(
    page_title="What's Trending Now",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .color-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin: 20px 0;
    }
    .color-swatch {
        height: 100px;
        border-radius: 8px;
        border: 1px solid #ddd;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 8px;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def load_trending_data(category):
    """Load prediction data for a given category."""
    try:
        file_path = Path("data") / "predictions" / f"{category}_predictions.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, skiprows=1, names=['category', 'predicted', 'confidence'])
            df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
            return df
        else:
            st.warning(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_bar_chart(df, category, title):
    """Create a bar chart for the given category."""
    # Get top 10 predictions
    df_top10 = df.head(10)
    
    # For other categories, use the original visualization
    fig = px.bar(
        df_top10,
        x='predicted',
        y='category',
        orientation='h',
        title=title,
        color='confidence',
        color_discrete_map={
            'High': '#2ecc71',
            'Medium': '#f1c40f',
            'Low': '#e74c3c'
        }
    )
    fig.update_layout(
        height=400,
        xaxis_title="Predicted Frequency (%)",
        yaxis_title=category.capitalize(),
        showlegend=True
    )
    
    return fig

def create_color_grid(df):
    """Create an HTML grid of color swatches."""
    top_10_colors = df.head(10)
    
    color_grid_html = '<div class="color-grid">'
    for _, row in top_10_colors.iterrows():
        color = row['category']
        if not color.startswith('#'):
            color = '#' + color
        color_grid_html += f'<div class="color-swatch" style="background-color: {color}"><div>{color}</div><div>{row["predicted"]:.2f}%</div></div>'
    color_grid_html += '</div>'
    
    return color_grid_html

def main():
    st.title("What's Trending Right Now?")
    st.markdown("""
        Explore the latest fashion trends predicted for Fall'25 across different categories.
        Each category shows the top 10 predicted trends.
    """)
    
    categories = {
        'style': 'Styles',
        'category': 'Clothing Items',
        'super_category': 'Clothing Categories',
        'pattern': 'Patterns',
        'color': 'Colors'
    }
    
    tabs = st.tabs([categories[cat] for cat in categories.keys()])
    
    for tab, (category, title) in zip(tabs, categories.items()):
        with tab:
            df = load_trending_data(category)
            if df is not None:
                if category == 'color':
                    st.write("### Top 10 Color Predictions")
                    st.write(create_color_grid(df), unsafe_allow_html=True)
                else:
                    st.plotly_chart(
                        create_bar_chart(df, category, f"Top 10 {title} for Fall'25"),
                        use_container_width=True
                    )
            else:
                st.error(f"No prediction data available for {title}")

if __name__ == "__main__":
    main() 