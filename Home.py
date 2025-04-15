import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
import os

# Set page config
st.set_page_config(
    page_title="Fashion Trend Analysis",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Force light mode */
    :root {
        --primary-color: #2c3e50;
        --background-color: #ffffff;
        --secondary-background-color: #f5f5f5;
        --text-color: #2c3e50;
        --font: sans-serif;
    }
    
    /* Main app styling */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font);
        margin: 0 auto;
    }
    
    /* Card styling */
    .card {
        background-color: var(--background-color);
        border-radius: 10px;
        margin-bottom: 20px;
        padding: 2px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: var(--text-color);
    }
    
    /* Title styling */
    .title {
        color: var(--primary-color);
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #7f8c8d;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    
    /* Quote box styling */
    .quote-box {
        background-color: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        padding: 20px;
        margin: 20px 0;
        font-style: italic;
        color: var(--text-color);
    }
    
    /* Process step styling */
    .process-step {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        padding: 10px;
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        color: var(--text-color);
    }
    
    .process-icon {
        font-size: 24px;
        margin-right: 15px;
        min-width: 40px;
    }
    
    .process-text {
        flex: 1;
        color: var(--text-color);
    }
    
    /* Footer styling */
    .footer {
        
        bottom: 0;
        left: 0;
        right: 0;
        background-color: var(--secondary-background-color);
        padding: 10px 20px;
        text-align: left;
        font-size: 0.9em;
        color: var(--text-color);
        border-top: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    .footer-content {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Override Streamlit's dark mode */
    [data-testid="stAppViewContainer"] {
        background-color: var(--background-color);
        padding-bottom: 60px; /* Add space for footer */
    }
    
    [data-testid="stHeader"] {
        background-color: var(--background-color);
    }
    
    [data-testid="stToolbar"] {
        background-color: var(--background-color);
    }
    
    /* Ensure text is readable */
    .stMarkdown {
        color: var(--text-color);
    }
    
    /* Style metrics */
    [data-testid="stMetricValue"] {
        color: var(--primary-color);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-color);
    }
    
    /* Style headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color);
    }
    
    /* Style Streamlit components */
    .element-container {
        color: var(--text-color);
    }
    
    /* Style Streamlit text */
    .stText {
        color: var(--text-color);
    }
    
    /* Style Streamlit markdown */
    .markdown-text-container {
        color: var(--text-color);
    }
    
    /* Style Streamlit headers */
    .streamlit-expanderHeader {
        color: var(--primary-color);
    }
    
    /* Style Streamlit buttons */
    .stButton > button {
        color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    """Load sample data for demonstration."""
    # This will be replaced with actual data once the scraper completes
    return {
        "total_shows": 0,
        "total_images": 0,
        "designers": [],
        "seasons": []
    }

def main():
    # Header
    st.markdown('<h1 class="title">Fashion Trend Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Using advanced ML techniques to analyze fashion trends for customizing retail storefronts</p>', unsafe_allow_html=True)
    
    # Overview Section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("📊 Project Overview")
        st.write("""
                The fashion industry is in flux. With hyper-connected consumers and tiktok influencers the fashion indsutry is faced with ever-accelerating trend cycles. They have redefined how style is discovered, 
                 shared—and shopped. A single image can turn a fringe look into a must-have, placing it in a shopping cart within seconds. Yet, for brands and retailers, 
                 keeping up with these shifts is harder than ever.

                 Despite the growth, fashion remains volatile. Traditional trend forecasting—based on intuition, 
                 sales history, and seasonal reports—can't keep pace. Yet, all these trends seem to trickle down from major fashion houses or are represented in fashion shows.For our project, we believe these fashion houses are ground zero for trends.
                 
                 
                 Our project aims to close the gap between orgination of trends and consumer behavior. 
                 By pairing computer vision with time series forecasting, we help brands not only anticipate what's next—but readily react to what's already popular.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # How It Works Section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("🧠 How It Works")
        st.write("We built a system that blends runway insight with market data to deliver trend intelligence at scale:")
        
        # Process Steps
        steps = [
            ("🖼", "Runway Image Scraping", "Scrape and structure thousands of looks from global fashion weeks, organized by designer, season, and show. We have used the last three seasons for our project."),
            ("🧵", "Fashion Attribute Recognition", "Using CLIP-based and CNN models, we detect garments, patterns, colors, and styles—no manual tagging required."),
            ("📊", "Time Series Modeling", "We track how each attribute appears over time, layering in Google Trends data to gauge real-world consumer interest."),
            ("🔮", "Trend Forecasting", "Through models like ARIMA, Prophet, and LSTM, we predict which fashion elements are rising, peaking, or fading—helping brands plan ahead.")
        ]
        
        for icon, title, description in steps:
            st.markdown(f"""
                <div class="process-step">
                    <div class="process-icon">{icon}</div>
                    <div class="process-text">
                        <strong>{title}</strong><br>
                        {description}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    

    
    # Navigation
    st.markdown("""
    ### 🔍 Look at the data
    Use the sidebar to navigate to different sections of the analysis:
    - **Color Trends**: View color analysis results
    - **Style Analysis**: Explore garment classifications
    - **Trend Reports**: Access detailed trend analysis
    """)

    st.markdown('<div class="quote-box">In an age of micro-trends and macro shifts, we believe that AI is fashion\'s most powerful forecasting tool.</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            <div class="footer-content">
                ❤️ Created by Ananya, Kaamil & Cece <br>
                <i> With much gratitude for Professor Gerber for this opportunity </i><br>
                Built @ <b>Northeastern University</b>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
