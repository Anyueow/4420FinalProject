# Fashion Trend Prediction and Analysis System

## Project Overview
This project implements a comprehensive fashion trend prediction system that combines computer vision, natural language processing, and time series analysis to forecast emerging fashion trends. The system analyzes runway show images, google search data, and historical trend data to provide actionable insights for the retail fashion industry.

## Data Flow and Architecture

### 1. Data Collection and Processing
The system processes data from multiple sources:

#### Runway Show Images
- Source: Fashion show images from major designers and retail products
- Collection: 
  - Automated scraping of runway images
  - Retail product images from Fashion Product Images Dataset (Kaggle)
- Processing:
  - Image classification using dual CNN approaches
  - Attribute extraction (style, category, pattern)
  - Color analysis using K-means clustering

#### Trend Data
- Source: Historical fashion data
- Collection: Structured data from fashion databases
- Processing:
  - Feature extraction
  - Temporal alignment
  - Trend normalization

### 2. Machine Learning Methods

#### A. Computer Vision Pipeline

1. **Retail Product Classification (CNN)**
   - Architecture from retail_cnn.ipynb:
     ```
     ├── Convolutional Layers
     │   ├── Conv2D (3 stacked layers)
     │   ├── MaxPooling2D (2 layers)
     │   └── Dropout (0.25)
     ├── Dense Layers
     │   ├── Fully Connected (512 units)
     │   ├── Dropout (0.5)
     │   └── Output Layers
     │       ├── Clothing Type (Multi-class)
     │       └── Color Classification
     ```
   - Training Details:
     - Input: 64x64 RGB images
     - Data Augmentation: Random flips, crops
     - Loss Functions:
       - Categorical Cross-entropy (clothing type)
       - Binary Cross-entropy (color attributes)
     - Metrics: Top-1 and Top-3 accuracy
     - Interpretability: Grad-CAM visualizations


2.. **Color Analysis**
   - Method: K-means clustering
   - Features:
     - Top 5 colors per designer
     - Top 10 colors per season
   - Output: Color trend predictions

#### B. Trend Analysis Pipeline
1. **Feature Frequency Analysis**
   - Calculates frequencies for:
     - Styles
     - Categories
     - Super Categories
     - Patterns
   - Temporal analysis across seasons (Fall24 → Spring25 → Fall25)

2. **Trend Momentum Analysis**
   - Method: Weighted moving average with momentum
   - Features:
     - Historical trend direction
     - Volatility measurement
     - Confidence scoring
   - Output: Spring26 predictions with confidence levels

### 3. Results and Interpretation

#### A. Computer Vision Results
- Retail CNN Performance:
  - Clothing Type Classification: 82% accuracy
  - Color Classification: 88% accuracy
  - Top-3 Accuracy: 94% for clothing type

- Color analysis precision: 90% for top color identification

#### B. Trend Analysis Results
1. **Style Trends**
   - Top predicted trends for Spring26:
     - Fur (133%, High confidence)
     - Techwear (18.1%, Increasing)
     - Relaxed Elegance (11.1%, Medium confidence)

2. **Category Trends**
   - Emerging categories:
     - Turtleneck Sweater (142%, High confidence)
     - Corset-style Belt (123%, High confidence)
     - Mesh Cover-up (Growing trend)

3. **Pattern Analysis**
   - Top patterns:
     - Long-pile Faux Fur (44.4%, Low confidence)
     - Open-weave Knit (16.7%, Medium confidence)
     - Shearling (16.7%, Medium confidence)

### 4. Project Structure
```
fashion-trend-prediction/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Processed and cleaned data
│   └── scraped/                # Web-scraped data
├── models/
│   ├── image_classifier/       # CNN and Mask R-CNN implementations
│   └── trend_predictor/        # Trend analysis models
├── notebooks/
│   └── retail_cnn.ipynb       # Retail product CNN implementation
├── webapp/
│   ├── pages/
│   │   ├── trend_analysis.py   # Trend visualization
│   │   └── color_analyzer.py   # Color trend analysis
└── src/
    ├── data_processing/        # Data cleaning scripts
    └── visualization/          # Plotting utilities
```

## Usage
1. Run the trend analysis:
   ```bash
   Rscript models/fashion_forecasting.R
   ```

2. Start the visualization dashboard:
   ```bash
   streamlit run webapp/pages/trend_analysis.py
   ```

## Dependencies
- Python 3.8+
- R 4.0+
- PyTorch
- TensorFlow
- Streamlit
- Plotly
- Tidyverse (R)

## Future Work
- Integration of social media trend data
- Real-time trend prediction
- Enhanced visualization capabilities
- Multi-modal trend analysis
