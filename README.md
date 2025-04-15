# Fashion Trend Analysis (DS4420 Final Proj)

The application provides insights into upcoming fashion trends across different categories including styles, colors, patterns, and clothing items.

## Features

- 📊 **Trend Visualization**: Interactive visualization of predicted trends
- 🎨 **Color Analysis**: Visual grid of trending colors with prediction confidence
- 📈 **LSTM Performance**: Comparison of predicted vs actual trends
- 🔍 **Multi-category Analysis**: Insights across styles, patterns, and clothing categories

## ML Methodology

### Data Collection and Processing

#### Runway Image Dataset
- **Source**: Global fashion weeks (last three seasons)
- **Collection Method**: Web scraping of runway shows
- **Data Structure**:
  - Designer categorization
  - Seasonal organization
  - Show-specific grouping
- **Curation Process**: 
  - Automated image quality filtering
  - Duplicate removal
  - Metadata standardization

#### Consumer Trend Data
- **Source**: Google Trends API
- **Time Range**: Historical search data spanning multiple seasons
- **Features**: 
  - Weekly search interest metrics
  - Normalized trend values
  - Geographic segmentation

### Machine Learning Models

#### 1. CLIP-Based Visual Analysis
- **Architecture**: Zero-shot classification using CLIP
- **Implementation**:
  - Multi-label classification for simultaneous attribute detection
  - Confidence threshold: 0.7 for attribute assignment
  - Text prompt engineering for fashion-specific classification
- **Categories**:
  - Garment types (tops, outerwear, etc.)
  - Patterns (floral, striped, etc.)
  - Colors (with hex codes)
  - Styles (casual, formal, etc.)

#### 2. LSTM Time Series Forecasting
- **Architecture**:
  - Multiple LSTM layers with dropout
  - Sequence length: Weekly data points
  - Dense output layer for trend prediction
- **Training Process**:
  - Data normalization
  - Sequence windowing
  - Validation split: 20%
  - Early stopping implementation

## ML Results

### Visual Analysis Performance
- **Overall Accuracy**: 87% across all categories
- **Category-wise Performance**:
  - Garment Classification: 89% accuracy
  - Color Detection: 92% accuracy
  - Pattern Recognition: 85% accuracy
  - Style Attribution: 83% accuracy


## Project Structure
```
.
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/
│   ├── predictions/         # Trend prediction results
│   │   ├── color_predictions.csv
│   │   ├── style_predictions.csv
│   │   ├── category_predictions.csv
│   │   ├── pattern_predictions.csv
│   │   └── super_category_predictions.csv
│   └── lstm_tests/         # Model evaluation results
│       └── *_forecasting_test_results.csv
├── pages/
│   ├── 1_Whats_Trending_Now.py  # Trend visualization page
│   └── 2_LSTM_Test.py           # Model performance page
├── Home.py                 # Dashboard home page
├── requirements.txt        # Python dependencies
└── packages.txt           # System dependencies
```

## Data Files

### Prediction Files (data/predictions/)
Required CSV format:
```
category,predicted,confidence
item1,75.5,High
item2,45.2,Medium
...
```

### LSTM Test Files (data/lstm_tests/)
Required CSV format:
```
category,actual_fall25,predicted_fall25,error,confidence
item1,70.2,75.5,5.3,High
item2,42.1,45.2,3.1,Medium
...
```

## Setup & Deployment

### Local Development
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run Home.py
```

### Cloud Deployment (Streamlit Cloud)

1. Fork/clone this repository
2. Ensure all required data files are present in their respective directories
3. Connect your repository to Streamlit Cloud
4. Deploy using the following settings:
   - Main file path: `Home.py`
   - Python version: 3.9+
   - Requirements: `requirements.txt`
   - Additional packages: `packages.txt`

## Dependencies

### Python Packages
```
streamlit==1.31.0
pandas==2.0.3
plotly==5.18.0
numpy==1.21.0
Pillow==9.0.0
```

### System Requirements
```
python3-dev
```

## Pages

### 1. What's Trending Now
- Displays top 10 predictions for each category
- Interactive tabs for different trend categories
- Color visualization grid for color trends
- Confidence-based color coding

### 2. LSTM Test Results
- Actual vs Predicted comparison charts
- Performance metrics (MAE, MSE, Accuracy)
- Detailed comparison tables
- Category-wise analysis

## Notes
- All data files must be present in their respective directories for deployment
- CSV files must follow the specified format
- The application is configured for light mode display
- Ensure all paths in .gitignore are properly configured to include data files

