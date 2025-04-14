# Fashion Trend Prediction from Social Media and Web Data

## Project Overview
This project aims to predict emerging fashion trends by analyzing social media content and web-scraped data. By combining computer vision for image analysis and NLP for text processing, we'll identify patterns in fashion preferences and forecast upcoming trends.

### Core Components
1. **Image Classification and Segmentation Model**
   Developed a custom Convolutional Neural Network (CNN) using PyTorch to classify fashion product images based on:
   - Clothing type (e.g., T-shirts, hoodies, baggy pants)
   - Dominant color (e.g., red, black, green)
The model was trained on the Fashion Product Images Dataset from Kaggle:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data
Multi-task learning architecture with two parallel output heads:
   - One for predicting the clothing article type
   - One for predicting the dominant color attribute
The architecture includes:
   - 3 stacked convolutional layers for progressively deeper visual feature extraction
   - Two max-pooling layers to reduce spatial resolution and capture hierarchical patterns
   - Dropout layer to prevent overfitting
   - Two fully connected dense layers before branching into separate classification heads
Input preprocessing:
   - Resized images to 64×64 pixels
   - Normalized pixel values
   - Data augmentation (random flips, crops) to improve generalization
Model Outputs:
   - Top-1 and top-3 predicted labels for each task
   - Comparison with ground truth labels for evaluation
   - Accuracy and confusion matrix to measure classification performance
Interpretability Features:
   - Added Grad-CAM visualizations to highlight which regions of the image the CNN used for prediction
   - Helps verify that the model focuses on the clothing itself (e.g., torso area) rather than background noise
This model supports the broader goal of connecting visual fashion trends (seen in real-world images) to the temporal popularity of these items in digital culture (tracked via Google Trends).

2. **Data Collection Pipeline**
   - Web scraping from fashion blogs and social media platforms
   - Focus on platforms: Google Trends
   - Collection of both images and associated text content

3. **Trend Analysis System**
Conducted a time series analysis of public interest in trending fashion items using Google Trends data over a 5-year period.
Focused on keywords closely related to clothing categories analyzed by the CNN model, including: "baggy jeans", "cargo pants", "leather jacket"
Utilized the R package gtrendsR to fetch global web search interest data from Google Trends.
Workflow included:
   - Fetching normalized weekly interest data from 2019 to 2024
   - Exporting raw and cleaned time series data to CSV for reproducibility
   - Visualizing search interest over time using ggplot2 to highlight trends and seasonal spikes
performed classical time series decomposition to separate:
Trend — long-term directional movement (e.g., steady rise in interest for "baggy jeans")
Seasonal — repeating annual cycles (e.g., spikes during spring/summer or fashion weeks)
Residual — random fluctuation and noise not explained by trend or seasonality




## Model Architecture
```
├── Backbone
│   ├── SpineNet-143
│   └── Feature Pyramid Network (FPN)
├── Heads
│   ├── Mask R-CNN Components
│   │   ├── Region Proposal Network (RPN)
│   │   ├── Bounding Box Detection
│   │   └── Mask Segmentation
│   └── Custom Attribute Head
│       └── Fashion Attribute Classification
```

### Training Details
- **Base Model**: Mask R-CNN pre-trained on COCO dataset
- **Input Resolution**: 1280x1280 pixels
- **Loss Functions**:
  - Standard Mask R-CNN losses for segmentation
  - Focal Loss for attribute classification
- **Augmentation Pipeline**:
  - Random scaling (0.5x - 2.0x)
  - AutoAugment v3 policy (modified for mask support)
  - Custom augmentations for fashion domain

## Project Structure
```
fashion-trend-prediction/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Processed and cleaned data
│   └── scraped/                # Web-scraped data
│       └── runway/            # Runway show images and metadata
├── models/
│   ├── image_classifier/       # Mask R-CNN + SpineNet implementation
│   │   ├── backbone/          # SpineNet + FPN modules
│   │   ├── heads/            # RPN, bbox, mask, and attribute heads
│   │   └── utils/            # Training and inference utilities
│   └── trend_predictor/       # Trend prediction model
├── src/
│   ├── data_collection/       # Web scraping scripts
│   │   ├── runway_scraper.py  # Runway show image scraper
│   │   └── trend_analyzer.py  # Trend analysis from runway images
│   ├── data_processing/       # Data cleaning and preprocessing
│   ├── model_training/        # Training scripts and configs
│   │   ├── augmentation/     # Custom augmentation policies
│   │   ├── losses/          # Loss function implementations
│   │   └── metrics/         # Evaluation metrics
│   └── analysis/             # Trend analysis scripts
├── notebooks/                 # Jupyter notebooks for exploration
└── tests/                    # Unit tests
```

## Implementation Phases

### Phase 1: Project Setup and Data Collection (Current)
- [x] Set up project repository
- [ ] Download and process iMaterialist dataset
- [x] Set up web scraping infrastructure
  - [x] Implement RunwayScraper for nowfashion.com
  - [x] Add image URL extraction and downloading
  - [x] Implement trend analysis pipeline
- [ ] Define target categories and attributes

### Phase 2: Model Development
- [ ] Implement SpineNet-143 backbone
- [ ] Integrate Mask R-CNN architecture
- [ ] Add custom attribute classification head
- [ ] Implement training pipeline with augmentations

### Phase 3: Integration and Deployment
- [ ] Train model on iMaterialist dataset
- [ ] Implement inference pipeline
- [ ] Create visualization dashboard
- [ ] Deploy model for web-scraped data analysis

### Phase 4: Runway Image Processing
- [x] Implement runway show scraping
  - [x] Extract show URLs and metadata
  - [x] Download high-resolution runway images
  - [x] Organize images by designer and show
- [x] Develop trend analysis pipeline
  - [x] Process runway images with trained model
  - [x] Track trend statistics (appearances, designers)
  - [x] Generate trend visualizations and reports
- [ ] Integrate with social media analysis
  - [ ] Combine runway trends with social media data
  - [ ] Create unified trend prediction system

## Technical Requirements
- Python 3.x
- PyTorch/TensorFlow for deep learning
- Detectron2 for Mask R-CNN implementation
- BeautifulSoup/Scrapy for web scraping
- NLTK/Spacy for NLP
- Pandas for data processing
- Matplotlib/Seaborn for visualization

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the iMaterialist dataset
4. Set up the environment:
   ```bash
   pip install torch torchvision detectron2
   pip install -r requirements.txt
