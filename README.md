# Fashion Trend Prediction from Social Media and Web Data

## Project Overview
This project aims to predict emerging fashion trends by analyzing social media content and web-scraped data. By combining computer vision for image analysis and NLP for text processing, we'll identify patterns in fashion preferences and forecast upcoming trends.

### Core Components
1. **Image Classification and Segmentation Model**
   - Using the iMaterialist Fashion 2020 dataset to train a model for:
     - Garment segmentation and classification
     - Attribute detection (colors, patterns, styles)
   - Based on Mask R-CNN architecture with SpineNet-143 backbone
   - Multi-task learning approach for both segmentation and attribute classification

2. **Data Collection Pipeline**
   - Web scraping from fashion blogs and social media platforms
   - Focus on platforms: X (Twitter) and fashion blogs (e.g., Vogue)
   - Collection of both images and associated text content

3. **Trend Analysis System**
   - NLP processing of captions and comments
   - Computer vision analysis of images
   - Time-series analysis for trend prediction

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
