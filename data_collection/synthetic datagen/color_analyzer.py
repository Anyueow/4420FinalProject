"""
Color analysis for runway images using K-means clustering.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import logging

logging.basicConfig(level=logging.INFO)

class ColorAnalyzer:
    def __init__(self, n_colors=5, resize_size=300):
        self.n_colors = n_colors
        self.resize_size = resize_size
        
    def preprocess_image(self, image_path):
        """Load and preprocess an image."""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for faster processing
            h, w = img.shape[:2]
            if max(h, w) > self.resize_size:
                scale = self.resize_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            return img
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None

    def extract_colors(self, img):
        """Extract dominant colors using K-means clustering."""
        try:
            # Reshape image
            pixels = img.reshape(-1, 3)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get colors and their percentages
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            counts = np.bincount(labels)
            percentages = (counts / len(pixels)) * 100
            
            # Convert to hex colors and create result list
            results = []
            for color, percentage in zip(colors, percentages):
                hex_color = rgb2hex(np.clip(color/255, 0, 1))
                results.append((hex_color, percentage))
            
            # Sort by percentage
            results.sort(key=lambda x: x[1], reverse=True)
            return results
            
        except Exception as e:
            logging.error(f"Error extracting colors: {str(e)}")
            return []

    def analyze_designer_collection(self, collection_path, season):
        """Analyze colors for a single designer collection."""
        try:
            collection_path = Path(collection_path)
            designer = collection_path.name
            
            # Get all images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(collection_path.glob(ext)))
            
            if not image_files:
                logging.warning(f"No images found in {collection_path}")
                return pd.DataFrame()
            
            # Process each image
            all_colors = []
            for img_path in image_files:
                img = self.preprocess_image(img_path)
                if img is not None:
                    colors = self.extract_colors(img)
                    all_colors.extend(colors)
            
            # Aggregate colors
            color_data = []
            if all_colors:
                color_counter = Counter()
                for color, percentage in all_colors:
                    color_counter[color] += percentage
                
                # Get top colors
                for color, total_percentage in color_counter.most_common(self.n_colors):
                    color_data.append({
                        'season': season,
                        'designer': designer,
                        'color': color,
                        'percentage': round(total_percentage / len(image_files), 2)
                    })
            
            return pd.DataFrame(color_data)
            
        except Exception as e:
            logging.error(f"Error analyzing collection {collection_path}: {str(e)}")
            return pd.DataFrame()

    def analyze_season(self, season_path):
        """Analyze all collections in a season."""
        try:
            season_path = Path(season_path)
            season = season_path.name
            logging.info(f"Analyzing season: {season}")
            
            # Get all designer directories
            designer_dirs = [d for d in season_path.iterdir() if d.is_dir()]
            
            if not designer_dirs:
                logging.warning(f"No designer directories found in {season_path}")
                return pd.DataFrame(), pd.DataFrame()
            
            # Analyze each designer
            all_designer_data = []
            for designer_dir in designer_dirs:
                designer_df = self.analyze_designer_collection(designer_dir, season)
                if not designer_df.empty:
                    all_designer_data.append(designer_df)
            
            if not all_designer_data:
                return pd.DataFrame(), pd.DataFrame()
            
            # Combine designer data
            designer_data = pd.concat(all_designer_data, ignore_index=True)
            
            # Calculate season summary
            season_data = (designer_data.groupby('color')['percentage']
                         .mean()
                         .reset_index()
                         .sort_values('percentage', ascending=False)
                         .head(10))
            season_data['season'] = season
            
            return designer_data, season_data
            
        except Exception as e:
            logging.error(f"Error analyzing season {season_path}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def save_results(self, designer_data, season_data, output_dir):
        """Save analysis results to CSV files."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not designer_data.empty:
                designer_data.to_csv(output_dir / 'designer_colors.csv', index=False)
                logging.info(f"Saved designer colors to {output_dir / 'designer_colors.csv'}")
            
            if not season_data.empty:
                season_data.to_csv(output_dir / 'season_colors.csv', index=False)
                logging.info(f"Saved season colors to {output_dir / 'season_colors.csv'}")
                
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")

    def plot_color_distribution(self, data, title):
        """Plot color distribution."""
        try:
            plt.figure(figsize=(12, 4))
            colors = data['color'].tolist()
            percentages = data['percentage'].tolist()
            
            plt.bar(range(len(colors)), percentages, color=colors)
            plt.title(title)
            plt.xticks(range(len(colors)), colors, rotation=45)
            plt.ylabel('Percentage (%)')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"Error plotting colors: {str(e)}") 