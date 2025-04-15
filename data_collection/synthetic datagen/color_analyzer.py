"""
Color analysis for runway images using K-means clustering and CSV integration.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, hex2color
import logging
import json

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

    def analyze_image_colors(self, image_path):
        """Analyze colors for a single image."""
        try:
            img = self.preprocess_image(image_path)
            if img is not None:
                colors = self.extract_colors(img)
                # Return top colors and their percentages
                return {
                    f'color_{i+1}': color[0] for i, color in enumerate(colors)
                }, {
                    f'color_{i+1}_percentage': round(color[1], 2) for i, color in enumerate(colors)
                }
            return {}, {}
        except Exception as e:
            logging.error(f"Error analyzing image colors {image_path}: {str(e)}")
            return {}, {}

    def process_fashion_labels(self, labels_path, output_path=None):
        """Add color analysis to existing fashion labels CSV."""
        try:
            # Read existing labels
            df = pd.read_csv(labels_path)
            logging.info(f"Processing {len(df)} images from fashion labels")

            # Initialize new columns for colors
            for i in range(1, self.n_colors + 1):
                df[f'color_{i}'] = ''
                df[f'color_{i}_percentage'] = 0.0

            # Process each image
            for idx, row in df.iterrows():
                image_path = row['image_path']
                colors, percentages = self.analyze_image_colors(image_path)
                
                # Update dataframe with color information
                for col, value in colors.items():
                    df.at[idx, col] = value
                for col, value in percentages.items():
                    df.at[idx, col] = value

                if idx % 10 == 0:
                    logging.info(f"Processed {idx + 1}/{len(df)} images")

            # Save updated CSV
            output_path = output_path or labels_path
            df.to_csv(output_path, index=False)
            logging.info(f"Saved updated fashion labels with colors to {output_path}")
            
            return df

        except Exception as e:
            logging.error(f"Error processing fashion labels: {str(e)}")
            return None

def main():
    """Main function to run color analysis pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description='Add color analysis to fashion labels')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to fashion_labels.csv')
    parser.add_argument('--output_csv', type=str, help='Path to save updated CSV (optional)')
    parser.add_argument('--n_colors', type=int, default=5, help='Number of dominant colors to extract')
    args = parser.parse_args()

    analyzer = ColorAnalyzer(n_colors=args.n_colors)
    analyzer.process_fashion_labels(args.input_csv, args.output_csv)

if __name__ == '__main__':
    main() 