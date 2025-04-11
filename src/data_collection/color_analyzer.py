"""
Color analysis for runway images using K-Means clustering.
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.colors as mcolors
from collections import Counter, defaultdict
import logging
from typing import Dict, List, Tuple
import json
import cv2

class ColorAnalyzer:
    """Analyze colors in runway images using K-Means clustering."""
    
    def __init__(self, 
                 n_colors: int = 5, 
                 resize_size: int = 250,
                 min_area_percentage: float = 5.0,
                 skin_tone_threshold: float = 0.1):
        self.n_colors = n_colors
        self.resize_size = resize_size
        self.min_area_percentage = min_area_percentage
        self.skin_tone_threshold = skin_tone_threshold
        
        # Define skin tone thresholds using YCrCb color space
        self.ycrcb_skin_min = np.array([0, 133, 77], dtype=np.uint8)
        self.ycrcb_skin_max = np.array([255, 173, 127], dtype=np.uint8)
        
        # Define refined HSV thresholds
        self.hsv_skin_lower = np.array([0, int(0.1 * 255), int(0.4 * 255)], dtype=np.uint8)
        self.hsv_skin_upper = np.array([int(0.28 * 180), int(0.6 * 255), 255], dtype=np.uint8)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image while preserving aspect ratio."""
        h, w = image.shape[:2]
        if max(h, w) > self.resize_size:
            scale = self.resize_size / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image
    
    def is_skin_tone(self, color: Tuple[int, int, int]) -> bool:
        """Check if a color is within skin tone ranges using both YCrCb and HSV."""
        # Convert single color to a 1x1 image
        color_array = np.uint8([[color]])
        
        # YCrCb test
        ycrcb = cv2.cvtColor(color_array, cv2.COLOR_RGB2YCrCb)[0][0]
        ycrcb_mask = np.all(ycrcb >= self.ycrcb_skin_min) and np.all(ycrcb <= self.ycrcb_skin_max)
        
        # HSV test
        hsv = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)[0][0]
        hsv_mask = np.all(hsv >= self.hsv_skin_lower) and np.all(hsv <= self.hsv_skin_upper)
        
        return ycrcb_mask or hsv_mask
    
    def extract_colors(self, image_path: str) -> List[Tuple[Tuple[int, int, int], float]]:
        """Extract dominant colors from an image using K-Means clustering."""
        try:
            # Open and convert image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Resize for efficiency
            img_array = self.resize_image(img_array)
            
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Flatten the image into a list of pixels
            pixels = img_rgb.reshape((-1, 3))
            
            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=self.n_colors * 2, 
                          random_state=42,
                          n_init=10)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_.astype(np.uint8)
            
            # Count pixels per cluster
            counts = np.bincount(labels)
            total_pixels = pixels.shape[0]
            
            # Process clusters
            color_percentages = []
            for center, count in zip(centers, counts):
                color = tuple(int(c) for c in center)
                percentage = (count / total_pixels) * 100
                
                # Skip skin tones and small clusters
                if not self.is_skin_tone(color) and percentage >= self.min_area_percentage:
                    color_percentages.append((color, percentage))
            
            # Sort by percentage and take top N
            color_percentages.sort(key=lambda x: x[1], reverse=True)
            return color_percentages[:self.n_colors]
            
        except Exception as e:
            logging.error(f"Error extracting colors from {image_path}: {str(e)}")
            return []
    
    def analyze_collection(self, image_dir: str) -> Dict[str, Dict[str, float]]:
        """Analyze colors in a collection of images."""
        color_data = defaultdict(lambda: {'count': 0, 'total_percentage': 0.0})
        
        try:
            # Get all images in the directory
            for filename in os.listdir(image_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, filename)
                    colors = self.extract_colors(image_path)
                    
                    # Aggregate color data
                    for color, percentage in colors:
                        hex_color = mcolors.to_hex(np.array(color)/255)
                        color_data[hex_color]['count'] += 1
                        color_data[hex_color]['total_percentage'] += percentage
            
            # Convert to final format
            result = {}
            for color, data in color_data.items():
                avg_percentage = data['total_percentage'] / data['count']
                result[color] = {
                    'count': data['count'],
                    'average_percentage': round(avg_percentage, 2)
                }
            
            return dict(sorted(result.items(), 
                             key=lambda x: x[1]['average_percentage'], 
                             reverse=True))
            
        except Exception as e:
            logging.error(f"Error analyzing collection {image_dir}: {str(e)}")
            return {}
    
    def create_color_dictionary(self, season_dir: str) -> Dict[str, Dict[str, List[Dict[str, any]]]]:
        """Create a comprehensive color dictionary for all collections."""
        color_dict = {}
        
        try:
            # Get all designer directories
            for designer_dir in os.listdir(season_dir):
                designer_path = os.path.join(season_dir, designer_dir)
                if not os.path.isdir(designer_path):
                    continue
                    
                logging.info(f"Analyzing colors for {designer_dir}...")
                
                # Analyze collection colors
                colors = self.analyze_collection(designer_path)
                if not colors:
                    continue
                
                # Get top colors
                top_colors = []
                for color, data in list(colors.items())[:5]:
                    top_colors.append({
                        'color': color,
                        'count': data['count'],
                        'average_percentage': data['average_percentage'],
                        'appears_in_images': data['count']
                    })
                
                # Count total images
                total_images = len([f for f in os.listdir(designer_path) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                # Add to dictionary
                color_dict[designer_dir] = {
                    'top_colors': top_colors,
                    'total_images': total_images
                }
            
            # Save results
            results_path = os.path.join(season_dir, 'color_dictionary.json')
            with open(results_path, 'w') as f:
                json.dump(color_dict, f, indent=2)
            
            logging.info(f"Color dictionary saved to {results_path}")
            return color_dict
            
        except Exception as e:
            logging.error(f"Error creating color dictionary: {str(e)}")
            return {}
    
    def analyze_season(self, season_dir: str) -> Dict[str, Dict[str, int]]:
        """Analyze colors for all collections in a season."""
        season_colors = {}
        
        try:
            # Get all designer directories
            for designer_dir in os.listdir(season_dir):
                designer_path = os.path.join(season_dir, designer_dir)
                if os.path.isdir(designer_path):
                    # Analyze collection colors
                    colors = self.analyze_collection(designer_path)
                    if colors:
                        season_colors[designer_dir] = colors
            
            # Save results
            results_path = os.path.join(season_dir, 'color_analysis.json')
            with open(results_path, 'w') as f:
                json.dump(season_colors, f, indent=2)
            
            return season_colors
            
        except Exception as e:
            logging.error(f"Error analyzing season {season_dir}: {str(e)}")
            return {}
    
    def get_top_colors(self, color_counts: Dict[str, int], n: int = 5) -> List[Tuple[str, int]]:
        """Get the top N most frequent colors."""
        return list(color_counts.items())[:n] 