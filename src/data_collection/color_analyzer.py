"""
Optimized color analysis for runway images using efficient clustering and parallel processing.
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.colors as mcolors
from collections import Counter, defaultdict
import logging
from typing import Dict, List, Tuple, Union
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import MiniBatchKMeans
import glob

class ColorAnalyzer:
    """Analyze colors in runway images using efficient clustering."""
    
    def __init__(self, 
                 n_colors: int = 5, 
                 resize_size: int = 250,
                 min_area_percentage: float = 2.0,
                 batch_size: int = 100,
                 n_jobs: int = 4):
        self.n_colors = n_colors
        self.resize_size = resize_size
        self.min_area_percentage = min_area_percentage
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        
        # Initialize MiniBatchKMeans for faster clustering
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_colors * 2,
            batch_size=batch_size,
            n_init=3,
            random_state=42
        )
        
        # Define skin tone thresholds in YCrCb space (more efficient)
        self.ycrcb_min = np.array([0, 133, 77], dtype=np.uint8)
        self.ycrcb_max = np.array([255, 173, 127], dtype=np.uint8)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for color analysis."""
        # Resize image while preserving aspect ratio
        h, w = image.shape[:2]
        if max(h, w) > self.resize_size:
            scale = self.resize_size / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def is_skin_tone(self, color: np.ndarray) -> bool:
        """Efficiently check if a color is a skin tone using YCrCb color space."""
        color_ycrcb = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_RGB2YCrCb)[0, 0]
        return np.all(color_ycrcb >= self.ycrcb_min) and np.all(color_ycrcb <= self.ycrcb_max)
    
    def extract_colors(self, image):
        """Extract dominant colors from an image using KMeans clustering."""
        # Resize image for faster processing
        image = cv2.resize(image, (self.resize_size, self.resize_size))
        
        # Reshape the image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Fit KMeans to the pixels
        self.kmeans.fit(pixels)
        
        # Get the cluster centers and normalize them to be between 0 and 255
        centers = np.clip(self.kmeans.cluster_centers_, 0, 255)
        centers = centers.astype(np.uint8)
        
        # Count pixels in each cluster
        labels = self.kmeans.labels_
        counts = np.bincount(labels)
        
        # Calculate percentages
        percentages = counts / len(pixels) * 100
        
        # Convert colors to hex format
        colors = ['#%02x%02x%02x' % (center[0], center[1], center[2]) 
                 for center in centers]
        
        return colors, percentages
    
    def process_image_batch(self, image_paths: List[str]) -> Dict[str, Dict[str, Union[int, float]]]:
        """Process a batch of images and aggregate color statistics."""
        color_data = defaultdict(lambda: {'count': 0, 'total_percentage': 0.0})
        total_images = 0
        
        try:
            # Process images in batches
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                batch_results = {}
                
                for path in batch_paths:
                    try:
                        # Read and process image
                        img = cv2.imread(path)
                        if img is None:
                            logging.warning(f"Failed to read image: {path}")
                            continue
                            
                        # Preprocess image
                        img = self.preprocess_image(img)
                        
                        # Extract colors
                        colors, percentages = self.extract_colors(img)
                        
                        # Filter colors based on area and skin tone
                        filtered_results = []
                        for color, percentage in zip(colors, percentages):
                            if percentage >= self.min_area_percentage:
                                # Convert hex to BGR for skin tone check
                                color_bgr = np.array([int(color[i:i+2], 16) for i in (5,3,1)])
                                if not self.is_skin_tone(color_bgr):
                                    filtered_results.append((color, percentage))
                        
                        batch_results[path] = filtered_results
                        total_images += 1
                        
                    except Exception as e:
                        logging.error(f"Error processing image {path}: {str(e)}")
                        continue
                
                # Aggregate results
                for results in batch_results.values():
                    for color, percentage in results:
                        color_data[color]['count'] += 1
                        color_data[color]['total_percentage'] += percentage
        
        except Exception as e:
            logging.error(f"Error in batch processing: {str(e)}")
        
        # Calculate average percentages
        if total_images > 0:
            for color_stats in color_data.values():
                color_stats['average_percentage'] = color_stats['total_percentage'] / color_stats['count']
        
        return dict(color_data)
    
    def analyze_collection(self, image_dir: str) -> Dict[str, Dict[str, float]]:
        """Analyze colors in a collection of images."""
        try:
            # Get all image paths
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            
            if not image_paths:
                logging.warning(f"No images found in directory: {image_dir}")
                return {}
            
            # Process images in batches
            color_data = self.process_image_batch(image_paths)
            
            # Sort colors by count and get top N
            sorted_colors = sorted(
                color_data.items(),
                key=lambda x: (x[1]['count'], x[1]['average_percentage']),
                reverse=True
            )[:self.n_colors]
            
            # Format results
            results = {
                'total_images': len(image_paths),
                'colors': {}
            }
            
            for color, stats in sorted_colors:
                results['colors'][color] = {
                    'count': stats['count'],
                    'percentage': stats['average_percentage']
                }
            
            return results
        
        except Exception as e:
            logging.error(f"Error analyzing collection {image_dir}: {str(e)}")
            return {}
    
    def create_color_dictionary(self, season_dir: str) -> Dict[str, Dict]:
        """Create a dictionary of color data for all collections in a season."""
        try:
            # Get all designer directories
            designer_dirs = [d for d in os.listdir(season_dir) 
                           if os.path.isdir(os.path.join(season_dir, d))]
            
            if not designer_dirs:
                logging.warning(f"No designer directories found in {season_dir}")
                return {}
            
            color_dict = {}
            total_collections = len(designer_dirs)
            
            for idx, designer in enumerate(designer_dirs, 1):
                designer_path = os.path.join(season_dir, designer)
                logging.info(f"Processing collection {idx}/{total_collections}: {designer}")
                
                # Analyze collection
                collection_data = self.analyze_collection(designer_path)
                
                if collection_data:
                    color_dict[designer] = {
                        'total_images': collection_data['total_images'],
                        'colors': collection_data['colors']
                    }
                    
                    # Log results
                    logging.info(f"Analyzed {collection_data['total_images']} images from {designer}")
                    logging.info("Top 5 colors:")
                    for color, stats in list(collection_data['colors'].items())[:5]:
                        logging.info(f"  {color}: {stats['count']} occurrences ({stats['percentage']:.2f}%)")
                else:
                    logging.warning(f"No valid data for collection: {designer}")
            
            # Save results to JSON
            output_path = os.path.join(season_dir, 'color_dictionary.json')
            with open(output_path, 'w') as f:
                json.dump(color_dict, f, indent=2)
            logging.info(f"Color dictionary saved to {output_path}")
            
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