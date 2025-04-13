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
from typing import Dict, List, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import MiniBatchKMeans

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
    
    def extract_colors(self, image_path: str) -> List[Tuple[Tuple[int, int, int], float]]:
        """Extract dominant colors from an image using MiniBatchKMeans."""
        try:
            # Read image directly with OpenCV for better performance
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Failed to read image: {image_path}")
                return []
            
            # Preprocess image
            img = self.preprocess_image(img)
            
            # Reshape for clustering
            pixels = img.reshape(-1, 3)
            
            # Fit MiniBatchKMeans
            self.kmeans.partial_fit(pixels)
            labels = self.kmeans.predict(pixels)
            centers = self.kmeans.cluster_centers_.astype(np.uint8)
            
            # Count pixels per cluster
            counts = np.bincount(labels)
            total_pixels = pixels.shape[0]
            
            # Process clusters
            color_percentages = []
            for center, count in zip(centers, counts):
                percentage = (count / total_pixels) * 100
                if percentage >= self.min_area_percentage and not self.is_skin_tone(center):
                    color_percentages.append((tuple(center), percentage))
            
            # Sort by percentage and return top N
            color_percentages.sort(key=lambda x: x[1], reverse=True)
            return color_percentages[:self.n_colors]
            
        except Exception as e:
            logging.error(f"Error extracting colors from {image_path}: {str(e)}")
            return []
    
    def process_image_batch(self, image_paths: List[str]) -> Dict[str, List[Tuple[Tuple[int, int, int], float]]]:
        """Process a batch of images in parallel."""
        results = {}
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_path = {
                executor.submit(self.extract_colors, path): path 
                for path in image_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    logging.error(f"Error processing {path}: {str(e)}")
        return results
    
    def analyze_collection(self, image_dir: str) -> Dict[str, Dict[str, float]]:
        """Analyze colors in a collection using batch processing."""
        color_data = defaultdict(lambda: {'count': 0, 'total_percentage': 0.0})
        
        try:
            # Get all image paths
            image_paths = [
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            # Process images in batches
            for i in range(0, len(image_paths), self.batch_size):
                batch = image_paths[i:i + self.batch_size]
                batch_results = self.process_image_batch(batch)
                
                # Aggregate results
                for colors in batch_results.values():
                    for color, percentage in colors:
                        hex_color = mcolors.to_hex(np.array(color)/255)
                        color_data[hex_color]['count'] += 1
                        color_data[hex_color]['total_percentage'] += percentage
            
            # Calculate averages and format results
            result = {}
            for color, data in color_data.items():
                if data['count'] > 0:
                    result[color] = {
                        'count': data['count'],
                        'average_percentage': round(data['total_percentage'] / data['count'], 2)
                    }
            
            return dict(sorted(result.items(), 
                             key=lambda x: x[1]['average_percentage'], 
                             reverse=True))
            
        except Exception as e:
            logging.error(f"Error analyzing collection {image_dir}: {str(e)}")
            return {}
    
    def create_color_dictionary(self, season_dir: str) -> Dict[str, Dict[str, List[Dict[str, any]]]]:
        """Create a comprehensive color dictionary using parallel processing."""
        color_dict = {}
        
        try:
            # Get all designer directories
            designer_dirs = [d for d in os.listdir(season_dir) 
                           if os.path.isdir(os.path.join(season_dir, d))]
            
            # Process collections in parallel
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_designer = {
                    executor.submit(self.analyze_collection, 
                                  os.path.join(season_dir, d)): d 
                    for d in designer_dirs
                }
                
                for future in as_completed(future_to_designer):
                    designer = future_to_designer[future]
                    try:
                        colors = future.result()
                        if colors:
                            # Get top colors
                            top_colors = []
                            for color, data in list(colors.items())[:5]:
                                top_colors.append({
                                    'color': color,
                                    'count': data['count'],
                                    'average_percentage': data['average_percentage']
                                })
                            
                            # Count total images
                            designer_path = os.path.join(season_dir, designer)
                            total_images = len([f for f in os.listdir(designer_path)
                                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            
                            color_dict[designer] = {
                                'top_colors': top_colors,
                                'total_images': total_images
                            }
                    except Exception as e:
                        logging.error(f"Error processing {designer}: {str(e)}")
            
            # Save results
            results_path = os.path.join(season_dir, 'color_dictionary.json')
            with open(results_path, 'w') as f:
                json.dump(color_dict, f, indent=2)
            
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