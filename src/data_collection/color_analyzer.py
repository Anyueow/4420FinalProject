from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import logging
from sklearn.cluster import KMeans

class ColorAnalyzer:
    def __init__(self, n_colors: int, resize_size: int, min_area_percentage: float):
        self.n_colors = n_colors
        self.resize_size = resize_size
        self.min_area_percentage = min_area_percentage

    def is_skin_tone(self, color: np.ndarray) -> bool:
        # Implement your logic to determine if a color is a skin tone
        return False

    def process_image(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        """Process a single image and extract its colors."""
        try:
            # Read image with error handling
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logging.error(f"Could not read image: {image_path}")
                return None
                
            # Handle different image formats
            if img.dtype == np.int32:  # CV_32S
                # Convert 32-bit signed integer to 8-bit unsigned
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            elif img.dtype != np.uint8:
                # Convert any other format to 8-bit unsigned
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # If image has alpha channel, remove it
            if img.shape[-1] == 4:
                img = img[..., :3]
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (self.resize_size, self.resize_size))
            
            # Reshape image for clustering
            pixels = img.reshape(-1, 3)
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=self.n_colors, n_init=10, random_state=42)
            kmeans.fit(pixels)
            
            # Get the colors and their counts
            colors = kmeans.cluster_centers_.astype(int)
            counts = np.bincount(kmeans.labels_)
            
            # Calculate percentages
            total_pixels = len(pixels)
            percentages = (counts / total_pixels) * 100
            
            # Filter out skin tones and small areas
            results = []
            for color, count, percentage in zip(colors, counts, percentages):
                if not self.is_skin_tone(color) and percentage >= self.min_area_percentage:
                    hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                    results.append({
                        'color': hex_color,
                        'count': int(count),
                        'percentage': float(percentage)
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return None 