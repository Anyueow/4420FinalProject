"""
Fashion trend analyzer using FashionCLIP for clothing type detection.
"""

import os
import logging
import json
from typing import Dict, List, Tuple
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

class TrendAnalyzer:
    """Analyze fashion trends using FashionCLIP."""
    
    def __init__(self):
        """Initialize the FashionCLIP model and processor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load FashionCLIP model and processor
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        
        # Define fashion categories for zero-shot classification
        self.categories = [
            "dress", "coat", "suit", "jacket", "blazer",
            "blouse", "shirt", "t-shirt", "sweater", "cardigan",
            "pants", "jeans", "skirt", "shorts",
            "boots", "sneakers", "heels", "sandals",
            "bag", "handbag", "clutch", "backpack",
            "accessories", "jewelry", "sunglasses"
        ]
        
        # Define style attributes
        self.styles = [
            "casual", "formal", "elegant", "streetwear", "minimalist",
            "bohemian", "vintage", "modern", "classic", "avant-garde",
            "feminine", "masculine", "unisex", "luxury", "sustainable"
        ]
        
        # Define patterns and textures
        self.patterns = [
            "floral", "striped", "plaid", "checkered", "polka dot",
            "geometric", "abstract", "animal print", "solid color",
            "metallic", "sequined", "embroidered", "lace", "leather"
        ]
    
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze a single image for clothing type, style, and patterns."""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            inputs = self.processor(
                images=image,
                text=self.categories + self.styles + self.patterns,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            # Get top predictions for each category
            n_categories = len(self.categories)
            n_styles = len(self.styles)
            
            # Split predictions
            category_probs = probs[:n_categories]
            style_probs = probs[n_categories:n_categories + n_styles]
            pattern_probs = probs[n_categories + n_styles:]
            
            # Get top predictions
            top_category = self.categories[category_probs.argmax()]
            top_style = self.styles[style_probs.argmax()]
            top_pattern = self.patterns[pattern_probs.argmax()]
            
            return {
                'category': {
                    'label': top_category,
                    'confidence': float(category_probs.max())
                },
                'style': {
                    'label': top_style,
                    'confidence': float(style_probs.max())
                },
                'pattern': {
                    'label': top_pattern,
                    'confidence': float(pattern_probs.max())
                }
            }
            
        except Exception as e:
            logging.error(f"Error analyzing image {image_path}: {str(e)}")
            return None
    
    def analyze_collection(self, collection_dir: str) -> Dict:
        """Analyze all images in a collection directory."""
        collection_stats = defaultdict(lambda: defaultdict(int))
        total_images = 0
        
        try:
            # Process all images in directory
            for filename in os.listdir(collection_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(collection_dir, filename)
                    results = self.analyze_image(image_path)
                    
                    if results:
                        # Update statistics
                        collection_stats['categories'][results['category']['label']] += 1
                        collection_stats['styles'][results['style']['label']] += 1
                        collection_stats['patterns'][results['pattern']['label']] += 1
                        total_images += 1
            
            # Calculate percentages
            stats = {}
            for category in ['categories', 'styles', 'patterns']:
                stats[category] = {
                    item: {
                        'count': count,
                        'percentage': (count / total_images) * 100 if total_images > 0 else 0
                    }
                    for item, count in collection_stats[category].items()
                }
            
            stats['total_images'] = total_images
            
            # Save results
            output_file = os.path.join(collection_dir, 'trend_analysis.json')
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error analyzing collection {collection_dir}: {str(e)}")
            return None
    
    def analyze_season(self, season_dir: str) -> Dict:
        """Analyze all collections in a season directory."""
        season_stats = {}
        
        try:
            # Process each designer collection
            for designer in os.listdir(season_dir):
                designer_path = os.path.join(season_dir, designer)
                if os.path.isdir(designer_path):
                    logging.info(f"Analyzing collection: {designer}")
                    stats = self.analyze_collection(designer_path)
                    if stats:
                        season_stats[designer] = stats
            
            # Save season results
            output_file = os.path.join(season_dir, 'season_trends.json')
            with open(output_file, 'w') as f:
                json.dump(season_stats, f, indent=2)
            
            return season_stats
            
        except Exception as e:
            logging.error(f"Error analyzing season {season_dir}: {str(e)}")
            return None