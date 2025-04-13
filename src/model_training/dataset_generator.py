"""
Generate labeled dataset using FashionCLIP for pseudo-labeling with detailed fashion categories.
"""

import os
import logging
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict

class DatasetGenerator:
    """Generate labeled dataset using FashionCLIP with detailed fashion categories."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize FashionCLIP model and detailed fashion categories."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        
        # Load FashionCLIP
        logging.info("Loading FashionCLIP model...")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        
        # Define hierarchical categories
        self.categories = {
            'tops': [
                'blouse', 'shirt', 't-shirt', 'tank top', 'sweater', 'cardigan',
                'hoodie', 'crop top', 'corset top', 'camisole'
            ],
            'outerwear': [
                'coat', 'jacket', 'blazer', 'parka', 'cape', 'vest', 'windbreaker'
            ],
            'bottoms': [
                'pants', 'jeans', 'skirt', 'shorts', 'leggings', 'culottes',
                'jumpsuit', 'romper'
            ],
            'dresses': [
                'dress', 'gown', 'sundress', 'knit dress', 'shirt dress', 
                'corset dress'
            ],
            'footwear': [
                'boots', 'sneakers', 'heels', 'sandals', 'flats', 'clogs',
                'espadrilles', 'slippers'
            ],
            'bags': [
                'handbag', 'clutch', 'backpack', 'crossbody', 'baguette', 
                'belt bag', 'oversized tote', 'micro bag'
            ],
            'accessories': [
                'jewelry', 'sunglasses', 'hat', 'scarf', 'belt', 'gloves',
                'hair accessories', 'socks', 'tights', 'watch', 'umbrella'
            ],
            'intimates': [
                'bra', 'underwear', 'bodysuit', 'lingerie', 'swimsuit', 'cover-up'
            ],
            'athleisure': [
                'sports bra', 'activewear top', 'activewear bottoms', 'tracksuit'
            ]
        }
        
        # Define style categories
        self.styles = {
            'core_aesthetics': [
                'casual', 'formal', 'elegant', 'streetwear', 'minimalist',
                'bohemian', 'vintage', 'modern', 'classic', 'avant-garde'
            ],
            'gender_identity': [
                'feminine', 'masculine', 'unisex', 'androgynous'
            ],
            'subcultures': [
                'preppy', 'grunge', 'punk', 'goth', 'cottagecore', 'balletcore',
                'gorpcore', 'techwear', 'normcore', 'kidcore'
            ],
            'occasion': [
                'party', 'resort', 'workwear', 'loungewear', 'festival'
            ],
            'value_driven': [
                'luxury', 'sustainable', 'artisanal', 'slow fashion', 'fast fashion'
            ],
            'micro_aesthetics': [
                'quiet luxury', 'loud luxury', 'mob wife', 'coastal grandma',
                'barbiecore', 'dopamine dressing'
            ]
        }
        
        # Define pattern categories
        self.patterns = {
            'classic_patterns': [
                'floral', 'striped', 'plaid', 'checkered', 'polka dot',
                'geometric', 'abstract', 'animal print'
            ],
            'textures': [
                'solid color', 'metallic', 'sequined', 'embroidered', 'lace',
                'leather', 'suede', 'velvet', 'shearling', 'corduroy', 'denim',
                'knit', 'quilted', 'mesh', 'tweed'
            ],
            'innovative': [
                'upcycled', 'organic', 'recycled', 'biofabricated', 'tie-dye',
                'ombré', 'colorblock', 'digital print', 'faux fur'
            ],
            'cultural': [
                'tapestry', 'batik', 'ikat', 'paisley', 'brocade', 'damask'
            ]
        }
        
        # Flatten categories for CLIP
        self.flat_categories = []
        self.category_to_super = {}
        
        # Process main categories
        for super_category, sub_categories in self.categories.items():
            self.flat_categories.extend(sub_categories)
            for category in sub_categories:
                self.category_to_super[category] = super_category
        
        # Process styles
        self.flat_styles = []
        self.style_to_super = {}
        for super_style, sub_styles in self.styles.items():
            self.flat_styles.extend(sub_styles)
            for style in sub_styles:
                self.style_to_super[style] = super_style
        
        # Process patterns
        self.flat_patterns = []
        self.pattern_to_super = {}
        for super_pattern, sub_patterns in self.patterns.items():
            self.flat_patterns.extend(sub_patterns)
            for pattern in sub_patterns:
                self.pattern_to_super[pattern] = super_pattern
    
    def process_image(self, image_path: str) -> dict:
        """Process a single image and return its classifications."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Get category prediction
            inputs = self.processor(
                images=image,
                text=self.flat_categories,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                category_probs = outputs.logits_per_image.softmax(dim=1)[0]
            
            # Get style prediction
            inputs = self.processor(
                images=image,
                text=self.flat_styles,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                style_probs = outputs.logits_per_image.softmax(dim=1)[0]
            
            # Get pattern prediction
            inputs = self.processor(
                images=image,
                text=self.flat_patterns,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                pattern_probs = outputs.logits_per_image.softmax(dim=1)[0]
            
            # Get top predictions
            max_category_prob, max_category_idx = torch.max(category_probs, dim=0)
            max_style_prob, max_style_idx = torch.max(style_probs, dim=0)
            max_pattern_prob, max_pattern_idx = torch.max(pattern_probs, dim=0)
            
            # Only return results if category confidence meets threshold
            if float(max_category_prob) >= self.confidence_threshold:
                predicted_category = self.flat_categories[max_category_idx]
                predicted_style = self.flat_styles[max_style_idx]
                predicted_pattern = self.flat_patterns[max_pattern_idx]
                
                return {
                    'path': image_path,
                    'category': {
                        'label': predicted_category,
                        'super_category': self.category_to_super[predicted_category],
                        'confidence': float(max_category_prob)
                    },
                    'style': {
                        'label': predicted_style,
                        'super_style': self.style_to_super[predicted_style],
                        'confidence': float(max_style_prob)
                    },
                    'pattern': {
                        'label': predicted_pattern,
                        'super_pattern': self.pattern_to_super[predicted_pattern],
                        'confidence': float(max_pattern_prob)
                    }
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None
    
    def generate_dataset(self, 
                        source_dir: str, 
                        output_dir: str,
                        n_jobs: int = 4) -> dict:
        """Generate labeled dataset from source images."""
        try:
            # Create output directory structure
            os.makedirs(output_dir, exist_ok=True)
            
            # Create category directories
            for super_cat in self.categories.keys():
                os.makedirs(os.path.join(output_dir, 'categories', super_cat), exist_ok=True)
            
            # Create style directories
            for super_style in self.styles.keys():
                os.makedirs(os.path.join(output_dir, 'styles', super_style), exist_ok=True)
            
            # Create pattern directories
            for super_pattern in self.patterns.keys():
                os.makedirs(os.path.join(output_dir, 'patterns', super_pattern), exist_ok=True)
            
            # Get all image paths
            image_paths = []
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(root, file))
            
            if not image_paths:
                logging.error(f"No images found in {source_dir}")
                return {}
            
            logging.info(f"Found {len(image_paths)} images")
            
            # Process images in parallel
            results = []
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(self.process_image, path) 
                          for path in image_paths]
                
                for future in tqdm(futures, desc="Processing images"):
                    result = future.result()
                    if result:
                        results.append(result)
            
            # Organize results
            dataset_info = {
                'total_images': len(image_paths),
                'labeled_images': len(results),
                'categories': defaultdict(int),
                'super_categories': defaultdict(int),
                'styles': defaultdict(int),
                'super_styles': defaultdict(int),
                'patterns': defaultdict(int),
                'super_patterns': defaultdict(int)
            }
            
            # Copy images and update statistics
            for result in results:
                src_path = result['path']
                filename = os.path.basename(src_path)
                
                # Handle category
                category = result['category']['label']
                super_category = result['category']['super_category']
                dataset_info['categories'][category] += 1
                dataset_info['super_categories'][super_category] += 1
                dst_path = os.path.join(output_dir, 'categories', super_category, filename)
                shutil.copy2(src_path, dst_path)
                
                # Handle style
                style = result['style']['label']
                super_style = result['style']['super_style']
                dataset_info['styles'][style] += 1
                dataset_info['super_styles'][super_style] += 1
                dst_path = os.path.join(output_dir, 'styles', super_style, filename)
                shutil.copy2(src_path, dst_path)
                
                # Handle pattern
                pattern = result['pattern']['label']
                super_pattern = result['pattern']['super_pattern']
                dataset_info['patterns'][pattern] += 1
                dataset_info['super_patterns'][super_pattern] += 1
                dst_path = os.path.join(output_dir, 'patterns', super_pattern, filename)
                shutil.copy2(src_path, dst_path)
            
            # Save dataset info
            info_path = os.path.join(output_dir, 'dataset_info.json')
            with open(info_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            logging.info(f"\nDataset generated with {len(results)} labeled images")
            logging.info(f"Dataset information saved to {info_path}")
            
            return dataset_info
            
        except Exception as e:
            logging.error(f"Error generating dataset: {str(e)}")
            return {} 