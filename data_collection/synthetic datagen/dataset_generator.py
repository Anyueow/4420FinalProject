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
import gc

class DatasetGenerator:
    """Generate labeled dataset using FashionCLIP with detailed fashion categories."""
    
    def __init__(self, confidence_threshold: float = 0.7, batch_size: int = 16):
        """Initialize CLIP model and detailed fashion categories."""
        # Force CPU mode and basic processing
        self.device = "cpu"
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        # Load basic CLIP model
        logging.info("Loading basic CLIP model in CPU mode...")
        try:
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                device_map="cpu"
            )
            logging.info("Successfully loaded CLIP model")
        except Exception as e:
            logging.error(f"Error loading CLIP model: {str(e)}")
            raise
        
        # Define detailed categories
        self.categories = {
            'tops': [
                'blouse', 'puff-sleeve blouse', 'tie-front blouse', 'shirred blouse',
                'shirt', 'button-down shirt', 'oxford shirt', 'oversized shirt',
                't-shirt', 'graphic t-shirt', 'cropped t-shirt', 'boxy t-shirt',
                'tank top', 'ribbed tank', 'halter top', 'muscle tank',
                'sweater', 'crewneck sweater', 'turtleneck sweater', 'chunky knit sweater',
                'cardigan', 'cropped cardigan', 'duster cardigan', 'button-front cardigan',
                'hoodie', 'oversized hoodie', 'zip-up hoodie', 'logoed hoodie',
                'crop top', 'bralette-style crop top', 'off-shoulder crop top',
                'corset top', 'structured corset', 'lace-up corset', 'boned corset',
                'camisole', 'silk camisole', 'lace-trimmed camisole', 'slip-style camisole'
            ],
            'outerwear': [
                'coat', 'trench coat', 'wool coat', 'puffer coat', 'duster coat',
                'jacket', 'bomber jacket', 'denim jacket', 'leather jacket', 'utility jacket',
                'blazer', 'tailored blazer', 'double-breasted blazer', 'relaxed blazer',
                'parka', 'fur-lined parka', 'technical parka', 'oversized parka',
                'cape', 'poncho-style cape', 'structured cape', 'wool cape',
                'vest', 'quilted vest', 'knit vest', 'tailored vest', 'puffer vest',
                'windbreaker', 'lightweight windbreaker', 'sporty windbreaker', 'neon windbreaker'
            ],
            'bottoms': [
                'pants', 'tailored pants', 'wide-leg pants', 'cargo pants', 'track pants',
                'jeans', 'straight-leg jeans', 'baggy jeans', 'low-rise jeans', 'flared jeans',
                'skirt', 'mini skirt', 'midi skirt', 'maxi skirt', 'pleated skirt', 'A-line skirt', 'slip skirt',
                'shorts', 'Bermuda shorts', 'tailored shorts', 'denim shorts', 'bike shorts',
                'leggings', 'high-waisted leggings', 'printed leggings', 'leather-look leggings',
                'culottes', 'wide-leg culottes', 'cropped culottes', 'flowy culottes',
                'jumpsuit', 'boiler jumpsuit', 'tailored jumpsuit', 'belted jumpsuit',
                'romper', 'short-sleeve romper', 'floral romper', 'utility romper'
            ],
            'dresses': [
                'dress', 'slip dress', 'shirt dress', 'wrap dress', 'maxi dress', 'mini dress', 'midi dress',
                'gown', 'ball gown', 'evening gown', 'red-carpet gown',
                'sundress', 'tiered sundress', 'smocked sundress', 'spaghetti-strap sundress',
                'knit dress', 'ribbed knit dress', 'bodycon knit dress', 'sweater-style knit dress',
                'shirt dress', 'belted shirt dress', 'oversized shirt dress', 'striped shirt dress',
                'corset dress', 'structured corset dress', 'romantic corset dress', 'layered corset dress'
            ],
            'footwear': [
                'boots', 'ankle boots', 'knee-high boots', 'combat boots', 'cowboy boots', 'Chelsea boots',
                'sneakers', 'chunky sneakers', 'retro sneakers', 'minimalist sneakers', 'high-top sneakers',
                'heels', 'stiletto heels', 'block heels', 'kitten heels', 'platform heels',
                'sandals', 'strappy sandals', 'slide sandals', 'fisherman sandals', 'wedge sandals',
                'flats', 'ballet flats', 'loafers', 'Mary Janes', 'mules',
                'clogs', 'wooden clogs', 'shearling-lined clogs', 'minimalist clogs',
                'espadrilles', 'platform espadrilles', 'lace-up espadrilles', 'canvas espadrilles',
                'slippers', 'shearling slippers', 'logoed slippers', 'outdoor-ready slippers'
            ],
            'bags': [
                'handbag', 'tote bag', 'satchel', 'shoulder bag', 'bucket bag',
                'clutch', 'envelope clutch', 'beaded clutch', 'box clutch',
                'backpack', 'mini backpack', 'technical backpack', 'leather backpack',
                'crossbody', 'saddle crossbody', 'chain-strap crossbody', 'quilted crossbody',
                'baguette', 'slim baguette', 'logoed baguette', 'vintage-inspired baguette',
                'belt bag', 'fanny pack', 'sporty belt bag', 'luxury belt bag',
                'oversized tote', 'canvas tote', 'leather tote', 'woven tote',
                'micro bag', 'charm-sized bag', 'novelty bag', 'statement bag'
            ],
            'accessories': [
                'jewelry', 'statement necklace', 'layered chains', 'chunky rings', 'mismatched earrings',
                'sunglasses', 'cat-eye sunglasses', 'oversized sunglasses', 'sporty sunglasses', 'shield sunglasses',
                'hat', 'bucket hat', 'beret', 'baseball cap', 'wide-brim hat', 'baker boy hat',
                'scarf', 'silk scarf', 'oversized scarf', 'blanket scarf', 'neckerchief',
                'belt', 'chain belt', 'wide belt', 'logoed belt', 'corset-style belt',
                'gloves', 'leather gloves', 'fingerless gloves', 'opera-length gloves',
                'hair accessories', 'claw clip', 'headband', 'scrunchie', 'barrette',
                'socks', 'sheer socks', 'athletic socks', 'statement socks', 'knee-high socks',
                'tights', 'patterned tights', 'fishnet tights', 'opaque tights', 'glitter tights',
                'watch', 'smart watch', 'vintage watch', 'chunky watch', 'minimalist watch',
                'umbrella', 'transparent umbrella', 'printed umbrella', 'compact umbrella'
            ],
            'intimates': [
                'bra', 'bralette', 'push-up bra', 'wireless bra', 'sports bra',
                'underwear', 'high-cut underwear', 'boy-short underwear', 'seamless underwear',
                'bodysuit', 'mesh bodysuit', 'snap-crotch bodysuit', 'long-sleeve bodysuit',
                'lingerie', 'teddy', 'slip', 'garter set',
                'swimsuit', 'one-piece swimsuit', 'bikini', 'tankini', 'cut-out swimsuit',
                'cover-up', 'sarong', 'kimono', 'mesh cover-up'
            ],
            'athleisure': [
                'sports bra', 'high-impact sports bra', 'strappy sports bra', 'cropped sports bra',
                'activewear top', 'tank top', 'long-sleeve top', 'cropped hoodie',
                'activewear bottoms', 'yoga pants', 'running shorts', 'compression tights',
                'tracksuit', 'matching tracksuit', 'velour tracksuit', 'technical tracksuit'
            ]
        }
        
        # Define detailed styles
        self.styles = {
            'core_aesthetics': [
                'casual', 'relaxed', 'everyday', 'effortless',
                'formal', 'structured', 'professional', 'event-ready',
                'elegant', 'refined', 'graceful', 'sophisticated',
                'streetwear', 'urban', 'bold', 'logo-heavy', 'sport-inspired',
                'minimalist', 'clean lines', 'neutral tones', 'understated',
                'bohemian', 'flowy', 'eclectic', 'earthy', 'layered',
                'vintage', 'retro', 'nostalgic', '70s', '90s', 'Y2K',
                'modern', 'sleek', 'innovative', 'forward-thinking',
                'classic', 'timeless', 'tailored', 'enduring',
                'avant-garde', 'experimental', 'artistic', 'boundary-pushing'
            ],
            'gender_identity': [
                'feminine', 'soft', 'delicate', 'romantic', 'floral',
                'masculine', 'structured', 'rugged', 'utilitarian',
                'unisex', 'gender-neutral', 'versatile', 'inclusive',
                'androgynous', 'blending masculine and feminine', 'ambiguous'
            ],
            'cultural_subcultural': [
                'preppy', 'collegiate', 'polished', 'Ivy League-inspired',
                'grunge', 'edgy', 'distressed', '90s-inspired', 'layered',
                'punk', 'rebellious', 'studded', 'leather-heavy',
                'goth', 'dark', 'moody', 'velvet', 'lace',
                'cottagecore', 'rustic', 'pastoral', 'floral', 'whimsical',
                'balletcore', 'delicate', 'tutu-inspired', 'satin', 'soft pinks',
                'gorpcore', 'outdoor', 'technical', 'hiking-inspired',
                'techwear', 'futuristic', 'functional', 'modular',
                'normcore', 'anti-fashion', 'basic', 'intentionally plain',
                'kidcore', 'playful', 'colorful', 'nostalgic', 'cartoon-inspired'
            ],
            'occasion_mood': [
                'party', 'glamorous', 'sparkly', 'bold',
                'resort', 'breezy', 'tropical', 'lightweight',
                'workwear', 'professional', 'versatile', 'polished',
                'loungewear', 'cozy', 'soft', 'home-ready',
                'festival', 'eclectic', 'vibrant', 'fringed'
            ],
            'value_driven': [
                'luxury', 'high-end', 'exclusive', 'premium materials',
                'sustainable', 'eco-friendly', 'ethical', 'upcycled', 'organic',
                'artisanal', 'handcrafted', 'small-batch', 'cultural',
                'slow fashion', 'mindful', 'durable', 'timeless',
                'fast fashion', 'trend-driven', 'affordable', 'mass-produced'
            ],
            'emerging_micro': [
                'quiet luxury', 'subtle wealth', 'logo-less', 'high-quality',
                'loud luxury', 'logo-heavy', 'opulent', 'maximalist',
                'mob wife', 'fur', 'animal print', 'gold jewelry', 'dramatic',
                'coastal grandma', 'linen', 'soft neutrals', 'relaxed elegance',
                'barbiecore', 'pink', 'hyper-feminine', 'playful',
                'dopamine dressing', 'colorful', 'mood-boosting', 'bold'
            ]
        }
        
        # Define detailed patterns and textures
        self.patterns = {
            'classic_patterns': [
                'floral', 'ditsy floral', 'oversized floral', 'botanical',
                'striped', 'pinstripe', 'Breton stripe', 'bold stripe',
                'plaid', 'tartan', 'glen plaid', 'buffalo check',
                'checkered', 'gingham', 'houndstooth', 'argyle',
                'polka dot', 'micro dot', 'oversized dot', 'irregular dot',
                'geometric', 'chevron', 'grid', 'tessellated',
                'abstract', 'painterly', 'surreal', 'fluid',
                'animal print', 'leopard', 'zebra', 'snakeskin', 'cheetah'
            ],
            'textural_patterns': [
                'solid color', 'matte', 'glossy', 'satin',
                'metallic', 'gold', 'silver', 'bronze', 'iridescent',
                'sequined', 'all-over sequins', 'gradient sequins', 'patchwork sequins',
                'embroidered', 'floral embroidery', 'geometric embroidery', 'cultural motifs',
                'lace', 'delicate lace', 'crochet lace', 'guipure lace',
                'leather', 'smooth leather', 'patent leather', 'distressed leather', 'vegan leather',
                'suede', 'soft suede', 'brushed suede', 'colorful suede',
                'velvet', 'crushed velvet', 'devoré velvet', 'embossed velvet',
                'shearling', 'natural shearling', 'dyed shearling', 'faux shearling',
                'corduroy', 'wide-wale corduroy', 'micro corduroy', 'printed corduroy',
                'denim', 'raw denim', 'washed denim', 'distressed denim', 'patchwork denim',
                'knit', 'cable knit', 'ribbed knit', 'open-weave knit',
                'quilted', 'diamond quilted', 'channel quilted', 'padded quilted',
                'mesh', 'athletic mesh', 'sheer mesh', 'metallic mesh',
                'tweed', 'herringbone tweed', 'bouclé tweed', 'multicolored tweed'
            ],
            'innovative_sustainable': [
                'upcycled', 'patched', 'repurposed', 'mixed-media',
                'organic', 'cotton', 'hemp', 'linen', 'bamboo',
                'recycled', 'PET plastic', 'reclaimed fabrics',
                'biofabricated', 'lab-grown leather', 'mushroom leather',
                'tie-dye', 'shibori', 'ombré tie-dye', 'modern tie-dye',
                'ombré', 'gradient', 'dip-dye', 'blurred',
                'colorblock', 'bold colorblock', 'tonal colorblock', 'asymmetrical colorblock',
                'digital print', 'pixelated', '3D-effect', 'surreal',
                'faux fur', 'long-pile faux fur', 'short-pile faux fur', 'colorful faux fur'
            ],
            'cultural_artisanal': [
                'tapestry', 'jacquard', 'woven', 'pictorial',
                'batik', 'wax-resist', 'indigo', 'modern',
                'ikat', 'blurred ikat', 'vibrant ikat', 'woven ikat',
                'paisley', 'traditional paisley', 'oversized paisley', 'monochrome paisley',
                'brocade', 'metallic brocade', 'floral brocade', 'opulent brocade',
                'damask', 'tonal damask', 'reversible damask', 'ornate damask'
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
        """Process a single image and return its classifications with optimized memory usage."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process with minimal memory usage
            with torch.no_grad():
                results = {}
                
                # Process categories in smaller groups
                category_groups = [list(self.categories.keys())[i:i+3] for i in range(0, len(self.categories), 3)]
                for group in category_groups:
                    # Get all subcategories for this group
                    group_categories = []
                    for cat in group:
                        group_categories.extend(self.categories[cat])
                    
                    # Process this group
                    inputs = self.processor(
                        images=image,
                        text=group_categories,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    category_probs = outputs.logits_per_image.softmax(dim=1)[0]
                    max_category_prob, max_category_idx = torch.max(category_probs, dim=0)
                    
                    # Clear memory immediately
                    del inputs, outputs, category_probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Only continue if category confidence meets threshold
                    if float(max_category_prob) >= self.confidence_threshold:
                        predicted_category = group_categories[max_category_idx]
                        results['category'] = {
                            'label': predicted_category,
                            'super_category': self.category_to_super[predicted_category],
                            'confidence': float(max_category_prob)
                        }
                        break
                
                if 'category' not in results:
                    return None
                
                # Process styles in smaller groups
                style_groups = [list(self.styles.keys())[i:i+2] for i in range(0, len(self.styles), 2)]
                for group in style_groups:
                    group_styles = []
                    for style in group:
                        group_styles.extend(self.styles[style])
                    
                    inputs = self.processor(
                        images=image,
                        text=group_styles,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    style_probs = outputs.logits_per_image.softmax(dim=1)[0]
                    max_style_prob, max_style_idx = torch.max(style_probs, dim=0)
                    
                    del inputs, outputs, style_probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if float(max_style_prob) >= self.confidence_threshold:
                        predicted_style = group_styles[max_style_idx]
                        results['style'] = {
                            'label': predicted_style,
                            'super_style': self.style_to_super[predicted_style],
                            'confidence': float(max_style_prob)
                        }
                        break
                
                # Process patterns in smaller groups
                pattern_groups = [list(self.patterns.keys())[i:i+2] for i in range(0, len(self.patterns), 2)]
                for group in pattern_groups:
                    group_patterns = []
                    for pattern in group:
                        group_patterns.extend(self.patterns[pattern])
                    
                    inputs = self.processor(
                        images=image,
                        text=group_patterns,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    pattern_probs = outputs.logits_per_image.softmax(dim=1)[0]
                    max_pattern_prob, max_pattern_idx = torch.max(pattern_probs, dim=0)
                    
                    del inputs, outputs, pattern_probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if float(max_pattern_prob) >= self.confidence_threshold:
                        predicted_pattern = group_patterns[max_pattern_idx]
                        results['pattern'] = {
                            'label': predicted_pattern,
                            'super_pattern': self.pattern_to_super[predicted_pattern],
                            'confidence': float(max_pattern_prob)
                        }
                        break
                
                results['path'] = image_path
                return results
            
            return None
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None
    
    def generate_dataset(self, source_dir: str, output_dir: str, n_jobs: int = 1, image_paths: list = None) -> dict:
        """Generate labeled dataset from source images with optimized batch processing."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # If no specific paths provided, get all images from source_dir
            if image_paths is None:
                image_paths = []
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_paths.append(os.path.join(root, file))
            
            if not image_paths:
                logging.error(f"No images found to process")
                return {}
            
            logging.info(f"Processing {len(image_paths)} images")
            
            # Initialize CSV data
            csv_data = []
            headers = ['image_path', 'category', 'super_category', 'style', 'super_style', 'pattern', 'super_pattern']
            
            # Process images in optimized batches
            results = []
            total_images = len(image_paths)
            batch_size = min(self.batch_size, 8)  # Reduced batch size for memory efficiency
            
            for i in range(0, total_images, batch_size):
                batch_paths = image_paths[i:i + batch_size]
                logging.info(f"\nProcessing batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
                
                # Process each image in the batch
                for path in tqdm(batch_paths, desc="Processing images"):
                    try:
                        result = self.process_image(path)
                        if result:
                            results.append(result)
                            # Add to CSV data
                            csv_data.append([
                                path,
                                result['category']['label'],
                                result['category']['super_category'],
                                result.get('style', {}).get('label', ''),
                                result.get('style', {}).get('super_style', ''),
                                result.get('pattern', {}).get('label', ''),
                                result.get('pattern', {}).get('super_pattern', '')
                            ])
                    except Exception as e:
                        logging.error(f"Error processing image {path}: {str(e)}")
                        continue
                    
                    # Force garbage collection after each image
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Additional cleanup after batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save to CSV
            import csv
            csv_path = os.path.join(output_dir, 'fashion_labels_fall24.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(csv_data)
            
            logging.info(f"\nDataset generated with {len(results)} labeled images")
            logging.info(f"Labels saved to {csv_path}")
            
            # Return summary statistics
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
            
            # Update statistics
            for result in results:
                dataset_info['categories'][result['category']['label']] += 1
                dataset_info['super_categories'][result['category']['super_category']] += 1
                if 'style' in result:
                    dataset_info['styles'][result['style']['label']] += 1
                    dataset_info['super_styles'][result['style']['super_style']] += 1
                if 'pattern' in result:
                    dataset_info['patterns'][result['pattern']['label']] += 1
                    dataset_info['super_patterns'][result['pattern']['super_pattern']] += 1
            
            return dataset_info
            
        except Exception as e:
            logging.error(f"Error generating dataset: {str(e)}")
            return {} 