"""
Script to generate training dataset using FashionCLIP pseudo-labeling with detailed categories.
"""

import os
import logging
import gc
import torch
from dataset_generator import DatasetGenerator
import sys
import random

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def generate_training_data(test_mode: bool = True, test_size: int = 5):
    """Generate training dataset from runway images.
    
    Args:
        test_mode: If True, only process a small subset of images
        test_size: Number of images to process in test mode
    """
    try:
        # Set memory-related settings
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Initialize dataset generator with very conservative settings
        generator = DatasetGenerator(
            confidence_threshold=0.7,  # Only keep predictions with 70%+ confidence
            batch_size=16  # Process 16 images at a time
        )
        
        # Define directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(current_dir, '..', 'data_collection', 'data', 'scraped', 'runway', 'Fall25')
        
        # Use test directory if in test mode
        if test_mode:
            output_dir = os.path.join(current_dir, '..', '..', 'data', 'processed', 'test_dataset')
            logging.info("Running in TEST MODE")
        else:
            output_dir = os.path.join(current_dir, '..', '..', 'data', 'processed', 'labeled_dataset')
        
        logging.info(f"Source directory: {source_dir}")
        logging.info(f"Output directory: {output_dir}")
        
        # Clear any existing memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get all image paths
        image_paths = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            logging.error(f"No images found in {source_dir}")
            return
        
        # In test mode, only process a small random subset
        if test_mode:
            if len(image_paths) > test_size:
                image_paths = random.sample(image_paths, test_size)
            logging.info(f"TEST MODE: Processing {len(image_paths)} random images")
        else:
            logging.info(f"Found {len(image_paths)} images")
        
        # Generate dataset with minimal memory usage
        dataset_info = generator.generate_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            n_jobs=1,  # Single thread processing
            image_paths=image_paths  # Pass specific images to process
        )
        
        if not dataset_info:
            logging.error("Failed to generate dataset")
            return
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print summary
        logging.info("\nDataset Generation Summary:")
        logging.info(f"Total images processed: {dataset_info['total_images']}")
        logging.info(f"Successfully labeled: {dataset_info['labeled_images']}")
        
        # Print statistics in batches to avoid memory issues
        def print_sorted_stats(data_dict, title):
            logging.info(f"\n{title}:")
            for key, count in sorted(data_dict.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"{key}: {count} images")
                # Small delay to allow system to process output
                gc.collect()
        
        # Print statistics in sections
        print_sorted_stats(dataset_info['super_categories'], "Super-categories")
        print_sorted_stats(dataset_info['categories'], "Detailed categories")
        print_sorted_stats(dataset_info['super_styles'], "Style super-categories")
        print_sorted_stats(dataset_info['styles'], "Detailed styles")
        print_sorted_stats(dataset_info['super_patterns'], "Pattern super-categories")
        print_sorted_stats(dataset_info['patterns'], "Detailed patterns")
        
    except Exception as e:
        logging.error(f"Error generating training data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        setup_logging()
        # Process full dataset
        generate_training_data(test_mode=False)
    except KeyboardInterrupt:
        logging.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1) 