"""
Script to generate training dataset using FashionCLIP pseudo-labeling with detailed categories.
"""

import os
import logging
from dataset_generator import DatasetGenerator

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def generate_training_data():
    """Generate training dataset from runway images."""
    try:
        # Initialize dataset generator
        generator = DatasetGenerator(
            confidence_threshold=0.7  # Only keep predictions with 70%+ confidence
        )
        
        # Define directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(current_dir, '..', 'data_collection', 'data', 'scraped', 'runway', 'Fall25')
        output_dir = os.path.join(current_dir, '..', '..', 'data', 'processed', 'labeled_dataset')
        
        logging.info(f"Source directory: {source_dir}")
        logging.info(f"Output directory: {output_dir}")
        
        # Generate dataset
        dataset_info = generator.generate_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            n_jobs=4  # Use 4 parallel threads
        )
        
        if not dataset_info:
            logging.error("Failed to generate dataset")
            return
        
        # Print summary
        logging.info("\nDataset Generation Summary:")
        logging.info(f"Total images processed: {dataset_info['total_images']}")
        logging.info(f"Successfully labeled: {dataset_info['labeled_images']}")
        
        # Print category statistics
        logging.info("\nCategory Distribution:")
        logging.info("\nSuper-categories:")
        for super_cat, count in sorted(
            dataset_info['super_categories'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logging.info(f"{super_cat}: {count} images")
        
        logging.info("\nDetailed categories:")
        for category, count in sorted(
            dataset_info['categories'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logging.info(f"{category}: {count} images")
        
        # Print style statistics
        logging.info("\nStyle Distribution:")
        logging.info("\nStyle super-categories:")
        for super_style, count in sorted(
            dataset_info['super_styles'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logging.info(f"{super_style}: {count} images")
        
        logging.info("\nDetailed styles:")
        for style, count in sorted(
            dataset_info['styles'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logging.info(f"{style}: {count} images")
        
        # Print pattern statistics
        logging.info("\nPattern Distribution:")
        logging.info("\nPattern super-categories:")
        for super_pattern, count in sorted(
            dataset_info['super_patterns'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logging.info(f"{super_pattern}: {count} images")
        
        logging.info("\nDetailed patterns:")
        for pattern, count in sorted(
            dataset_info['patterns'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logging.info(f"{pattern}: {count} images")
        
    except Exception as e:
        logging.error(f"Error generating training data: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    generate_training_data() 