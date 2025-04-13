"""
Test script for FashionCLIP-based trend analysis.
"""

import os
import logging
from trend_analyzer import TrendAnalyzer

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_single_collection():
    """Test trend analysis on a single collection."""
    try:
        # Initialize trend analyzer
        analyzer = TrendAnalyzer()
        
        # Specify collection directory
        collection_dir = "src/data_collection/data/scraped/runway/Fall25/Shiatzy_Chen"
        
        # Verify directory exists
        if not os.path.exists(collection_dir):
            logging.error(f"Directory not found: {collection_dir}")
            return
        
        # Count images
        image_files = [f for f in os.listdir(collection_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            logging.error(f"No images found in {collection_dir}")
            return
        
        logging.info(f"Found {len(image_files)} images in collection")
        
        # Analyze collection
        logging.info(f"Analyzing trends for collection: {collection_dir}")
        stats = analyzer.analyze_collection(collection_dir)
        
        if not stats:
            logging.error("No trends were extracted from the collection")
            return
        
        # Print results
        logging.info("\nTrend Analysis Results:")
        
        logging.info("\nTop Categories:")
        for category, data in sorted(
            stats['categories'].items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:5]:
            logging.info(f"{category}: {data['count']} items ({data['percentage']:.1f}%)")
        
        logging.info("\nTop Styles:")
        for style, data in sorted(
            stats['styles'].items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:5]:
            logging.info(f"{style}: {data['count']} items ({data['percentage']:.1f}%)")
        
        logging.info("\nTop Patterns:")
        for pattern, data in sorted(
            stats['patterns'].items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:5]:
            logging.info(f"{pattern}: {data['count']} items ({data['percentage']:.1f}%)")
        
    except Exception as e:
        logging.error(f"Error during trend analysis: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    test_single_collection() 