"""
Test script for color analysis on a single collection directory.
"""

import os
import logging
from color_analyzer import ColorAnalyzer

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_single_collection():
    """Test color analysis on a single collection."""
    try:
        # Initialize analyzer with optimized parameters
        analyzer = ColorAnalyzer(
            n_colors=8,  # Increased for better color detection
            resize_size=300,
            min_area_percentage=2.0,  # Lowered to catch more color variations
            skin_tone_threshold=0.1
        )
        
        # Specify the collection directory
        collection_dir = "src/data_collection/data/scraped/runway/Fall25/Shiatzy_Chen"
        
        # Verify directory exists
        if not os.path.exists(collection_dir):
            logging.error(f"Directory not found: {collection_dir}")
            return
            
        # Count images in directory
        image_files = [f for f in os.listdir(collection_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            logging.error(f"No images found in {collection_dir}")
            return
            
        logging.info(f"Found {len(image_files)} images in collection")
        
        # Analyze collection
        logging.info(f"Analyzing colors for collection: {collection_dir}")
        colors = analyzer.analyze_collection(collection_dir)
        
        if not colors:
            logging.error("No colors were extracted from the collection")
            return
            
        # Print results
        logging.info("\nColor Analysis Results:")
        for color, data in list(colors.items())[:5]:  # Top 5 colors
            percentage = data['average_percentage']
            count = data['count']
            logging.info(f"Color {color}: {percentage:.1f}% ({count} occurrences)")
            
        # Save results to file
        result_file = os.path.join(collection_dir, "color_analysis.json")
        with open(result_file, 'w') as f:
            import json
            json.dump(colors, f, indent=2)
        logging.info(f"\nResults saved to: {result_file}")
        
    except Exception as e:
        logging.error(f"Error during color analysis: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    test_single_collection() 