"""
Generate color dictionary for all runway collections.
"""

import os
import sys
import logging
import warnings
from color_analyzer import ColorAnalyzer

def setup_logging():
    """Setup logging configuration."""
    # Suppress scikit-learn warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def generate_color_dictionary():
    """Generate color dictionary for all collections."""
    try:
        # Initialize color analyzer
        analyzer = ColorAnalyzer(n_colors=5)
        
        # Path to season directory
        season_dir = "data/scraped/runway/Fall25"
        if not os.path.exists(season_dir):
            logging.error(f"Season directory not found: {season_dir}")
            return
        
        # Create color dictionary
        logging.info("Generating color dictionary for all collections...")
        color_dict = analyzer.create_color_dictionary(season_dir)
        
        # Print summary
        logging.info("\nColor Analysis Summary:")
        for designer, data in color_dict.items():
            logging.info(f"\n{designer}:")
            logging.info(f"Total Images: {data['total_images']}")
            logging.info("Top 5 Colors:")
            for color_data in data['top_colors']:
                logging.info(f"  - {color_data['color']}: {color_data['average_percentage']}% ({color_data['count']} occurrences)")
        
    except Exception as e:
        logging.error(f"Error generating color dictionary: {str(e)}")

if __name__ == "__main__":
    setup_logging()
    generate_color_dictionary() 