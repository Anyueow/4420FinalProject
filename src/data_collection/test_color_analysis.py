"""
Test script for color analysis of runway images.
"""

import os
import sys
import logging
from color_analyzer import ColorAnalyzer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def test_color_analysis():
    """Test color analysis functionality."""
    try:
        # Initialize color analyzer
        analyzer = ColorAnalyzer(n_colors=5)
        
        # Test with a specific collection
        collection_dir = "src/data_collection/data/scraped/runway/Fall25/Shiatzy_Chen"
        if not os.path.exists(collection_dir):
            logging.error(f"Collection directory not found: {collection_dir}")
            return
        
        # Analyze collection colors
        logging.info(f"Analyzing colors for collection: {collection_dir}")
        colors = analyzer.analyze_collection(collection_dir)
        
        # Get top 5 colors
        top_colors = analyzer.get_top_colors(colors, n=5)
        
        # Print results
        logging.info("\nTop 5 colors in the collection:")
        for color, count in top_colors:
            logging.info(f"Color: {color}, Count: {count}")
        
        # Test full season analysis
        season_dir = "data/scraped/runway/Fall25"
        if os.path.exists(season_dir):
            logging.info("\nAnalyzing colors for all collections in the season...")
            season_colors = analyzer.analyze_season(season_dir)
            
            # Print results for each collection
            for designer, colors in season_colors.items():
                logging.info(f"\n{designer} - Top 5 colors:")
                top_colors = analyzer.get_top_colors(colors, n=5)
                for color, count in top_colors:
                    logging.info(f"Color: {color}, Count: {count}")
        
    except Exception as e:
        logging.error(f"Error in test: {str(e)}")

if __name__ == "__main__":
    setup_logging()
    test_color_analysis() 