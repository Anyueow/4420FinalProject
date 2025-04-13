"""
Script to generate color dictionary for all runway collections.
"""

import os
import sys
import logging
from color_analyzer import ColorAnalyzer

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def generate_color_dictionary():
    """Generate color dictionary for all collections."""
    try:
        # Initialize color analyzer with optimized parameters
        analyzer = ColorAnalyzer(
            n_colors=8,  # Increased for better color detection
            resize_size=300,  # Slightly larger size for better detail
            min_area_percentage=2.0,  # Lower threshold to catch more colors
            batch_size=100,  # Process 100 images at a time
            n_jobs=4  # Use 4 parallel threads
        )
        
        # Define the season directory with correct path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        season_dir = os.path.join(current_dir, 'data', 'scraped', 'runway', 'Fall25')
        
        logging.info(f"Looking for images in: {season_dir}")
        
        # Check if directory exists
        if not os.path.exists(season_dir):
            logging.error(f"Season directory not found: {season_dir}")
            return
            
        # Check for designer directories
        designer_dirs = [d for d in os.listdir(season_dir) 
                        if os.path.isdir(os.path.join(season_dir, d))]
        
        if not designer_dirs:
            logging.error(f"No designer directories found in {season_dir}")
            logging.info("Please ensure images are scraped and organized in designer-specific folders")
            return
            
        logging.info(f"Found designer directories: {', '.join(designer_dirs)}")
        logging.info("Generating color dictionary for all collections...")
        
        # Count total images before processing
        total_images = 0
        for d in designer_dirs:
            designer_path = os.path.join(season_dir, d)
            images = [f for f in os.listdir(designer_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
            logging.info(f"Found {len(images)} images in {d}")
        
        if total_images == 0:
            logging.error("No images found in any designer directory")
            logging.info("Please ensure images are properly scraped before running analysis")
            return
            
        logging.info(f"Found {total_images} images across {len(designer_dirs)} designer collections")
        
        # Generate color dictionary
        color_dict = analyzer.create_color_dictionary(season_dir)
        
        if not color_dict:
            logging.error("No colors were extracted from the images")
            return
            
        logging.info("\nColor Analysis Summary:")
        for designer, data in color_dict.items():
            logging.info(f"\n{designer}:")
            for color in data['top_colors']:
                logging.info(f"  {color['color']}: {color['average_percentage']:.1f}% ({color['count']} images)")
                
    except Exception as e:
        logging.error(f"Error generating color dictionary: {str(e)}")
        logging.error(f"Current working directory: {os.getcwd()}")
        raise

if __name__ == "__main__":
    setup_logging()
    generate_color_dictionary() 