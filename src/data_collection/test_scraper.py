"""
Test script for runway image scraper.
"""

import os
import sys
import logging
from runway_scraper import RunwayScraper

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def test_scraper():
    """Test the runway scraper functionality."""
    try:
        # Initialize scraper
        scraper = RunwayScraper()
        
        # Test show links extraction
        logging.info("Testing show links extraction...")
        show_links = scraper.get_show_links()
        logging.info(f"Found {len(show_links)} show links")
        
        if show_links:
            # Test first show
            first_show = show_links[0]
            logging.info(f"\nTesting first show: {first_show}")
            
            # Get runway images
            images, designer = scraper.get_runway_images(first_show)
            logging.info(f"Found {len(images)} images for {designer}")
            
            if images:
                # Test image download
                logging.info("\nTesting image download...")
                image_path = scraper.download_image(images[0], designer, 0)
                if image_path:
                    logging.info(f"Successfully downloaded image to: {image_path}")
                else:
                    logging.error("Failed to download image")
            
            # Test full season scraping
            logging.info("\nTesting full season scraping...")
            season_data = scraper.scrape_season()
            logging.info(f"Scraped data for {len(season_data['shows'])} shows")
            
    except Exception as e:
        logging.error(f"Error in test: {str(e)}")

if __name__ == "__main__":
    setup_logging()
    test_scraper() 