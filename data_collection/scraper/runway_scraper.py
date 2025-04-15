"""
Runway image scraper for nowfashion.com, url chnaged for each season
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
import re
import urllib.parse
import torch
import optparse

class RunwayScraper:
    """Scraper for runway images from nowfashion.com"""
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, 
                 base_url: str = "https://nowfashion.com/fashion/collections/fall-2023-ready-to-wear/",
                 save_dir: str = os.path.join(parent_dir, "data/scraped/runway/Fall23")):
        self.base_url = base_url
        self.save_dir = save_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        logging.basicConfig(
            filename=os.path.join(self.save_dir, 'scraping.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def get_show_links(self) -> List[str]:
        """Get links to all runway shows."""
        shows = []
        try:
            # Get the main landing page
            url = self.base_url
            logging.info(f"Fetching landing page: {url}")
            
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                logging.error(f"Failed to fetch landing page: Status {response.status_code}")
                return shows
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all article links that match the runway show pattern
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'ready-to-wear' in href.lower():
                    if href not in shows:
                        shows.append(href)
                        logging.info(f"Found show: {href}")
            
            logging.info(f"Found {len(shows)} shows")
            
        except Exception as e:
            logging.error(f"Error getting show links: {str(e)}")
            
        return shows
    
    def get_runway_images(self, show_url: str) -> Tuple[List[str], str]:
        """Get all runway images from a show."""
        images = []
        designer = ""
        try:
            logging.info(f"Fetching show: {show_url}")
            
            # Make sure we have the full URL
            if not show_url.startswith('http'):
                show_url = urllib.parse.urljoin(self.base_url, show_url)
                
            response = requests.get(show_url, headers=self.headers)
            
            if response.status_code != 200:
                logging.error(f"Failed to fetch show: Status {response.status_code}")
                return images, designer
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract designer name from URL pattern: saint-laurent-ready-to-wear...
            url_parts = show_url.split('/')
            for part in url_parts:
                if 'ready-to-wear' in part:
                    designer = part.split('-ready-to-wear')[0].replace('-', ' ').title()
                    break
            
            logging.info(f"Found designer: {designer}")
            
            # Find all img tags in the article content
            article = soup.find('article') or soup
            for img in article.find_all('img'):
                src = img.get('src', '')
                if src and ('.jpg' in src.lower() or '.jpeg' in src.lower() or '.png' in src.lower()):
                    # Get the highest resolution version by removing size suffix
                    src = re.sub(r'-\d+x\d+', '', src)
                    if src not in images:
                        images.append(src)
                        logging.info(f"Found image: {src}")
            
            logging.info(f"Found {len(images)} images for {designer}")
            
        except Exception as e:
            logging.error(f"Error getting runway images for {show_url}: {str(e)}")
            
        return images, designer
    
    def download_image(self, url: str, designer: str, idx: int) -> str:
        """Download an image and return its local path."""
        try:
            # Clean URL and make absolute if needed
            url = urllib.parse.urljoin(self.base_url, url)
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logging.error(f"Failed to download image {url}: Status {response.status_code}")
                return None
            
            # Create designer directory
            designer_dir = os.path.join(self.save_dir, designer.replace(' ', '_'))
            os.makedirs(designer_dir, exist_ok=True)
            
            # Save image
            image_path = os.path.join(designer_dir, f"look_{idx:03d}.jpg")
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"Downloaded image {idx+1} for {designer}")
            return image_path
            
        except Exception as e:
            logging.error(f"Error downloading image {url}: {str(e)}")
            return None
    
    def scrape(self) -> Dict:
        """Scrape all shows and their images."""
        data = {
            'scrape_date': datetime.now().isoformat(),
            'shows': defaultdict(dict)
        }
        
        # Get all show links
        show_links = self.get_show_links()
        
        for show_url in show_links:
            # Get runway images for show
            images, designer = self.get_runway_images(show_url)
            
            if not designer:
                logging.warning(f"Could not find designer name for {show_url}")
                continue
            
            # Download images
            image_paths = []
            for idx, img_url in enumerate(images):
                path = self.download_image(img_url, designer, idx)
                if path:
                    image_paths.append(path)
                time.sleep(1)  # Be nice to the server
            
            # Store show data
            data['shows'][designer] = {
                'url': show_url,
                'image_count': len(image_paths),
                'image_paths': image_paths
            }
            
            # Save progress after each show
            self._save_data(data)
            
        return data
    
    def _save_data(self, data: Dict):
        """Save data to JSON file."""
        with open(os.path.join(self.save_dir, 'runway_data.json'), 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    # Initialize the scraper
    scraper = RunwayScraper()
    
    # Start scraping
    print("Starting runway scraping...")
    data = scraper.scrape()
    
    print(f"Scraping complete! Data saved to {scraper.save_dir}/runway_data.json") 