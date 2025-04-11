"""
Test script to analyze nowfashion.com structure and update selectors.
"""

import requests
from bs4 import BeautifulSoup
import json
from pprint import pprint
import logging
import sys

def analyze_page_structure(url: str):
    """Analyze the HTML structure of a page."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    try:
        # Get the page
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Print page title
        print(f"\nPage Title: {soup.title.text if soup.title else 'No title found'}")
        
        # Analyze article structure
        print("\nAnalyzing article structure:")
        articles = soup.find_all('article')
        for idx, article in enumerate(articles[:3]):  # Look at first 3 articles
            print(f"\nArticle {idx + 1}:")
            # Get article classes
            print(f"Classes: {article.get('class', [])}")
            # Get title
            title = article.find(['h1', 'h2'], class_='entry-title')
            print(f"Title: {title.text.strip() if title else 'No title found'}")
            # Get links
            links = article.find_all('a')
            print("Links:")
            for link in links:
                print(f"  - {link.get('href', 'No href')} ({link.get('class', ['No class'])})")
            # Get images
            images = article.find_all('img')
            print("Images:")
            for img in images:
                print(f"  - {img.get('src', 'No src')} ({img.get('class', ['No class'])})")
        
        # Analyze pagination
        print("\nAnalyzing pagination:")
        pagination = soup.find_all(class_='pagination')
        if pagination:
            print("Found pagination elements:")
            for p in pagination:
                print(f"Classes: {p.get('class')}")
                links = p.find_all('a')
                print("Pagination links:")
                for link in links:
                    print(f"  - {link.get('href')} ({link.text.strip()})")
        else:
            print("No pagination found")
        
        # Save the HTML structure for manual inspection
        with open('page_structure.html', 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
            
        print("\nSaved full HTML structure to 'page_structure.html' for manual inspection")
        
    except Exception as e:
        print(f"Error analyzing page: {str(e)}")

def test_show_extraction():
    """Test extraction of show information."""
    from runway_scraper import RunwayScraper
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    try:
        # Initialize scraper
        scraper = RunwayScraper()
        
        # Test show links extraction
        print("\nTesting show links extraction:")
        show_links = scraper.get_show_links()
        print(f"Found {len(show_links)} show links")
        if show_links:
            print("\nFirst 3 shows:")
            for idx, link in enumerate(show_links[:3]):
                print(f"\nShow {idx + 1}:")
                print(f"URL: {link}")
                images, designer = scraper.get_runway_images(link)
                print(f"Designer: {designer}")
                print(f"Number of images: {len(images)}")
                if images:
                    print("First image URL:", images[0])
        
    except Exception as e:
        print(f"Error testing show extraction: {str(e)}")

if __name__ == "__main__":
    # Test main page structure
    print("Analyzing main page structure...")
    analyze_page_structure("https://nowfashion.com/shows")
    
    # Test show extraction
    print("\nTesting show extraction...")
    test_show_extraction() 