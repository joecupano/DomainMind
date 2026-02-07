"""
Web scraping module for RF communications websites
"""
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
import requests
from urllib.parse import urljoin, urlparse
import trafilatura

from utils import clean_text, chunk_text, display_error, display_success, display_info, is_url_valid
from config import CHUNK_SIZE, CHUNK_OVERLAP, SCRAPED_DIR, DEFAULT_RF_WEBSITES

logger = logging.getLogger(__name__)

class WebScraper:
    """Handles web scraping for RF communications content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.delay_between_requests = 1  # Be respectful to websites
    
    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a single URL"""
        try:
            if not is_url_valid(url):
                display_error(f"Invalid URL format: {url}")
                return None
            
            display_info(f"Scraping content from {url}...")
            
            # Fetch the webpage
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                display_error(f"Failed to fetch content from {url}")
                return None
            
            # Extract main text content
            text_content = trafilatura.extract(downloaded)
            if not text_content:
                display_error(f"No text content extracted from {url}")
                return None
            
            # Clean and chunk the text
            cleaned_text = clean_text(text_content)
            chunks = chunk_text(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)
            
            if not chunks:
                display_error(f"No valid chunks created from {url}")
                return None
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata and metadata.title else urlparse(url).netloc
            
            document_data = {
                'url': url,
                'title': title,
                'content': cleaned_text,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'source_type': 'web',
                'domain': urlparse(url).netloc
            }
            
            # Save scraped content
            self._save_scraped_content(document_data)
            
            display_success(f"Successfully scraped {url} - {len(chunks)} chunks created")
            return document_data
            
        except Exception as e:
            display_error(f"Error scraping {url}: {str(e)}")
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def scrape_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape content from multiple URLs"""
        results = []
        
        if not urls:
            display_info("No URLs provided for scraping")
            return results
        
        display_info(f"Starting to scrape {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            display_info(f"Processing URL {i}/{len(urls)}: {url}")
            
            result = self.scrape_url(url)
            if result:
                results.append(result)
            
            # Be respectful - add delay between requests
            if i < len(urls):
                time.sleep(self.delay_between_requests)
        
        display_success(f"Successfully scraped {len(results)} out of {len(urls)} URLs")
        return results
    
    def scrape_rf_websites(self, custom_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scrape default RF communications websites plus any custom URLs"""
        urls_to_scrape = DEFAULT_RF_WEBSITES.copy()
        
        if custom_urls:
            urls_to_scrape.extend(custom_urls)
        
        # Remove duplicates while preserving order
        urls_to_scrape = list(dict.fromkeys(urls_to_scrape))
        
        return self.scrape_multiple_urls(urls_to_scrape)
    
    def scrape_sitemap(self, base_url: str, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Scrape multiple pages from a website using sitemap or crawling"""
        results = []
        
        try:
            # Try to find sitemap
            sitemap_urls = [
                urljoin(base_url, '/sitemap.xml'),
                urljoin(base_url, '/sitemap_index.xml'),
                urljoin(base_url, '/robots.txt')
            ]
            
            pages_to_scrape = [base_url]  # Start with the base URL
            
            # For now, implement basic page discovery
            # In a production system, you might want to implement proper sitemap parsing
            
            # Limit the number of pages to scrape
            pages_to_scrape = pages_to_scrape[:max_pages]
            
            return self.scrape_multiple_urls(pages_to_scrape)
            
        except Exception as e:
            display_error(f"Error in sitemap scraping for {base_url}: {str(e)}")
            logger.error(f"Sitemap scraping error: {e}")
            return results
    
    def _save_scraped_content(self, document_data: Dict[str, Any]):
        """Save scraped content to local storage"""
        try:
            # Create filename from URL
            url = document_data['url']
            domain = urlparse(url).netloc
            filename = f"{domain}_{hash(url)}.txt"
            file_path = SCRAPED_DIR / filename
            
            # Save the content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write(f"Title: {document_data['title']}\n")
                f.write(f"Domain: {document_data['domain']}\n")
                f.write("=" * 50 + "\n")
                f.write(document_data['content'])
            
            logger.info(f"Saved scraped content to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving scraped content: {e}")
    
    def get_scraped_files(self) -> List[Path]:
        """Get list of previously scraped files"""
        return list(SCRAPED_DIR.glob("*.txt"))
    
    def load_scraped_content(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load previously scraped content from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the saved content
            lines = content.split('\n')
            url = lines[0].replace('URL: ', '') if len(lines) > 0 else ""
            title = lines[1].replace('Title: ', '') if len(lines) > 1 else ""
            domain = lines[2].replace('Domain: ', '') if len(lines) > 2 else ""
            
            # Find content start (after the separator line)
            content_start = 4
            main_content = '\n'.join(lines[content_start:])
            
            chunks = chunk_text(main_content, CHUNK_SIZE, CHUNK_OVERLAP)
            
            return {
                'url': url,
                'title': title,
                'content': main_content,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'source_type': 'web',
                'domain': domain
            }
            
        except Exception as e:
            logger.error(f"Error loading scraped content from {file_path}: {e}")
            return None
