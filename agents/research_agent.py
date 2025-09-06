"""
ArXiv Paper Scraper

This script scrapes recent papers from arXiv based on a specified domain/category.
It extracts paper titles, methodologies (from abstracts), and downloads PDFs automatically.

Features:
- Search arXiv by domain/category
- Extract paper metadata (title, authors, abstract, etc.)
- Download PDFs to local data folder
- Extract methodology information from abstracts
- Save results to JSON file
"""

import os
import json
import re
import time
import io
import requests
import feedparser
from datetime import datetime, timedelta
from urllib.parse import urljoin
from pathlib import Path

# Optional imports for multimodal functionality
try:
    import cv2
    import numpy as np
    from PIL import Image
    from pdf2image import convert_from_path
    import fitz  # PyMuPDF
    MULTIMODAL_AVAILABLE = True
    
except ImportError:
    MULTIMODAL_AVAILABLE = False


class ArxivScraper:
    """
    A class to scrape recent papers from arXiv based on specified domains.
    """

    def __init__(self, data_folder="data"):
        """
        Initialize the ArxivScraper.

        Args:
            data_folder (str): Folder to store downloaded PDFs and metadata
        """
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of content
        self.images_folder = self.data_folder / "images"
        self.images_folder.mkdir(exist_ok=True)
        
        self.base_url = "http://export.arxiv.org/api/query"
        self.papers_data = []

        # Common methodology keywords to look for in abstracts
        self.methodology_keywords = [
            "method",
            "approach",
            "algorithm",
            "technique",
            "framework",
            "model",
            "architecture",
            "system",
            "implementation",
            "protocol",
            "procedure",
            "strategy",
            "solution",
            "mechanism",
            "pipeline",
            "network",
            "learning",
            "training",
            "optimization",
            "analysis",
        ]
        
        # Initialize models for figure extraction
        # No model initialization needed for pdf2image approach
        
        # Image classification keywords for filtering relevant figures
        self.architecture_keywords = [
            "architecture", "framework", "model", "network", "diagram", "flowchart",
            "pipeline", "structure", "design", "schematic", "graph", "chart",
            "figure", "fig", "workflow", "process", "algorithm", "method"
        ]

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        if not MULTIMODAL_AVAILABLE:
            print("⚠️ PDF image extraction requires additional packages: pdf2image, opencv-python, pillow")
            print("   Install with: pip install pdf2image opencv-python pillow")
            return False
        
        # Test pdf2image functionality (which requires Poppler)
        try:
            from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
            # Try to import pdf2image and test if Poppler is available
            test_result = convert_from_path.__doc__  # Simple check that the function exists
            return True
        except Exception as e:
            print("❌ pdf2image/Poppler not properly installed!")
            print("\n📋 Installation Instructions:")
            print("=" * 50)
            print("1. Install Python packages:")
            print("   pip install pdf2image opencv-python pillow")
            print("\n2. Install Poppler (required by pdf2image):")
            print("   🪟 Windows:")
            print("     - Download from: https://github.com/oschwartz10612/poppler-windows/releases")
            print("     - Extract to C:\\poppler")
            print("     - Add C:\\poppler\\bin to your PATH environment variable")
            print("     - OR use conda: conda install -c conda-forge poppler")
            print("\n   🐧 Linux (Ubuntu/Debian):")
            print("     sudo apt-get install poppler-utils")
            print("\n   🍎 macOS:")
            print("     brew install poppler")
            print("\n3. Restart your terminal/IDE after installation")
            print("=" * 50)
            return False
        

    def search_papers_by_keywords(
        self, keywords, max_results=10, sort_by="relevance", sort_order="descending",
        include_categories=None, time_range_days=180, start_date=None, end_date=None
    ):
        """
        Search for papers using natural language keywords across multiple fields.

        Args:
            keywords (str): Natural language search query (e.g., "biomedical image segmentation")
            max_results (int): Maximum number of papers to retrieve
            sort_by (str): Sort criteria ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order (str): Sort order ('ascending', 'descending')
            include_categories (list): Optional list of arXiv categories to include
            time_range_days (int): Number of days to look back (default: 6 months)
            start_date (str): Start date in YYYY-MM-DD format (optional)
            end_date (str): End date in YYYY-MM-DD format (optional)

        Returns:
            list: List of paper dictionaries
        """
        print(f"🔍 Searching for papers with keywords: '{keywords}'")

        # Handle custom date range or default time range
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                print(f"📅 Custom date range: {start_date} to {end_date}")
            except ValueError as e:
                print(f"❌ Invalid date format: {e}")
                return []
        else:
            # Calculate date range (ensure we're looking back in time)
            end_dt = datetime.now()
            # Use default 180 days if time_range_days is None
            days_to_look_back = time_range_days if time_range_days is not None else 180
            start_dt = end_dt - timedelta(days=days_to_look_back)
            print(f"📅 Looking back {days_to_look_back} days, fetching up to {max_results} papers...")

        # Format dates for arXiv API (YYYYMMDD format)
        start_date_str = start_dt.strftime("%Y%m%d")
        end_date_str = end_dt.strftime("%Y%m%d")

        print(f"📅 Date range: {start_date_str} to {end_date_str}")

        # Prepare search terms (simplified)
        search_terms = self._prepare_search_terms(keywords)

        # Auto-detect relevant categories if not provided
        if include_categories is None:
            include_categories = self._suggest_categories(keywords)
            print(f"🎯 Auto-detected relevant categories: {', '.join(include_categories)}")

        # Build simplified search query to avoid ArXiv API limits
        # Start with basic keyword search in title and abstract
        basic_query = f"all:{keywords}"

        # Add category filter if specified (simplified)
        if include_categories:
            category_query = " OR ".join([f"cat:{cat}" for cat in include_categories[:3]])  # Limit to 3 categories
            final_query = f"({basic_query}) AND ({category_query})"
        else:
            final_query = basic_query

        # Add date range filter
        date_filter = f"submittedDate:[{start_date_str}0000 TO {end_date_str}2359]"
        final_query = f"({final_query}) AND {date_filter}"

        print(f"📝 Query: {final_query}")

        params = {
            "search_query": final_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            # Make the API request
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            # Parse the feed
            feed = feedparser.parse(response.content)

            if feed.bozo:
                print("Warning: Feed may have parsing issues")

            papers = []
            for entry in feed.entries:
                paper_data = self.extract_paper_data(entry)
                papers.append(paper_data)

            print(f"✅ Successfully found {len(papers)} papers")
            self.papers_data = papers
            return papers

        except requests.RequestException as e:
            print(f"❌ Error fetching papers: {e}")
            return []
        except Exception as e:
            print(f"❌ Error parsing feed: {e}")
            return []

    def _prepare_search_terms(self, keywords):
        """
        Prepare search terms for arXiv API query.
        
        Args:
            keywords (str): Raw search keywords
            
        Returns:
            str: Formatted search terms
        """
        # Clean and prepare search terms
        terms = keywords.strip().lower()
        
        # Split into individual words
        words = re.findall(r'\b\w+\b', terms)
        
        # Create search combinations
        # 1. Exact phrase (if multiple words)
        if len(words) > 1:
            exact_phrase = f'"{keywords}"'
            
            # 2. All words must appear (AND)
            all_words = " AND ".join(words)
            
            # 3. Any words can appear (OR) 
            any_words = " OR ".join(words)
            
            # Combine for comprehensive search
            search_terms = f'({exact_phrase}) OR ({all_words}) OR ({any_words})'
        else:
            # Single word search
            search_terms = words[0]
            
        return search_terms

    def _suggest_categories(self, keywords):
        """
        Suggest relevant arXiv categories based on keywords.
        
        Args:
            keywords (str): Search keywords
            
        Returns:
            list: List of suggested arXiv categories
        """
        keywords_lower = keywords.lower()
        suggested_categories = []
        
        # Define category mappings
        category_mappings = {
            # Computer Science
            'cs.AI': ['artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning', 
                     'neural network', 'reinforcement learning', 'nlp', 'natural language'],
            'cs.CV': ['computer vision', 'image', 'visual', 'object detection', 'segmentation', 
                     'face recognition', 'optical', 'video', 'image processing'],
            'cs.LG': ['learning', 'algorithm', 'classification', 'regression', 'clustering', 
                     'supervised', 'unsupervised', 'training', 'model'],
            'cs.CL': ['language', 'text', 'nlp', 'translation', 'sentiment', 'speech', 
                     'linguistic', 'chatbot', 'dialogue'],
            'cs.RO': ['robot', 'robotics', 'autonomous', 'control', 'navigation', 'manipulation'],
            'cs.CR': ['security', 'cryptography', 'privacy', 'encryption', 'authentication'],
            'cs.DB': ['database', 'data', 'query', 'storage', 'retrieval'],
            'cs.DC': ['distributed', 'parallel', 'cloud', 'grid', 'concurrent'],
            'cs.NE': ['network', 'internet', 'protocol', 'communication', 'wireless'],
            
            # Statistics
            'stat.ML': ['statistical', 'statistics', 'bayesian', 'inference', 'probability'],
            
            # Mathematics
            'math.OC': ['optimization', 'control', 'optimal', 'minimize', 'maximize'],
            'math.ST': ['statistics', 'statistical', 'stochastic', 'random'],
            'math.NA': ['numerical', 'computational', 'simulation', 'algorithm'],
            
            # Physics
            'physics.comp-ph': ['computational physics', 'simulation', 'modeling'],
            'physics.data-an': ['data analysis', 'experimental', 'measurement'],
            
            # Quantitative Biology
            'q-bio.QM': ['biomedical', 'biological', 'medical', 'health', 'clinical'],
            'q-bio.NC': ['neuroscience', 'brain', 'neural', 'cognitive'],
            'q-bio.GN': ['genomics', 'genetics', 'dna', 'gene'],
            
            # Economics
            'econ.EM': ['economics', 'econometrics', 'market', 'financial'],
        }
        
        # Find matching categories
        for category, keywords_list in category_mappings.items():
            if any(keyword in keywords_lower for keyword in keywords_list):
                suggested_categories.append(category)
        
        # Default fallback categories if no matches
        if not suggested_categories:
            suggested_categories = ['cs.AI', 'cs.LG', 'cs.CV']  # Most common research areas
            
        return suggested_categories

    # Keep the original domain-based search for backward compatibility
    def search_papers(
        self, domain, max_results=10, sort_by="submittedDate", sort_order="descending"
    ):
        """
        Search for papers in a specific domain on arXiv (legacy method).

        Args:
            domain (str): arXiv category (e.g., 'cs.AI', 'cs.CV', 'cs.LG', 'math.CO')
            max_results (int): Maximum number of papers to retrieve
            sort_by (str): Sort criteria ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order (str): Sort order ('ascending', 'descending')

        Returns:
            list: List of paper dictionaries
        """
        print(f"🔍 Searching for papers in domain: {domain}")
        print(f"📅 Fetching {max_results} papers from the last 6 months...")

        # Calculate date range for the last 6 months (broader range for better diversity)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months instead of 30 days
        
        # Format dates for arXiv API (YYYYMMDD format)
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        
        # Construct the query with date range
        query = f"cat:{domain} AND submittedDate:[{start_date_str}0000 TO {end_date_str}2359]"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            print(f"📝 Query: {query}")
            
            # Make the API request
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            # Parse the feed
            feed = feedparser.parse(response.content)

            if feed.bozo:
                print("Warning: Feed may have parsing issues")

            papers = []
            for entry in feed.entries:
                paper_data = self.extract_paper_data(entry)
                papers.append(paper_data)

            print(f"✅ Successfully found {len(papers)} papers for domain {domain}")
            self.papers_data = papers
            return papers

        except requests.RequestException as e:
            print(f"❌ Error fetching papers: {e}")
            return []
        except Exception as e:
            print(f"❌ Error parsing feed: {e}")
            return []

    def extract_paper_data(self, entry):
        """
        Extract relevant data from a paper entry.

        Args:
            entry: feedparser entry object

        Returns:
            dict: Extracted paper data
        """
        # Extract basic information
        title = entry.title.replace("\n", " ").strip()
        abstract = entry.summary.replace("\n", " ").strip()
        authors = [author.name for author in entry.authors]

        # Extract arXiv ID and PDF URL
        arxiv_id = entry.id.split("/")[-1]
        pdf_url = None
        for link in entry.links:
            if link.type == "application/pdf":
                pdf_url = link.href
                break

        # If no direct PDF link, construct it
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Extract dates
        published = entry.published
        updated = getattr(entry, "updated", published)

        # Extract methodology from abstract
        methodology = self.extract_methodology(abstract)

        # Extract categories
        categories = []
        if hasattr(entry, "tags"):
            categories = [tag.term for tag in entry.tags]

        paper_data = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "methodology": methodology,
            "categories": categories,
            "published_date": published,
            "updated_date": updated,
            "pdf_url": pdf_url,
            "arxiv_url": entry.id,
            "downloaded": False,
            "pdf_filename": None,
        }

        return paper_data

    def extract_methodology(self, abstract):
        """
        Extract methodology information from the abstract.

        Args:
            abstract (str): Paper abstract

        Returns:
            list: List of methodology-related sentences
        """
        methodology_info = []

        # Split abstract into sentences
        sentences = re.split(r"[.!?]+", abstract)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            # Check if sentence contains methodology keywords
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in self.methodology_keywords):
                methodology_info.append(sentence)

        return methodology_info

    def download_pdfs_parallel(self, papers=None, max_workers=3, progress_callback=None):
        """
        Download PDFs for the papers in parallel.

        Args:
            papers (list): List of paper dictionaries. If None, uses self.papers_data
            max_workers (int): Maximum number of parallel downloads
            progress_callback (function): Optional callback function to report progress
        """
        if papers is None:
            papers = self.papers_data

        if not papers:
            print("No papers to download")
            return

        print(f"Starting parallel download of {len(papers)} PDFs with {max_workers} workers...")
        
        completed_count = 0

        def download_single_paper(paper):
            """Download a single paper"""
            try:
                paper_title = paper['title'][:50]
                print(f"📥 Downloading: {paper_title}...")

                # Create safe filename
                safe_title = self.create_safe_filename(paper["title"])
                filename = f"{paper['arxiv_id']}_{safe_title}.pdf"
                filepath = self.data_folder / filename

                # Remove existing file if it exists to always download fresh copy
                if filepath.exists():
                    filepath.unlink()

                # Download PDF
                response = requests.get(paper["pdf_url"], stream=True, timeout=60)
                response.raise_for_status()

                # Save PDF
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                paper["downloaded"] = True
                paper["pdf_filename"] = filename
                print(f"✅ Successfully downloaded: {filename}")
                return paper

            except requests.RequestException as e:
                print(f"❌ Error downloading {paper['title'][:30]}: {e}")
                paper["downloaded"] = False
                return paper
            except Exception as e:
                print(f"❌ Unexpected error with {paper['title'][:30]}: {e}")
                paper["downloaded"] = False
                return paper

        # Use ThreadPoolExecutor for parallel downloads
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_paper = {executor.submit(download_single_paper, paper): paper for paper in papers}

            # Process completed downloads
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    result = future.result()
                    # Update the original paper object
                    paper.update(result)
                    completed_count += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(completed_count, len(papers), paper)
                        
                except Exception as e:
                    print(f"❌ Download task failed for {paper['title'][:30]}: {e}")
                    paper["downloaded"] = False
                    completed_count += 1
                    
                    # Call progress callback even for failed downloads
                    if progress_callback:
                        progress_callback(completed_count, len(papers), paper)

        print(f"🎉 Parallel download completed for {len(papers)} papers")

    def download_pdfs(self, papers=None):
        """
        Download PDFs for the papers (legacy sequential method).

        Args:
            papers (list): List of paper dictionaries. If None, uses self.papers_data
        """
        # Use parallel download by default
        self.download_pdfs_parallel(papers, max_workers=3)

    def create_safe_filename(self, title):
        """
        Create a safe filename from paper title.

        Args:
            title (str): Paper title

        Returns:
            str: Safe filename
        """
        # Remove special characters and limit length
        safe_name = re.sub(r"[^\w\s-]", "", title)
        safe_name = re.sub(r"[-\s]+", "_", safe_name)
        return safe_name[:50]  # Limit length

    def save_metadata(self, filename=None):
        """
        Save paper metadata to JSON file.

        Args:
            filename (str): Output filename. If None, uses timestamp
        """
        if not self.papers_data:
            print("No paper data to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_papers_{timestamp}.json"

        filepath = self.data_folder / filename

        # Prepare data for JSON serialization
        json_data = {
            "search_timestamp": datetime.now().isoformat(),
            "total_papers": len(self.papers_data),
            "papers": self.papers_data,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"Metadata saved to: {filepath}")
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def print_summary(self):
        """Print a summary of the scraped papers."""
        if not self.papers_data:
            print("No papers found")
            return

        print(f"\n{'='*60}")
        print(f"ARXIV PAPERS SUMMARY")
        print(f"{'='*60}")
        print(f"Total papers found: {len(self.papers_data)}")

        downloaded_count = sum(1 for paper in self.papers_data if paper["downloaded"])
        print(f"PDFs downloaded: {downloaded_count}/{len(self.papers_data)}")

        print(f"\n{'='*60}")
        print("PAPER DETAILS:")
        print(f"{'='*60}")

        for i, paper in enumerate(self.papers_data, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   ArXiv ID: {paper['arxiv_id']}")
            print(
                f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}"
            )
            print(f"   Published: {paper['published_date']}")
            print(f"   Categories: {', '.join(paper['categories'])}")
            print(f"   PDF Downloaded: {'Yes' if paper['downloaded'] else 'No'}")

            if paper["methodology"]:
                print(f"   Methodology highlights:")
                for method in paper["methodology"][
                    :2
                ]:  # Show first 2 methodology sentences
                    print(f"     • {method[:100]}{'...' if len(method) > 100 else ''}")

    def extract_single_page_image(self, paper_data, page_number):
        """
        Extract a single page from PDF as image for code generation.
        
        Args:
            paper_data (dict): Paper data dictionary containing pdf_filename
            page_number (int): Page number to extract (1-based)
            
        Returns:
            dict: Image information or None if failed
        """
        if not self._check_dependencies():
            return None
            
        if not paper_data.get("pdf_filename") or not paper_data.get("downloaded"):
            print(f"PDF not available for {paper_data['title'][:50]}...")
            return None
            
        pdf_path = self.data_folder / paper_data["pdf_filename"]
        if not pdf_path.exists():
            print(f"PDF file not found: {pdf_path}")
            return None
            
        try:
            # Create paper-specific image folder
            paper_image_folder = self.images_folder / paper_data["arxiv_id"]
            paper_image_folder.mkdir(exist_ok=True)
            
            print(f"🔬 Extracting page {page_number} from {paper_data['title'][:50]}...")
            
            # Convert specific PDF page to image using pdf2image
            try:
                pages = convert_from_path(
                    str(pdf_path), 
                    dpi=200,  # Higher DPI for better code extraction
                    first_page=page_number, 
                    last_page=page_number
                )
            except Exception as e:
                if "poppler" in str(e).lower() or "unable to get page count" in str(e).lower():
                    print("❌ Poppler not found! Please install Poppler:")
                    print("   Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
                    print("   Extract to C:\\poppler and add C:\\poppler\\bin to PATH")
                    print("   OR use conda: conda install -c conda-forge poppler")
                    return None
                else:
                    raise e
            
            if not pages:
                print(f"❌ Could not extract page {page_number}")
                return None
                
            page_image = pages[0]
            
            # Generate filename
            image_filename = f"page_{page_number}_code_extraction.png"
            image_path = paper_image_folder / image_filename
            
            # Save the page image with high quality for code extraction
            page_image.save(image_path, "PNG", quality=95, optimize=True)
            
            # Create image info
            image_info = {
                "filename": image_filename,
                "path": str(image_path),
                "page_number": page_number,
                "size": page_image.size,
                "label": f"Page {page_number} (Code Extraction)",
                "format": "PNG",
                "size_bytes": image_path.stat().st_size if image_path.exists() else 0,
                "type": "code_extraction",
                "extraction_method": "pdf2image",
                "dpi": 200
            }
            
            print(f"✅ Successfully extracted page {page_number} for code generation")
            return image_info
            
        except Exception as e:
            print(f"❌ Error extracting page {page_number} from PDF {pdf_path}: {e}")
            return None

    def get_pdf_page_count(self, paper_data):
        """
        Get the total number of pages in a PDF.
        
        Args:
            paper_data (dict): Paper data dictionary
            
        Returns:
            int: Number of pages or 0 if error
        """
        if not paper_data.get("pdf_filename") or not paper_data.get("downloaded"):
            return 0
            
        pdf_path = self.data_folder / paper_data["pdf_filename"]
        if not pdf_path.exists():
            return 0
            
        try:
            # Use PyMuPDF to get page count (lighter than pdf2image)
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            print(f"❌ Error getting page count: {e}")
            return 0

    def extract_images_from_pdf(self, paper_data):
        """
        DEPRECATED: This method has been replaced with on-demand page extraction.
        Use extract_single_page_image() for code generation instead.
        
        This method now only provides PDF metadata without extracting all pages.
        
        Args:
            paper_data (dict): Paper data dictionary containing pdf_filename
            
        Returns:
            list: Empty list (pages are now extracted on-demand)
        """
        if not self._check_dependencies():
            return []
            
        if not paper_data.get("pdf_filename") or not paper_data.get("downloaded"):
            print(f"PDF not available for {paper_data['title'][:50]}...")
            return []
            
        pdf_path = self.data_folder / paper_data["pdf_filename"]
        if not pdf_path.exists():
            print(f"PDF file not found: {pdf_path}")
            return []
            
        try:
            # Just get PDF metadata without extracting all pages
            page_count = self.get_pdf_page_count(paper_data)
            
            # Update paper data with PDF info
            paper_data["pdf_available"] = True
            paper_data["total_pages"] = page_count
            paper_data["images_extracted"] = False  # No longer auto-extracting
            
            print(f"📄 PDF ready for on-demand page extraction: {page_count} pages available")
            return []  # Return empty - pages extracted on demand
            
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {e}")
            return []
    
    
    def download_from_url(self, arxiv_url):
        """
        Download a paper from a specific ArXiv URL.
        
        Args:
            arxiv_url (str): ArXiv URL (e.g., https://arxiv.org/abs/2408.12345)
            
        Returns:
            dict: Paper data dictionary or None if failed
        """
        import re
        
        # Extract arXiv ID from URL
        arxiv_id_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', arxiv_url)
        if not arxiv_id_match:
            print(f"❌ Invalid ArXiv URL format: {arxiv_url}")
            return None
            
        arxiv_id = arxiv_id_match.group(1)
        
        try:
            print(f"🔍 Fetching paper info for {arxiv_id}...")
            
            # Use arXiv API to get paper metadata
            api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            # Parse the feed
            feed = feedparser.parse(response.content)
            
            if not feed.entries:
                print(f"❌ No paper found for ID: {arxiv_id}")
                return None
                
            # Extract paper data
            entry = feed.entries[0]
            paper_data = self.extract_paper_data(entry)
            
            print(f"✅ Found paper: {paper_data['title'][:50]}...")
            
            # Download the PDF
            print(f"📥 Downloading PDF...")
            self.download_pdfs([paper_data])
            
            return paper_data
            
        except Exception as e:
            print(f"❌ Error downloading from URL {arxiv_url}: {e}")
            return None

    def search_and_download(self, domain, max_results=10):
        """
        Search for papers and download them in one operation.
        
        Args:
            domain (str): ArXiv category
            max_results (int): Maximum number of papers
            
        Returns:
            list: List of downloaded papers
        """
        try:
            # Search for papers
            papers = self.search_papers(domain, max_results)
            
            if papers:
                # Download PDFs
                self.download_pdfs(papers)
                
            return papers
            
        except Exception as e:
            print(f"❌ Error in search and download: {e}")
            return []

    def deep_research_analysis(self, paper_data):
        """
        Perform deep research analysis with PDF preparation for on-demand code extraction.
        
        Args:
            paper_data (dict): Paper data dictionary
            
        Returns:
            dict: Enhanced paper data with PDF metadata
        """
        print(f"\n🔬 Starting deep research analysis for: {paper_data['title'][:50]}...")
        
        # Prepare PDF for on-demand page extraction (no bulk conversion)
        self.extract_images_from_pdf(paper_data)
        
        # Add deep research metadata
        paper_data.update({
            "deep_research_completed": True,
            "deep_research_timestamp": datetime.now().isoformat(),
            "code_extraction_ready": paper_data.get("pdf_available", False)
        })
        
        total_pages = paper_data.get("total_pages", 0)
        print(f"✅ Deep research analysis completed. PDF ready with {total_pages} pages for on-demand code extraction")
        
        return paper_data
