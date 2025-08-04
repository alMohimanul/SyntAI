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
import requests
import feedparser
from datetime import datetime, timedelta
from urllib.parse import urljoin
from pathlib import Path


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

    def search_papers(
        self, domain, max_results=10, sort_by="submittedDate", sort_order="descending"
    ):
        """
        Search for papers in a specific domain on arXiv.

        Args:
            domain (str): arXiv category (e.g., 'cs.AI', 'cs.CV', 'cs.LG', 'math.CO')
            max_results (int): Maximum number of papers to retrieve
            sort_by (str): Sort criteria ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order (str): Sort order ('ascending', 'descending')

        Returns:
            list: List of paper dictionaries
        """
        print(f"Searching for papers in domain: {domain}")
        print(f"Fetching {max_results} most recent papers...")

        # Construct the query
        query = f"cat:{domain}"
        params = {
            "search_query": query,
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

            print(f"Successfully found {len(papers)} papers")
            self.papers_data = papers
            return papers

        except requests.RequestException as e:
            print(f"Error fetching papers: {e}")
            return []
        except Exception as e:
            print(f"Error parsing feed: {e}")
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

    def download_pdfs(self, papers=None):
        """
        Download PDFs for the papers.

        Args:
            papers (list): List of paper dictionaries. If None, uses self.papers_data
        """
        if papers is None:
            papers = self.papers_data

        if not papers:
            print("No papers to download")
            return

        print(f"Starting download of {len(papers)} PDFs...")

        for i, paper in enumerate(papers, 1):
            try:
                print(f"Downloading {i}/{len(papers)}: {paper['title'][:50]}...")

                # Create safe filename
                safe_title = self.create_safe_filename(paper["title"])
                filename = f"{paper['arxiv_id']}_{safe_title}.pdf"
                filepath = self.data_folder / filename

                # Skip if already downloaded
                if filepath.exists():
                    print(f"  Already exists: {filename}")
                    paper["downloaded"] = True
                    paper["pdf_filename"] = filename
                    continue

                # Download PDF
                response = requests.get(paper["pdf_url"], stream=True, timeout=60)
                response.raise_for_status()

                # Save PDF
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                paper["downloaded"] = True
                paper["pdf_filename"] = filename
                print(f"  Successfully downloaded: {filename}")

                # Be respectful to the server
                time.sleep(1)

            except requests.RequestException as e:
                print(f"  Error downloading {paper['title'][:30]}: {e}")
                paper["downloaded"] = False
            except Exception as e:
                print(f"  Unexpected error with {paper['title'][:30]}: {e}")
                paper["downloaded"] = False

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
