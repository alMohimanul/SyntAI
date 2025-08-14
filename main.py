#!/usr/bin/env python3
"""
Enhanced PaperWhisperer Workflow with Progress Tracking

This enhanced workflow provides real-time progress updates for the UI.
"""

import sys
import os
from pathlib import Path
from typing import TypedDict, Dict, Any, List, Callable, Optional
from datetime import datetime
import time

# Add the Agents directory to the Python path
agents_dir = Path(__file__).parent / "agents"
sys.path.append(str(agents_dir))

from agents.research_agent import ArxivScraper


class EnhancedWorkflowState(TypedDict):
    """Enhanced state with progress tracking"""

    domain: str
    max_results: int
    download_pdfs: bool
    deep_research: bool
    papers: List[Dict[str, Any]]
    metadata_saved: bool
    workflow_complete: bool
    error_message: str
    user_input: Dict[str, Any]
    progress_callback: Optional[Callable]
    current_step: str
    total_steps: int
    current_step_number: int


class ProgressTracker:
    """Simple progress tracker for workflow steps"""

    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.current_step = 0
        self.total_steps = 0
        self.current_message = ""

    def set_total_steps(self, total: int):
        self.total_steps = total
        self.current_step = 0

    def update(
        self, step: int, message: str, sub_progress: int = 0, sub_total: int = 0
    ):
        self.current_step = step
        self.current_message = message
        if self.callback:
            self.callback(step, self.total_steps, message, sub_progress, sub_total)

    def next_step(self, message: str):
        self.current_step += 1
        self.current_message = message
        if self.callback:
            self.callback(self.current_step, self.total_steps, message)


class EnhancedPaperWhispererWorkflow:
    """Enhanced workflow with progress tracking"""

    def __init__(self):
        """Initialize the enhanced workflow"""
        self.progress_tracker = None
        self.scraper = None

    def run(
        self,
        domain: str,
        max_results: int = 10,
        deep_research: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the enhanced workflow with progress tracking

        Args:
            domain: arXiv domain to search
            max_results: Number of papers to fetch
            deep_research: Whether to perform deep analysis
            progress_callback: Callback function for progress updates

        Returns:
            List of processed papers
        """

        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(progress_callback)
        self.progress_tracker.set_total_steps(5)

        papers = []

        try:
            # Step 1: Initialize
            self.progress_tracker.update(1, "Initializing research agent...")
            self.scraper = ArxivScraper("data")
            time.sleep(0.3)  # Brief pause for UI effect

            # Step 2: Search papers
            self.progress_tracker.update(2, f"Searching {domain} for papers...")
            papers = self.scraper.search_papers(domain, max_results)

            if not papers:
                self.progress_tracker.update(5, "No papers found", 0, 0)
                return []

            # Step 3: Download PDFs with sub-progress
            self.progress_tracker.update(3, "Downloading research papers...")
            self._download_with_progress(papers)

            # Step 4: Deep research analysis
            if deep_research:
                self.progress_tracker.update(4, "Performing deep neural analysis...")
                self._deep_analysis_with_progress(papers)
            else:
                self.progress_tracker.update(4, "Skipping deep analysis...")
                time.sleep(0.2)

            # Step 5: Complete
            self.progress_tracker.update(5, "Research mission complete!")

            return papers

        except Exception as e:
            self.progress_tracker.update(0, f"Error: {str(e)}")
            raise e

    def _download_with_progress(self, papers: List[Dict[str, Any]]):
        """Download PDFs with progress tracking"""
        total_papers = len(papers)

        for i, paper in enumerate(papers, 1):
            if self.progress_tracker.callback:
                self.progress_tracker.callback(
                    3, 5, f"Downloading paper {i}/{total_papers}...", i, total_papers
                )

            try:
                # Create safe filename
                safe_title = self.scraper.create_safe_filename(paper["title"])
                filename = f"{paper['arxiv_id']}_{safe_title}.pdf"
                filepath = self.scraper.data_folder / filename

                # Remove existing file if it exists
                if filepath.exists():
                    filepath.unlink()

                # Download PDF
                import requests

                response = requests.get(paper["pdf_url"], stream=True, timeout=60)
                response.raise_for_status()

                # Save PDF
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                paper["downloaded"] = True
                paper["pdf_filename"] = filename

                # Brief pause to make progress visible
                time.sleep(0.1)

            except Exception as e:
                print(f"Error downloading {paper['title'][:30]}: {e}")
                paper["downloaded"] = False

    def _deep_analysis_with_progress(self, papers: List[Dict[str, Any]]):
        """Perform deep analysis with progress tracking"""
        downloaded_papers = [p for p in papers if p.get("downloaded")]
        total_papers = len(downloaded_papers)

        for i, paper in enumerate(downloaded_papers, 1):
            if self.progress_tracker.callback:
                self.progress_tracker.callback(
                    4, 5, f"Analyzing paper {i}/{total_papers}...", i, total_papers
                )
            try:
                enhanced_paper = self.scraper.deep_research_analysis(paper)
                papers[papers.index(paper)] = enhanced_paper
                time.sleep(0.2)
            except Exception as e:
                print(f"Error in deep analysis for {paper['title'][:30]}: {e}")


enhanced_workflow = EnhancedPaperWhispererWorkflow()
