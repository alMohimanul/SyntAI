#!/usr/bin/env python3
"""
PaperWhisperer - Main Application with LangGraph Workflow

This is the main entry point for the PaperWhisperer application.
It orchestrates multiple agents using LangGraph workflows.

Features:
- LangGraph-based workflow orchestration
- Research Agent for arXiv paper scraping
- Modular agent architecture
- Interactive CLI interface
"""

import sys
import os
from pathlib import Path
from typing import TypedDict, Dict, Any, List
from datetime import datetime

# Add the Agents directory to the Python path
agents_dir = Path(__file__).parent / "agents"
sys.path.append(str(agents_dir))

from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnablePassthrough
from agents.research_agent import ArxivScraper


class WorkflowState(TypedDict):
    """
    State object that flows through the workflow.
    Contains all data and parameters needed by different agents.
    """

    domain: str
    max_results: int
    download_pdfs: bool
    papers: List[Dict[str, Any]]
    metadata_saved: bool
    workflow_complete: bool
    error_message: str
    user_input: Dict[str, Any]


class PaperWhispererWorkflow:
    """
    Main workflow orchestrator using LangGraph.
    Coordinates different agents to accomplish the paper research task.
    """

    def __init__(self):
        """Initialize the workflow with agents and graph structure."""
        self.research_agent = ArxivScraper(data_folder="data")
        self.workflow_graph = None
        self.build_graph()

    def build_graph(self):
        """Build the LangGraph workflow graph."""
        # Create the workflow graph with state schema
        workflow = StateGraph(WorkflowState)

        # Add nodes (each representing a step in the workflow)
        workflow.add_node("collect_user_input", self.collect_user_input)
        workflow.add_node("search_papers", self.search_papers_node)
        workflow.add_node("download_papers", self.download_papers_node)
        workflow.add_node("save_metadata", self.save_metadata_node)
        workflow.add_node("generate_summary", self.generate_summary_node)

        # Define the workflow edges (flow between nodes)
        workflow.add_edge(START, "collect_user_input")
        workflow.add_edge("collect_user_input", "search_papers")
        workflow.add_edge("search_papers", "download_papers")
        workflow.add_edge("download_papers", "save_metadata")
        workflow.add_edge("save_metadata", "generate_summary")
        workflow.add_edge("generate_summary", END)

        self.workflow_graph = workflow.compile()

    def collect_user_input(self, state: WorkflowState) -> WorkflowState:
        """
        Collect user input for the research parameters.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with user parameters
        """
        print("🔬 PaperWhisperer - ArXiv Research Assistant")
        print("=" * 50)

        # Display common arXiv domains
        print("\n📚 Common arXiv domains:")
        domains = {
            "cs.AI": "Artificial Intelligence",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.LG": "Machine Learning",
            "cs.CL": "Computation and Language",
            "cs.NE": "Neural and Evolutionary Computing",
            "cs.RO": "Robotics",
            "cs.CR": "Cryptography and Security",
            "stat.ML": "Machine Learning (Statistics)",
            "math.CO": "Combinatorics",
            "physics.data-an": "Data Analysis, Statistics and Probability",
        }

        for code, description in domains.items():
            print(f"  {code:12} - {description}")

        # Get domain from user
        domain = input("\n🎯 Enter arXiv domain (e.g., cs.AI): ").strip()
        if not domain:
            print("No domain specified. Using cs.AI as default.")
            domain = "cs.AI"

        # Get number of papers
        try:
            max_results = int(
                input("📊 Number of papers to fetch (default 10): ") or "10"
            )
        except ValueError:
            max_results = 10

        # Ask about PDF downloads
        download_choice = input("📥 Download PDFs? (y/n, default y): ").strip().lower()
        download_pdfs = download_choice != "n"

        # Update state
        state["domain"] = domain
        state["max_results"] = max_results
        state["download_pdfs"] = download_pdfs
        state["user_input"] = {
            "domain": domain,
            "max_results": max_results,
            "download_pdfs": download_pdfs,
            "timestamp": datetime.now().isoformat(),
        }

        print(
            f"\n✅ Configuration set: {domain}, {max_results} papers, {'with' if download_pdfs else 'without'} PDFs"
        )
        return state

    def search_papers_node(self, state: WorkflowState) -> WorkflowState:
        """
        Search for papers using the research agent.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with found papers
        """
        print(f"\n🔍 Searching for papers in {state['domain']}...")

        try:
            papers = self.research_agent.search_papers(
                domain=state["domain"], max_results=state["max_results"]
            )

            state["papers"] = papers

            if papers:
                print(f"✅ Found {len(papers)} papers")
            else:
                state["error_message"] = "No papers found for the specified domain"
                print("❌ No papers found")

        except Exception as e:
            state["error_message"] = f"Error searching papers: {str(e)}"
            state["papers"] = []
            print(f"❌ Error: {str(e)}")

        return state

    def download_papers_node(self, state: WorkflowState) -> WorkflowState:
        """
        Download PDFs if requested.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state after download attempt
        """
        if not state["download_pdfs"]:
            print("📥 Skipping PDF downloads as requested")
            return state

        if not state["papers"]:
            print("📥 No papers to download")
            return state

        print(f"\n📥 Starting PDF downloads...")

        try:
            self.research_agent.download_pdfs(state["papers"])
            print("✅ PDF downloads completed")
        except Exception as e:
            print(f"⚠️  Some downloads may have failed: {str(e)}")

        return state

    def save_metadata_node(self, state: WorkflowState) -> WorkflowState:
        """
        Save metadata to JSON file.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state after saving metadata
        """
        if not state["papers"]:
            print("💾 No metadata to save")
            state["metadata_saved"] = False
            return state

        print("\n💾 Saving metadata...")

        try:
            self.research_agent.save_metadata()
            state["metadata_saved"] = True
            print("✅ Metadata saved successfully")
        except Exception as e:
            state["metadata_saved"] = False
            state["error_message"] = f"Error saving metadata: {str(e)}"
            print(f"❌ Error saving metadata: {str(e)}")

        return state

    def generate_summary_node(self, state: WorkflowState) -> WorkflowState:
        """
        Generate and display a summary of the workflow results.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state marking completion
        """
        print("\n📊 Generating summary...")

        try:
            if state["papers"]:
                self.research_agent.print_summary()
                print(
                    f"\n📁 All files saved in: {self.research_agent.data_folder.absolute()}"
                )
            else:
                print("❌ No papers were found or processed")

            state["workflow_complete"] = True

        except Exception as e:
            print(f"⚠️  Error generating summary: {str(e)}")
            state["workflow_complete"] = False

        return state

    def run_workflow(self, initial_state: WorkflowState = None) -> WorkflowState:
        """
        Execute the complete workflow.

        Args:
            initial_state: Optional initial state. If None, creates empty state.

        Returns:
            Final workflow state
        """
        if initial_state is None:
            initial_state = WorkflowState(
                domain="",
                max_results=10,
                download_pdfs=True,
                papers=[],
                metadata_saved=False,
                workflow_complete=False,
                error_message="",
                user_input={},
            )

        try:
            # Execute the workflow
            final_state = self.workflow_graph.invoke(initial_state)
            return final_state

        except Exception as e:
            print(f"❌ Workflow execution failed: {str(e)}")
            initial_state["error_message"] = str(e)
            initial_state["workflow_complete"] = False
            return initial_state


def create_batch_workflow(
    domains: List[str], max_results: int = 10, download_pdfs: bool = True
) -> List[WorkflowState]:
    """
    Run the workflow for multiple domains in batch mode.

    Args:
        domains: List of arXiv domains to process
        max_results: Number of papers per domain
        download_pdfs: Whether to download PDFs

    Returns:
        List of final states for each domain
    """
    workflow = PaperWhispererWorkflow()
    results = []

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain}")
        print(f"{'='*60}")

        initial_state = WorkflowState(
            domain=domain,
            max_results=max_results,
            download_pdfs=download_pdfs,
            papers=[],
            metadata_saved=False,
            workflow_complete=False,
            error_message="",
            user_input={
                "domain": domain,
                "max_results": max_results,
                "download_pdfs": download_pdfs,
                "batch_mode": True,
            },
        )

        # Skip user input collection for batch mode
        final_state = workflow.workflow_graph.invoke(initial_state)
        results.append(final_state)

    return results


def main():
    """
    Main entry point for the PaperWhisperer application.
    """
    try:
        # Ask user for mode
        print("🔬 PaperWhisperer - ArXiv Research Assistant")
        print("=" * 50)
        print("\nSelect mode:")
        print("1. Interactive mode (default)")
        print("2. Batch mode (multiple domains)")

        mode = input("\nEnter mode (1 or 2): ").strip()

        if mode == "2":
            # Batch mode
            print("\nBatch Mode - Enter domains separated by commas")
            print("Example: cs.AI, cs.CV, cs.LG")
            domains_input = input("Domains: ").strip()

            if not domains_input:
                print("No domains specified. Exiting.")
                return

            domains = [d.strip() for d in domains_input.split(",")]

            try:
                max_results = int(input("Papers per domain (default 5): ") or "5")
            except ValueError:
                max_results = 5

            download_pdfs = (
                input("Download PDFs? (y/n, default y): ").strip().lower() != "n"
            )

            # Run batch workflow
            results = create_batch_workflow(domains, max_results, download_pdfs)

            # Print batch summary
            print(f"\n{'='*60}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")

            total_papers = 0
            for i, result in enumerate(results):
                domain = domains[i]
                papers_found = len(result.get("papers", []))
                total_papers += papers_found
                status = (
                    "✅ Success" if result.get("workflow_complete") else "❌ Failed"
                )
                print(f"{domain:15} - {papers_found:3d} papers - {status}")

                if result.get("error_message"):
                    print(f"                  Error: {result['error_message']}")

            print(f"\nTotal papers processed: {total_papers}")

        else:
            # Interactive mode
            workflow = PaperWhispererWorkflow()
            final_state = workflow.run_workflow()

            if final_state["workflow_complete"]:
                print("\n🎉 Workflow completed successfully!")
            else:
                print("\n❌ Workflow completed with errors")
                if final_state.get("error_message"):
                    print(f"Error: {final_state['error_message']}")

    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
