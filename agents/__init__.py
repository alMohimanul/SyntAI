"""
Agents Module

This directory contains all the agent implementations for PaperWhisperer.

Current Agents:
- research_agent.py - ArXiv paper scraping and metadata extraction
- More agents will be added here as the system expands

Agent Structure:
Each agent should follow these patterns:
- Be a self-contained class with clear methods
- Not include main() functions (those go in main.py)
- Return structured data that can be used by other agents
- Handle errors gracefully
- Include proper documentation
"""

from .research_agent import ArxivScraper

__all__ = ["ArxivScraper"]
