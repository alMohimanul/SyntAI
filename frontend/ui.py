#!/usr/bin/env python3
"""
PaperWhisperer - SyntAI Futuristic Research Interface

A professional, futuristic interface for the PaperWhisperer arXiv research assistant.
Features comprehensive progress tracking, neural-themed UI, and advanced paper analysis.

Author: SyntAI Development Team
Version: 2.0.0
Created: August 2025
"""

# Standard Library Imports
import importlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    import feedparser
except ImportError:
    feedparser = None
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

main_dir = Path(__file__).parent.parent
sys.path.append(str(main_dir))

from main import enhanced_workflow
import service.llm

importlib.reload(service.llm)
from service.llm import code_generation_service
from service.pdf_parser import pdf_text_extractor

try:
    from service.llm import PaperSummarizationService

    paper_summarization_service = PaperSummarizationService()
except ImportError:
    paper_summarization_service = None

try:
    from agents.research_agent import ArxivScraper
    import fitz
except ImportError:
    ArxivScraper = None
    fitz = None

# Configure Streamlit page settings
st.set_page_config(
    page_title="SyntAI Neural Research Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css():
    """
    Load comprehensive futuristic CSS styling for SyntAI interface.

    This function injects custom CSS styles that create a professional,
    sci-fi inspired interface with:
    - Futuristic color scheme (dark blue gradients, cyan accents)
    - Modern typography (Space Grotesk, JetBrains Mono)
    - Animated elements (spinners, glows, transitions)
    - Professional card layouts and components
    - Responsive design for various screen sizes

    The styling includes:
    - Main app background and typography
    - Custom spinners and loading animations
    - Paper card components with gradient backgrounds
    - Button and form element styling
    - Navigation and interface elements
    - Accessibility improvements
    """
    st.markdown(
        """
        <style>
        /* Import futuristic fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        
        /* Futuristic theme */
        .stApp {
            font-family: 'Space Grotesk', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e6ed;
        }
        
        /* Header */
        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #00d4ff;
            text-align: center;
            margin-bottom: 0.5rem;
            text-transform: none;
            letter-spacing: 0.1em;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.6), 0 0 60px rgba(0, 212, 255, 0.4);
            filter: brightness(1.3);
            position: relative;
        }
        
        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-weight: 300;
            letter-spacing: 0.05em;
        }
        
        /* Cards */
        .paper-card {
            background: linear-gradient(145deg, #1e293b, #334155);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(100, 116, 139, 0.2);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .paper-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, transparent);
            transition: left 0.5s ease;
        }
        
        .paper-card:hover::before {
            left: 100%;
        }
        
        .paper-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 212, 255, 0.15);
        }
        
        .paper-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }
        
        .paper-meta {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Agent Status */
        .agent-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            background: rgba(15, 15, 35, 0.8);
            border-radius: 8px;
            border: 1px solid rgba(100, 116, 139, 0.3);
            margin: 1rem 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }
        
        .agent-spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-top: 2px solid #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Progress bars */
        .progress-container {
            background: rgba(15, 15, 35, 0.6);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid rgba(100, 116, 139, 0.2);
        }
        
        .progress-bar {
            background: rgba(100, 116, 139, 0.3);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #06ffa5);
            border-radius: 4px;
            transition: width 0.3s ease;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(45deg, #00d4ff, #5b21b6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 500 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
            position: relative !important;
            overflow: hidden !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4) !important;
            background: linear-gradient(45deg, #00d4ff, #5b21b6) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px) !important;
            box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3) !important;
        }
        
        /* Enhanced form inputs */
        .stSelectbox > div > div {
            background: rgba(15, 15, 35, 0.9) !important;
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
            border-radius: 8px !important;
            color: #f1f5f9 !important;
            transition: all 0.3s ease !important;
            box-shadow: inset 0 0 10px rgba(0, 212, 255, 0.1) !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: rgba(0, 212, 255, 0.6) !important;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.2) !important;
        }
        
        .stNumberInput > div > div > input {
            background: rgba(15, 15, 35, 0.9) !important;
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
            border-radius: 8px !important;
            color: #f1f5f9 !important;
            transition: all 0.3s ease !important;
            box-shadow: inset 0 0 10px rgba(0, 212, 255, 0.1) !important;
        }
        
        .stNumberInput > div > div > input:focus {
            border-color: rgba(0, 212, 255, 0.8) !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
            backdrop-filter: blur(10px);
        }
        
        .status-success {
            background: rgba(6, 255, 165, 0.2);
            color: #06ffa5;
            border: 1px solid rgba(6, 255, 165, 0.3);
            box-shadow: 0 0 20px rgba(6, 255, 165, 0.2);
        }
        
        .status-warning {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
            border: 1px solid rgba(255, 193, 7, 0.3);
            box-shadow: 0 0 20px rgba(255, 193, 7, 0.2);
        }
        
        .status-error {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
        }
        
        .status-processing {
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            border: 1px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Code styling */
        .stCode {
            background: rgba(15, 15, 35, 0.9) !important;
            border: 1px solid rgba(100, 116, 139, 0.3) !important;
            border-radius: 8px !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: linear-gradient(145deg, #1e293b, #334155) !important;
            border: 1px solid rgba(100, 116, 139, 0.2) !important;
            border-radius: 8px !important;
            color: #f1f5f9 !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }
        
        /* Mission Configuration */
        .mission-config-header {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.1), rgba(91, 33, 182, 0.1));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        }
        
        .mission-config-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, #5b21b6, #00d4ff, transparent);
            animation: scan 2s linear infinite;
        }
        
        @keyframes scan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .mission-title {
            font-size: 1.4rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            color: #00d4ff;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        
        .mission-subtitle {
            font-size: 0.9rem;
            color: #94a3b8;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 300;
            letter-spacing: 0.05em;
        }
        
        .mission-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #00d4ff, transparent);
            margin: 1.5rem 0;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }
        
        /* Enhanced form inputs */
        .stSelectbox > div > div {
            background: rgba(15, 15, 35, 0.9) !important;
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
            border-radius: 8px !important;
            color: #f1f5f9 !important;
            transition: all 0.3s ease !important;
            box-shadow: inset 0 0 10px rgba(0, 212, 255, 0.1) !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: rgba(0, 212, 255, 0.6) !important;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.2) !important;
        }
        
        .stNumberInput > div > div > input {
            background: rgba(15, 15, 35, 0.9) !important;
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
            border-radius: 8px !important;
            color: #f1f5f9 !important;
            transition: all 0.3s ease !important;
            box-shadow: inset 0 0 10px rgba(0, 212, 255, 0.1) !important;
        }
        
        .stNumberInput > div > div > input:focus {
            border-color: rgba(0, 212, 255, 0.8) !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        }
        
        /* Ultra-Futuristic Database Section */
        .syntai-database-header {
            background: linear-gradient(135deg, 
                rgba(6, 255, 165, 0.1) 0%, 
                rgba(0, 212, 255, 0.15) 30%,
                rgba(139, 69, 19, 0.05) 70%,
                rgba(147, 51, 234, 0.1) 100%);
            padding: 3rem 2rem;
            border-radius: 25px;
            border: 1px solid rgba(6, 255, 165, 0.3);
            margin: 2rem 0;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(25px);
        }
        
        .syntai-database-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, 
                transparent, 
                rgba(6, 255, 165, 0.1), 
                rgba(0, 212, 255, 0.1), 
                transparent);
            animation: scanAnimation 3s ease-in-out infinite;
        }
        
        @keyframes scanAnimation {
            0% { left: -100%; }
            50% { left: 100%; }
            100% { left: -100%; }
        }
        
        .database-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #06ffa5, #00d4ff, #9333ea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-family: 'Space Grotesk', sans-serif;
            text-shadow: 0 0 30px rgba(6, 255, 165, 0.5);
        }
        
        .database-subtitle {
            text-align: center;
            color: rgba(248, 250, 252, 0.8);
            font-size: 1.2rem;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 1.5rem;
        }
        
        .papers-count-display {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .count-badge {
            background: linear-gradient(135deg, rgba(6, 255, 165, 0.2), rgba(0, 212, 255, 0.2));
            border: 1px solid rgba(6, 255, 165, 0.4);
            border-radius: 50px;
            padding: 1rem 2rem;
            font-size: 1.5rem;
            font-weight: 700;
            color: #06ffa5;
            font-family: 'Space Grotesk', sans-serif;
            box-shadow: 0 0 20px rgba(6, 255, 165, 0.3);
            backdrop-filter: blur(10px);
        }
        
        /* Revolutionary Paper Cards Grid */
        .papers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
        }
        
        .quantum-paper-card {
            background: linear-gradient(145deg, 
                rgba(15, 23, 42, 0.95) 0%,
                rgba(30, 41, 59, 0.9) 50%,
                rgba(51, 65, 85, 0.85) 100%);
            border: 1px solid rgba(6, 255, 165, 0.2);
            border-radius: 25px;
            padding: 2.5rem;
            position: relative;
            overflow: hidden;
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            backdrop-filter: blur(30px);
            box-shadow: 
                0 10px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                0 0 0 1px rgba(6, 255, 165, 0.1);
        }
        
        .quantum-paper-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, 
                rgba(6, 255, 165, 0.05) 0%,
                transparent 30%,
                transparent 70%,
                rgba(0, 212, 255, 0.05) 100%);
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .quantum-paper-card::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, 
                #06ffa5, #00d4ff, #9333ea, #06ffa5);
            border-radius: 27px;
            opacity: 0;
            z-index: -1;
            animation: borderGlow 3s ease-in-out infinite;
        }
        
        @keyframes borderGlow {
            0%, 100% { opacity: 0; }
            50% { opacity: 0.6; }
        }
        
        .quantum-paper-card:hover {
            transform: translateY(-12px) scale(1.03);
            border-color: rgba(6, 255, 165, 0.6);
            box-shadow: 
                0 25px 80px rgba(6, 255, 165, 0.25),
                0 0 0 1px rgba(6, 255, 165, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        
        .quantum-paper-card:hover::before {
            opacity: 1;
        }
        
        .quantum-paper-card:hover::after {
            opacity: 0.8;
        }
        
        .paper-neural-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 2rem;
            position: relative;
        }
        
        .paper-title-quantum {
            font-size: 1.3rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.4;
            font-family: 'Space Grotesk', sans-serif;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            flex: 1;
            margin-right: 1rem;
        }
        
        .neural-status-indicator {
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            backdrop-filter: blur(10px);
        }
        
        .status-neural-ready {
            background: linear-gradient(135deg, rgba(6, 255, 165, 0.2), rgba(34, 197, 94, 0.2));
            border: 1px solid rgba(6, 255, 165, 0.5);
            color: #06ffa5;
            box-shadow: 0 0 15px rgba(6, 255, 165, 0.3);
        }
        
        .status-neural-processing {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.2));
            border: 1px solid rgba(251, 191, 36, 0.5);
            color: #fbbf24;
            box-shadow: 0 0 15px rgba(251, 191, 36, 0.3);
        }
        
        .paper-meta-neural {
            background: linear-gradient(135deg, 
                rgba(15, 15, 35, 0.6) 0%,
                rgba(30, 30, 60, 0.4) 100%);
            padding: 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            margin-bottom: 2rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.8;
            backdrop-filter: blur(15px);
        }
        
        .meta-field {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.8rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .meta-field:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        
        .meta-label {
            color: #00d4ff;
            font-weight: 600;
            min-width: 100px;
        }
        
        .meta-value {
            color: #cbd5e1;
            text-align: right;
            flex: 1;
        }
        
        /* Revolutionary Action Buttons */
        .neural-actions-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 2rem;
        }
        
        .quantum-action-btn {
            background: linear-gradient(135deg, 
                rgba(6, 255, 165, 0.1) 0%,
                rgba(0, 212, 255, 0.1) 100%);
            border: 1px solid rgba(6, 255, 165, 0.3);
            border-radius: 15px;
            padding: 1rem 1.5rem;
            color: #06ffa5;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            text-decoration: none;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .quantum-action-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, 
                transparent, 
                rgba(6, 255, 165, 0.2), 
                transparent);
            transition: left 0.4s ease;
        }
        
        .quantum-action-btn:hover {
            transform: translateY(-2px);
            border-color: rgba(6, 255, 165, 0.6);
            box-shadow: 0 8px 25px rgba(6, 255, 165, 0.25);
            color: #ffffff;
        }
        
        .quantum-action-btn:hover::before {
            left: 100%;
        }
        
        .btn-preview { 
            border-color: rgba(0, 212, 255, 0.3);
            color: #00d4ff;
        }
        .btn-preview:hover { 
            border-color: rgba(0, 212, 255, 0.6);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.25);
        }
        
        .btn-analysis { 
            border-color: rgba(147, 51, 234, 0.3);
            color: #9333ea;
        }
        .btn-analysis:hover { 
            border-color: rgba(147, 51, 234, 0.6);
            box-shadow: 0 8px 25px rgba(147, 51, 234, 0.25);
        }
        
        .btn-download { 
            border-color: rgba(34, 197, 94, 0.3);
            color: #22c55e;
        }
        .btn-download:hover { 
            border-color: rgba(34, 197, 94, 0.6);
            box-shadow: 0 8px 25px rgba(34, 197, 94, 0.25);
        }

        /* Enhanced Papers Acquired Section */
        .papers-section-header {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.15), rgba(91, 33, 182, 0.08));
            border: 1px solid rgba(0, 212, 255, 0.4);
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0 1.5rem 0;
            position: relative;
            overflow: hidden;
            text-align: center;
            box-shadow: 0 12px 48px rgba(0, 212, 255, 0.12);
            backdrop-filter: blur(15px);
        }
        
        .papers-section-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, #00d4ff, #06ffa5, #5b21b6, #00d4ff, transparent);
            animation: header-scan 3s linear infinite;
        }
        
        .papers-section-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent);
            animation: footer-scan 4s linear infinite reverse;
        }
        
        @keyframes header-scan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        @keyframes footer-scan {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        .papers-section-title {
            font-size: 2.2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #00d4ff;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin: 0 0 0.5rem 0;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.4);
            filter: brightness(1.2);
        }
        
        .papers-section-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            font-weight: 300;
            letter-spacing: 0.05em;
            font-family: 'Space Grotesk', sans-serif;
            opacity: 0.9;
        }
        
        /* Enhanced Paper Cards */
        .enhanced-paper-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(51, 65, 85, 0.8));
            border: 1px solid rgba(0, 212, 255, 0.25);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            backdrop-filter: blur(20px);
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .enhanced-paper-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, #06ffa5, transparent);
            transition: left 0.6s ease;
        }
        
        .enhanced-paper-card::after {
            content: '';
            position: absolute;
            inset: 0;
            padding: 1px;
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.2), transparent, rgba(91, 33, 182, 0.2));
            border-radius: 20px;
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            mask-composite: xor;
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .enhanced-paper-card:hover::before {
            left: 100%;
        }
        
        .enhanced-paper-card:hover::after {
            opacity: 1;
        }
        
        .enhanced-paper-card:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: rgba(0, 212, 255, 0.5);
            box-shadow: 
                0 20px 60px rgba(0, 212, 255, 0.25),
                0 0 0 1px rgba(0, 212, 255, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        
        .enhanced-paper-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 1rem;
            line-height: 1.4;
            font-family: 'Space Grotesk', sans-serif;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .enhanced-paper-meta {
            color: #cbd5e1;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.6;
            background: rgba(15, 15, 35, 0.4);
            padding: 1rem;
            border-radius: 12px;
            border-left: 3px solid #00d4ff;
        }
        
        .enhanced-paper-meta strong {
            color: #00d4ff;
            text-shadow: 0 0 8px rgba(0, 212, 255, 0.5);
        }
        
        .enhanced-paper-meta a {
            color: #06ffa5 !important;
            text-decoration: none !important;
            transition: all 0.3s ease;
            text-shadow: 0 0 8px rgba(6, 255, 165, 0.5);
        }
        
        .enhanced-paper-meta a:hover {
            color: #00d4ff !important;
            text-shadow: 0 0 12px rgba(0, 212, 255, 0.8);
        }
        
        /* Status Indicators */
        .paper-status-indicator {
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            backdrop-filter: blur(10px);
            animation: pulse-glow 2s ease-in-out infinite alternate;
        }
        
        .status-downloaded {
            background: rgba(6, 255, 165, 0.2);
            color: #06ffa5;
            border: 1px solid rgba(6, 255, 165, 0.4);
            box-shadow: 0 0 20px rgba(6, 255, 165, 0.3);
        }
        
        .status-pending {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
            border: 1px solid rgba(255, 193, 7, 0.4);
            box-shadow: 0 0 20px rgba(255, 193, 7, 0.3);
        }
        
        @keyframes pulse-glow {
            0% { 
                box-shadow: 0 0 5px currentColor; 
                transform: scale(1);
            }
            100% { 
                box-shadow: 0 0 20px currentColor, 0 0 30px currentColor; 
                transform: scale(1.05);
            }
        }
        
        /* Enhanced Action Buttons Grid */
        .action-buttons-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }
        
        .enhanced-action-button {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.1), rgba(91, 33, 182, 0.1));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            cursor: pointer;
            backdrop-filter: blur(10px);
            text-decoration: none !important;
            color: inherit !important;
        }
        
        .enhanced-action-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
            transition: left 0.6s ease;
        }
        
        .enhanced-action-button:hover::before {
            left: 100%;
        }
        
        .enhanced-action-button:hover {
            transform: translateY(-4px) scale(1.05);
            border-color: rgba(0, 212, 255, 0.6);
            box-shadow: 0 8px 30px rgba(0, 212, 255, 0.25);
        }
        
        .button-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            display: block;
            filter: drop-shadow(0 0 8px currentColor);
        }
        
        .button-label {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.85rem;
            color: #00d4ff;
        }
        
        .button-description {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 0.25rem;
            font-family: 'Space Grotesk', sans-serif;
        }
        
        /* Papers Grid Layout */
        .papers-grid {
            display: grid;
            gap: 2rem;
            margin: 2rem 0;
        }
        
        /* Enhanced Expander for Interactive Sections */
        .enhanced-expander {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.05), rgba(91, 33, 182, 0.05));
            border: 1px solid rgba(0, 212, 255, 0.25);
            border-radius: 12px;
            margin: 1rem 0;
            overflow: hidden;
        }
        
        /* Loading States */
        .papers-loading {
            text-align: center;
            padding: 3rem;
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.05), rgba(91, 33, 182, 0.05));
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 16px;
            margin: 2rem 0;
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(0, 212, 255, 0.2);
            border-top: 3px solid #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem auto;
        }
        
        .loading-text {
            color: #00d4ff;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        /* Enhanced Navigation */
        .page-navigation {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.08), rgba(91, 33, 182, 0.08));
            border: 1px solid rgba(0, 212, 255, 0.25);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }
        
        /* Analysis Results Styling */
        .analysis-result {
            background: linear-gradient(145deg, rgba(6, 255, 165, 0.05), rgba(0, 212, 255, 0.05));
            border: 1px solid rgba(6, 255, 165, 0.25);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #06ffa5;
        }
        
        .analysis-result h3 {
            color: #06ffa5;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.1), rgba(91, 33, 182, 0.1));
            border-radius: 12px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.2), rgba(91, 33, 182, 0.2));
            color: #00d4ff;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .papers-stats {
                flex-direction: column;
                gap: 1.5rem;
            }
            
            .action-buttons-grid {
                grid-template-columns: 1fr;
            }
            
            .enhanced-paper-title {
                font-size: 1.2rem;
            }
            
            .papers-section-title {
                font-size: 1.8rem;
            }
        }
        
        /* Accessibility Improvements */
        .enhanced-action-button:focus,
        .enhanced-paper-card:focus-within {
            outline: 2px solid #00d4ff;
            outline-offset: 2px;
        }
        
        /* Paper Card Components */
        .paper-card-header {
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        
        .paper-title {
            color: #f8fafc;
            font-size: 1.3rem;
            font-weight: 600;
            margin: 0;
            line-height: 1.4;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            font-family: 'Space Grotesk', sans-serif;
            flex: 1;
            padding-right: 1rem;
        }
        
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            backdrop-filter: blur(10px);
            white-space: nowrap;
        }
        
        .status-ready {
            background: rgba(6, 255, 165, 0.2);
            color: #06ffa5;
            border: 1px solid rgba(6, 255, 165, 0.4);
        }
        
        .status-processing {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
            border: 1px solid rgba(255, 193, 7, 0.4);
        }
        
        /* Smooth Transitions */
        * {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Enhanced Expander */
        .streamlit-expanderHeader {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.08), rgba(91, 33, 182, 0.08)) !important;
            border: 1px solid rgba(0, 212, 255, 0.25) !important;
            border-radius: 10px !important;
            color: #00d4ff !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 500 !important;
            padding: 1rem !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .streamlit-expanderHeader::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, transparent);
            transition: left 0.5s ease;
        }
        
        .streamlit-expanderHeader:hover::before {
            left: 100%;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: rgba(0, 212, 255, 0.5) !important;
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.2) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Analysis Configuration Headers */
        .analysis-header {
            font-family: 'JetBrains Mono', monospace !important;
            color: #00d4ff !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin: 1.5rem 0 1rem 0 !important;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.4) !important;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3) !important;
            padding-bottom: 0.5rem !important;
        }
        
        /* Futuristic Document Preview */
        .futuristic-preview-container {
            background: linear-gradient(145deg, rgba(10, 18, 30, 0.95), rgba(20, 30, 50, 0.9));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            position: relative;
            backdrop-filter: blur(20px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .futuristic-preview-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, #00d4ff, #06ffa5, #8b5cf6, transparent);
            border-radius: 20px 20px 0 0;
            animation: border-glow 3s ease-in-out infinite;
        }
        
        .preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .preview-title {
            color: #f8fafc;
            font-size: 1.4rem;
            font-weight: 600;
            font-family: 'Space Grotesk', sans-serif;
            text-shadow: 0 2px 8px rgba(0, 212, 255, 0.3);
        }
        
        .page-indicator {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.1), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 25px;
            padding: 0.8rem 1.5rem;
            color: #00d4ff;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
        }
        
        .jump-control {
            background: linear-gradient(145deg, rgba(15, 15, 35, 0.8), rgba(30, 30, 60, 0.6));
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
        }
        
        .document-image-container {
            position: relative;
            border-radius: 15px;
            overflow: hidden;
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 8px 30px rgba(0, 212, 255, 0.15);
            margin: 2rem 0;
        }
        
        /* Futuristic Spinner */
        .futuristic-spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            position: relative;
            margin: 0 10px;
        }
        
        .futuristic-spinner::before,
        .futuristic-spinner::after {
            content: '';
            position: absolute;
            border-radius: 50%;
            animation: spinner-pulse 2s ease-in-out infinite;
        }
        
        .futuristic-spinner::before {
            width: 100%;
            height: 100%;
            border: 2px solid transparent;
            border-top: 2px solid #00d4ff;
            border-right: 2px solid #06ffa5;
            animation: spinner-rotate 1s linear infinite;
        }
        
        .futuristic-spinner::after {
            width: 60%;
            height: 60%;
            top: 20%;
            left: 20%;
            border: 2px solid transparent;
            border-bottom: 2px solid #8b5cf6;
            border-left: 2px solid #06ffa5;
            animation: spinner-rotate 1.5s linear infinite reverse;
        }
        
        @keyframes spinner-rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes spinner-pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        
        @keyframes border-glow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Enhanced Metrics */
        .stMetric {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.05), rgba(91, 33, 182, 0.05)) !important;
            border: 1px solid rgba(0, 212, 255, 0.2) !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stMetric:hover {
            border-color: rgba(0, 212, 255, 0.4) !important;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.15) !important;
            transform: translateY(-2px) !important;
        }
        
        .stMetric > div > div {
            color: #00d4ff !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important;
        }
        
        /* Button Alignment */
        .button-alignment-spacer {
            height: 1.7rem;
        }
        
        /* Text Area Styling for Code Display */
        .stTextArea > div > div > textarea {
            background: rgba(15, 15, 35, 0.95) !important;
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
            border-radius: 8px !important;
            color: #f8f8f2 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.9rem !important;
            line-height: 1.5 !important;
            resize: none !important;
            tab-size: 4 !important;
            white-space: pre !important;
            overflow-wrap: normal !important;
        }
        
        /* Disabled text area (read-only code display) */
        .stTextArea > div > div > textarea[disabled] {
            background: rgba(12, 12, 30, 0.98) !important;
            color: #e6e6e6 !important;
            cursor: text !important;
            opacity: 1 !important;
            border: 1px solid rgba(0, 212, 255, 0.4) !important;
            box-shadow: inset 0 0 15px rgba(0, 212, 255, 0.1) !important;
        }
        
        /* Code syntax-like styling for text area */
        .stTextArea > div > div > textarea {
            /* Python-like syntax coloring simulation */
            text-shadow: 
                0 0 5px rgba(102, 217, 239, 0.3),
                0 0 10px rgba(102, 217, 239, 0.1);
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: rgba(0, 212, 255, 0.8) !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
            outline: none !important;
        }
        
        /* Custom scrollbar for text area */
        .stTextArea > div > div > textarea::-webkit-scrollbar {
            width: 8px;
        }
        
        .stTextArea > div > div > textarea::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 4px;
        }
        
        .stTextArea > div > div > textarea::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #00d4ff, #5b21b6);
            border-radius: 4px;
        }
        
        .stTextArea > div > div > textarea::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(45deg, #00d4ff, #06ffa5);
        }
        
        /* Code area container styling */
        .stTextArea > label {
            color: #00d4ff !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        
        /* Keep expander styling for Report section only */
        .streamlit-expanderContent {
            max-height: 200px !important;
            overflow-y: auto !important;
        }
        
        /* Deep Dive Analysis Styles */
        .deep-dive-container {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.03), rgba(91, 33, 182, 0.03));
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }
        
        .deep-dive-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, #06ffa5, #00d4ff, transparent);
            animation: analysis-scan 3s linear infinite;
        }
        
        @keyframes analysis-scan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .insight-item {
            background: rgba(0, 212, 255, 0.05);
            border-left: 3px solid #00d4ff;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 6px;
            transition: all 0.3s ease;
        }
        
        .insight-item:hover {
            background: rgba(0, 212, 255, 0.08);
            border-left-color: #06ffa5;
            transform: translateX(5px);
        }
        
        .content-type-badge {
            background: linear-gradient(45deg, #00d4ff, #0099cc);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            display: inline-block;
            margin-bottom: 15px;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }
        
        .analysis-section {
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            background: rgba(0, 212, 255, 0.02);
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        
        .analysis-section h3 {
            color: #00d4ff !important;
            font-family: 'JetBrains Mono', monospace !important;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.4) !important;
        }
        
        /* Status indicators */
        .status-success {
            background: linear-gradient(45deg, rgba(6, 255, 165, 0.1), rgba(0, 212, 255, 0.1));
            border: 1px solid rgba(6, 255, 165, 0.3);
            color: #06ffa5;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        
        .status-warning {
            background: linear-gradient(45deg, rgba(255, 193, 7, 0.1), rgba(255, 152, 0, 0.1));
            border: 1px solid rgba(255, 193, 7, 0.3);
            color: #ffc107;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        
        .status-error {
            background: linear-gradient(45deg, rgba(220, 53, 69, 0.1), rgba(255, 82, 82, 0.1));
            border: 1px solid rgba(220, 53, 69, 0.3);
            color: #ff5252;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        
        /* SyntAI Action Buttons */
        .syntai-button-container {
            margin: 0.5rem 0;
        }
        
        .syntai-action-button {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 1rem 0.5rem;
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.1), rgba(91, 33, 182, 0.1));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            color: #00d4ff;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            text-decoration: none;
            min-height: 80px;
        }
        
        .syntai-action-button:hover {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.15), rgba(91, 33, 182, 0.15));
            border-color: rgba(0, 212, 255, 0.6);
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
            color: #06ffa5;
        }
        
        .syntai-action-button .button-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.6);
        }
        
        .syntai-action-button .button-text {
            font-size: 0.8rem;
            font-weight: 700;
            text-align: center;
            line-height: 1.2;
        }
        
        .syntai-action-button .button-scan {
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, #06ffa5, transparent);
            transition: left 0.6s ease;
        }
        
        .syntai-action-button:hover .button-scan {
            left: 100%;
        }
        
        /* Deep Dive Button Special Styling */
        .deep-dive-button {
            background: linear-gradient(145deg, rgba(6, 255, 165, 0.1), rgba(0, 212, 255, 0.1));
            border: 1px solid rgba(6, 255, 165, 0.4);
            color: #06ffa5;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.8rem 2rem;
            border-radius: 8px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            text-align: center;
            display: block;
            margin: 1rem auto;
            min-width: 200px;
        }
        
        .deep-dive-button:hover {
            background: linear-gradient(145deg, rgba(6, 255, 165, 0.2), rgba(0, 212, 255, 0.2));
            border-color: rgba(6, 255, 165, 0.8);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(6, 255, 165, 0.4);
            text-shadow: 0 0 15px rgba(6, 255, 165, 0.8);
        }
        
        .deep-dive-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(6, 255, 165, 0.2), transparent);
            transition: left 0.6s ease;
        }
        
        .deep-dive-button:hover::before {
            left: 100%;
        }
        
        /* Hide duplicate buttons for custom styling */
        [data-testid*="preview_"] .stButton,
        [data-testid*="btn_deep_dive_page_"] .stButton {
            display: none !important;
        }
        
        /* Hide form submit button behind custom styling */
        form[data-testid*="preview_form_"] button,
        form[data-testid*="deep_dive_form_"] button {
            opacity: 0 !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            z-index: 10 !important;
            cursor: pointer !important;
            border: none !important;
            background: transparent !important;
        }
        
        /* Make the custom button container clickable */
        .syntai-button-container,
        .deep-dive-button {
            position: relative;
            cursor: pointer;
        }
        
        .streamlit-expanderContent::-webkit-scrollbar {
            width: 6px;
        }
        
        .streamlit-expanderContent::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 3px;
        }
        
        .streamlit-expanderContent::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #00d4ff, #5b21b6);
            border-radius: 3px;
        }
        
        .streamlit-expanderContent::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(45deg, #00d4ff, #06ffa5);
        }
        
        /* Futuristic SyntAI Spinner */
        .syntai-spinner-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.05), rgba(91, 33, 182, 0.05));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            margin: 1rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .syntai-spinner-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, #5b21b6, #00d4ff, transparent);
            animation: spinner-scan 2s linear infinite;
        }
        
        @keyframes spinner-scan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .syntai-spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(0, 212, 255, 0.2);
            border-top: 3px solid #00d4ff;
            border-radius: 50%;
            animation: syntai-spin 1s linear infinite;
            margin-bottom: 1rem;
            position: relative;
        }
        
        .syntai-spinner::after {
            content: '';
            position: absolute;
            top: 3px;
            left: 3px;
            right: 3px;
            bottom: 3px;
            border: 2px solid rgba(91, 33, 182, 0.2);
            border-bottom: 2px solid #5b21b6;
            border-radius: 50%;
            animation: syntai-spin-inner 1.5s linear infinite reverse;
        }
        
        @keyframes syntai-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes syntai-spin-inner {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .syntai-spinner-text {
            font-family: 'JetBrains Mono', monospace;
            color: #00d4ff;
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            animation: text-pulse 2s ease-in-out infinite;
        }
        
        @keyframes text-pulse {
            0%, 100% { opacity: 0.8; }
            50% { opacity: 1; text-shadow: 0 0 15px rgba(0, 212, 255, 0.8); }
        }
        
        .syntai-spinner-subtitle {
            font-family: 'Space Grotesk', sans-serif;
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 0.5rem;
            letter-spacing: 0.05em;
        }
        
        /* New UI Flow Styles */
        .hero-section {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.1), rgba(91, 33, 182, 0.1));
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            border: 1px solid rgba(0, 212, 255, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .hero-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, #5b21b6, #00d4ff, transparent);
            animation: hero-scan 3s linear infinite;
        }
        
        @keyframes hero-scan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .story-summary {
            background: rgba(15, 15, 35, 0.8);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #00d4ff;
            font-size: 1.1rem;
            line-height: 1.6;
            color: #e0e6ed;
        }
        
        .thumbnail-container {
            background: linear-gradient(145deg, #1e293b, #334155);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(100, 116, 139, 0.3);
        }
        
        .tab-container {
            background: rgba(15, 15, 35, 0.6);
            border-radius: 16px;
            padding: 0;
            margin: 1rem 0;
            border: 1px solid rgba(100, 116, 139, 0.2);
            overflow: hidden;
        }
        
        .tab-header {
            background: linear-gradient(145deg, #1e293b, #334155);
            padding: 1rem;
            border-bottom: 1px solid rgba(100, 116, 139, 0.2);
            color: #00d4ff;
            font-weight: 600;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .tab-content {
            padding: 1.5rem;
            background: rgba(15, 15, 35, 0.4);
        }
        
        .narration-box {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.05), rgba(91, 33, 182, 0.05));
            border-radius: 10px;
            padding: 1.2rem;
            margin: 1rem 0;
            border-left: 3px solid #00d4ff;
            font-style: italic;
            color: #e0e6ed;
        }
        
        .key-points {
            background: rgba(15, 15, 35, 0.6);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .key-points ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .key-points li {
            background: rgba(0, 212, 255, 0.1);
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            border-left: 3px solid #00d4ff;
            color: #e0e6ed;
        }
        
        .contribution-item {
            background: linear-gradient(145deg, rgba(91, 33, 182, 0.1), rgba(0, 212, 255, 0.05));
            border-radius: 10px;
            padding: 1rem;
            margin: 0.8rem 0;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .contribution-title {
            color: #00d4ff;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .contribution-explanation {
            color: #cbd5e1;
            line-height: 1.5;
        }
        
        .methodology-step {
            background: rgba(15, 15, 35, 0.7);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.8rem 0;
            border-left: 4px solid #5b21b6;
            position: relative;
        }
        
        .step-number {
            background: #5b21b6;
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 1rem;
            font-size: 0.9rem;
        }
        
        .results-insight {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.08), rgba(91, 33, 182, 0.08));
            border-radius: 10px;
            padding: 1rem;
            margin: 0.8rem 0;
            border: 1px solid rgba(0, 212, 255, 0.15);
        }
        
        .interpretation-box {
            background: rgba(15, 15, 35, 0.8);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(0, 212, 255, 0.3);
            color: #e0e6ed;
            line-height: 1.6;
        }
        
        .why-matters-box {
            background: linear-gradient(145deg, rgba(91, 33, 182, 0.2), rgba(0, 212, 255, 0.1));
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 2px solid rgba(0, 212, 255, 0.4);
            position: relative;
            overflow: hidden;
        }
        
        .why-matters-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, transparent);
            animation: matter-scan 2s linear infinite;
        }
        
        @keyframes matter-scan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .code-demo-section {
            background: linear-gradient(145deg, #1e293b, #334155);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 2rem 0;
            border: 1px solid rgba(100, 116, 139, 0.3);
        }
        
        .demo-button {
            background: linear-gradient(145deg, #00d4ff, #5b21b6);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .demo-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
        }
        .neural-search-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(0, 212, 255, 0.05), transparent);
            animation: matrix-scan 4s linear infinite;
            pointer-events: none;
        }
        
        @keyframes matrix-scan {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .search-mode-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: #00d4ff;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.8);
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }
        
        .search-mode-subtitle {
            color: #94a3b8;
            font-size: 1rem;
            font-weight: 300;
            letter-spacing: 0.05em;
            position: relative;
            z-index: 1;
        }
        
        .search-section {
            background: rgba(15, 15, 35, 0.8);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            position: relative;
        }
        
        .search-section-title {
            color: #00d4ff;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        
        .search-section-desc {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            font-style: italic;
        }
        
        .status-success {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(21, 128, 61, 0.2));
            border: 1px solid rgba(34, 197, 94, 0.4);
            border-radius: 8px;
            padding: 0.75rem;
            color: #4ade80;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            text-align: center;
            margin: 0.5rem 0;
            text-shadow: 0 0 10px rgba(74, 222, 128, 0.3);
        }
        
        .status-error {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(153, 27, 27, 0.2));
            border: 1px solid rgba(239, 68, 68, 0.4);
            border-radius: 8px;
            padding: 0.75rem;
            color: #f87171;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            text-align: center;
            margin: 0.5rem 0;
            text-shadow: 0 0 10px rgba(248, 113, 113, 0.3);
        }
        
        .status-warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(180, 83, 9, 0.2));
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-radius: 8px;
            padding: 0.75rem;
            color: #fbbf24;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            text-align: center;
            margin: 0.5rem 0;
            text-shadow: 0 0 10px rgba(251, 191, 36, 0.3);
        }
        
        /* Enhanced tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(15, 15, 35, 0.8);
            border-radius: 12px;
            padding: 0.5rem;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 8px;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(0, 212, 255, 0.1);
            border-color: rgba(0, 212, 255, 0.6);
            color: #00d4ff;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(91, 33, 182, 0.2)) !important;
            border-color: #00d4ff !important;
            color: #00d4ff !important;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_syntai_spinner(message: str, subtitle: str = ""):
    """Display futuristic SyntAI spinner with message"""
    subtitle_html = (
        f'<div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">{subtitle}</div>'
        if subtitle
        else ""
    )

    st.markdown(
        f"""
        <div style="text-align: center; margin: 3rem 0;">
            <div class="futuristic-spinner"></div>
            <span style="color: #00d4ff; font-family: 'Space Grotesk', sans-serif; margin-left: 1rem; font-weight: 600; font-size: 1.1rem;">
                🚀 {message}
            </span>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_agent_status(message: str, status_type: str = "processing"):
    """Display agent status with spinner"""
    status_icons = {"processing": "🤖", "success": "✅", "warning": "⚠️", "error": "❌"}

    icon = status_icons.get(status_type, "🤖")

    if status_type == "processing":
        st.markdown(
            f"""
        <div class="agent-status">
            <div class="agent-spinner"></div>
            <span>{icon} Agent Status: {message}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="agent-status">
            <span>{icon} Agent Status: {message}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )


def show_progress_bar(current: int, total: int, message: str):
    """Display a futuristic progress bar"""
    percentage = (current / total) * 100 if total > 0 else 0

    st.markdown(
        f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">{message}</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">{current}/{total} ({percentage:.1f}%)</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {percentage}%;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def clear_data_folder():
    """Clear all PDFs from data folder before new search"""
    data_folder = Path("data")
    if data_folder.exists():
        # Remove all PDF files
        for pdf_file in data_folder.glob("*.pdf"):
            try:
                pdf_file.unlink()
                print(f"Removed: {pdf_file.name}")
            except Exception as e:
                print(f"Error removing {pdf_file.name}: {e}")

        # Remove images folder
        images_folder = data_folder / "images"
        if images_folder.exists():
            try:
                shutil.rmtree(images_folder)
                print("Removed images folder")
            except Exception as e:
                print(f"Error removing images folder: {e}")


def display_search_interface():
    """
    Display the main SyntAI neural search interface.

    This function creates the primary user interface for the SyntAI research system,
    featuring:
    - Futuristic title and branding elements
    - Tabbed interface for Domain Search and URL Import modes
    - Advanced search parameters (domain selection, result limits, cache options)
    - Neural-themed submit buttons with dynamic text
    - Professional form validation and user feedback

    Returns:
        tuple: (search_btn, domain, max_results, clear_existing, active_search_method, paper_url)
            - search_btn (bool): Whether the search button was clicked
            - domain (str): Selected arXiv domain for search
            - max_results (int): Maximum number of papers to retrieve
            - clear_existing (bool): Whether to clear existing data
            - active_search_method (str): Either "Domain Search" or "URL Import"
            - paper_url (str): ArXiv URL for single paper import (URL Import mode only)
    """
    st.markdown('<h1 class="main-title">SyntAI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Your AI-Powered Research Ops Agent — Always On. Always Ahead.</p>',
        unsafe_allow_html=True,
    )

    # Search form
    with st.form("search_form"):
        st.markdown(
            """
                <div class="mission-config-header">
                    <div class="mission-title">MISSION CONFIGURATION</div>
                    <div class="mission-subtitle">Configure research parameters for neural deployment</div>
                </div>
                """,
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["Domain Specific", "Direct URL"])

        with tab1:
            col1, col2 = st.columns([3, 1])

            with col1:
                domain = st.selectbox(
                    "Select Domain",
                    options=["cs.AI", "cs.CV", "cs.LG", "cs.CL", "cs.NE", "cs.RO"],
                    format_func=lambda x: {
                        "cs.AI": "Artificial Intelligence",
                        "cs.CV": "Computer Vision",
                        "cs.LG": "Machine Learning",
                        "cs.CL": "Natural Language Processing",
                        "cs.NE": "Neural Computing",
                        "cs.RO": "Robotics",
                    }.get(x, x),
                    key="domain_selector",
                )

            with col2:
                max_results = st.number_input(
                    "Target Count",
                    min_value=1,
                    max_value=20,
                    value=5,
                    key="target_count",
                )

            paper_url = ""
            search_method = "Domain Search"

        with tab2:
            st.markdown(
                """
            <div class="search-section">
                <div class="search-section-title">Direct Target Acquisition</div>
                <div class="search-section-desc">Lock onto specific research paper via ArXiv coordinates</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            paper_url = st.text_input(
                "ArXiv Target Coordinates",
                placeholder="https://arxiv.org/abs/2301.07041",
                help="Enter ArXiv paper URL for direct SyntAI analysis",
                key="arxiv_target_url",
            )

            if paper_url:
                if not paper_url.strip():
                    st.markdown(
                        """
                    <div class="status-warning">
                        COORDINATES REQUIRED - Enter target URL
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                elif not "arxiv.org" in paper_url:
                    st.markdown(
                        """
                    <div class="status-error">
                        INVALID COORDINATES - ArXiv domain required
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                elif not ("abs/" in paper_url or "pdf/" in paper_url):
                    st.markdown(
                        """
                    <div class="status-error">
                        MALFORMED URL - Missing abs/ or pdf/ path
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    paper_id_match = re.search(r"(\d{4}\.\d{4,5})", paper_url)
                    if paper_id_match:
                        paper_id = paper_id_match.group(1)
                        st.markdown(
                            f"""
                        <div class="status-success">
                            TARGET LOCKED - Paper ID: {paper_id}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                        <div class="status-warning">
                            EXTRACTION FAILED - Could not parse paper ID
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            domain = "cs.AI"
            max_results = 1
            search_method = "URL Import"

        clear_existing = True

        st.markdown('<div class="mission-divider"></div>', unsafe_allow_html=True)

        if paper_url and paper_url.strip():
            active_search_method = "URL Import"
        else:
            active_search_method = "Domain Search"

        button_text = (
            "Deploy SyntAI"
            if active_search_method == "Domain Search"
            else "Lock & Analyze Target"
        )
        search_btn = st.form_submit_button(button_text, use_container_width=True)

    return (
        search_btn,
        domain,
        max_results,
        clear_existing,
        active_search_method,
        paper_url,
    )


def display_enhanced_paper_viewer(papers: list):
    """Display papers with revolutionary quantum-inspired design"""
    st.markdown('<div class="papers-grid">', unsafe_allow_html=True)

    for i, paper in enumerate(papers):
        authors_text = ", ".join(paper["authors"][:3]) + (
            "..." if len(paper["authors"]) > 3 else ""
        )
        arxiv_id = paper.get("arxiv_id", "N/A")
        published_date = (
            paper.get("published_date", "")[:10]
            if paper.get("published_date")
            else "N/A"
        )
        categories_text = ", ".join(paper.get("categories", ["Research"]))
        status_class = (
            "status-neural-ready"
            if paper.get("downloaded")
            else "status-neural-processing"
        )
        status_text = "READY" if paper.get("downloaded") else "PROCESSING"
        title = paper["title"]
        st.markdown(
            f"""
<div style="
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 24px rgba(0, 0, 0, 0.12)'" 
   onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 16px rgba(0, 0, 0, 0.08)'">
    
    <!-- Subtle accent line -->
    <div style="
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
    "></div>
    
    <!-- Header Section -->
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 20px;
        gap: 16px;
    ">
        <div style="
            flex: 1;
            color: #1e293b;
            font-size: 18px;
            font-weight: 700;
            line-height: 1.4;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            {title}
        </div>
        <div style="
            background: {'linear-gradient(135deg, #10b981, #059669)' if status_class == 'status-neural-ready' else 'linear-gradient(135deg, #f59e0b, #d97706)'};
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        ">
            {status_text}
        </div>
    </div>
    
    <!-- Metadata Grid -->
    <div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
    ">
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
        ">
            <div style="
                color: #64748b;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            ">Authors</div>
            <div style="
                color: #334155;
                font-size: 14px;
                font-weight: 500;
                line-height: 1.3;
            ">{authors_text}</div>
        </div>
        
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
        ">
            <div style="
                color: #64748b;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            ">ArXiv ID</div>
            <div style="
                color: #334155;
                font-size: 14px;
                font-weight: 500;
                font-family: 'JetBrains Mono', monospace;
            ">{arxiv_id}</div>
        </div>
        
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
        ">
            <div style="
                color: #64748b;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            ">Published</div>
            <div style="
                color: #334155;
                font-size: 14px;
                font-weight: 500;
            ">{published_date}</div>
        </div>
        
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
        ">
            <div style="
                color: #64748b;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            ">Categories</div>
            <div style="
                color: #334155;
                font-size: 14px;
                font-weight: 500;
            ">{categories_text}</div>
        </div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(
                "Preview",
                key=f"preview_{i}",
                use_container_width=True,
                help="SyntAI Document Interface",
            ):
                st.session_state[f"show_preview_{i}"] = True
                st.rerun()

        with col2:
            if st.button(
                "ARXIV PORTAL",
                key=f"quantum_arxiv_{i}",
                use_container_width=True,
                help="Open in ArXiv dimension",
            ):
                st.markdown(
                    f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <a href="{paper.get('arxiv_url', '#')}" target="_blank" 
                           style="color: #00d4ff; text-decoration: none; font-weight: 600;">
                            Jump To ARXIV →
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col3:
            if st.button(
                "QUANTUM STORE",
                key=f"quantum_download_{i}",
                use_container_width=True,
                help="Access quantum storage",
            ):
                if paper.get("pdf_filename"):
                    st.success(f"Quantum File: {paper['pdf_filename']}")
                else:
                    st.warning("Quantum file not materialized")

        with col4:
            if st.button(
                "NEURAL DEEP DIVE",
                key=f"quantum_analysis_{i}",
                use_container_width=True,
                help="Activate neural analysis protocols",
            ):
                st.session_state[f"show_analysis_{i}"] = True
                st.rerun()

        # Only show interactive controls for downloaded papers
        if not paper.get("downloaded"):
            # Quantum separator for non-downloaded papers
            st.markdown(
                """
                <div style="
                    height: 2px;
                    background: linear-gradient(90deg, 
                        transparent, 
                        rgba(6, 255, 165, 0.3), 
                        rgba(0, 212, 255, 0.3), 
                        transparent);
                    margin: 2rem 0;
                    border-radius: 1px;
                "></div>
                """,
                unsafe_allow_html=True,
            )
            continue

        # Quantum Document Preview Section
        if st.session_state.get(f"show_preview_{i}", False):
            # Create quantum preview container
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, 
                        rgba(6, 255, 165, 0.05) 0%, 
                        rgba(0, 212, 255, 0.05) 50%,
                        rgba(139, 69, 19, 0.02) 100%);
                    border: 1px solid rgba(6, 255, 165, 0.3);
                    border-radius: 25px;
                    padding: 2rem;
                    margin: 2rem 0;
                    backdrop-filter: blur(20px);
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        text-align: center;
                        font-size: 1.5rem;
                        font-weight: 700;
                        color: #06ffa5;
                        margin-bottom: 1.5rem;
                        font-family: 'Space Grotesk', sans-serif;
                        text-shadow: 0 0 20px rgba(6, 255, 165, 0.5);
                    ">
                    DOCUMENT INTERFACE
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                scraper = ArxivScraper("data")
                pdf_path = Path("data") / paper["pdf_filename"]

                if pdf_path.exists():
                    doc = fitz.open(str(pdf_path))
                    total_pages = len(doc)
                    doc.close()

                    # Initialize page state
                    current_page_key = f"current_page_{i}"
                    if current_page_key not in st.session_state:
                        st.session_state[current_page_key] = 1

                    # Quantum Jump Control
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        jump_page = st.selectbox(
                            "TELEPORT TO PAGE:",
                            range(1, total_pages + 1),
                            index=st.session_state[current_page_key] - 1,
                            key=f"jump_{i}",
                        )
                        if jump_page != st.session_state[current_page_key]:
                            st.session_state[current_page_key] = jump_page
                            st.rerun()

                    # Display current page in quantum container
                    try:
                        image_info = scraper.extract_single_page_image(
                            paper, st.session_state[current_page_key]
                        )

                        if image_info and Path(image_info["path"]).exists():
                            # Quantum document display
                            st.markdown(
                                f"""
                                <div style="
                                    background: linear-gradient(145deg, rgba(15, 15, 35, 0.8), rgba(30, 30, 60, 0.6));
                                    border: 1px solid rgba(6, 255, 165, 0.2);
                                    border-radius: 20px;
                                    padding: 2rem;
                                    margin: 2rem 0;
                                    backdrop-filter: blur(25px);
                                    box-shadow: 0 20px 60px rgba(6, 255, 165, 0.1);
                                ">
                                """,
                                unsafe_allow_html=True,
                            )

                            col_left, col_img, col_right = st.columns([0.5, 4, 0.5])
                            with col_img:
                                st.image(
                                    image_info["path"],
                                    caption=f"🧬 Quantum Scan #{st.session_state[current_page_key]} • Neural Pattern: {paper['title'][:40]}...",
                                    use_container_width=True,
                                )

                            st.markdown("</div>", unsafe_allow_html=True)

                            # Quantum Neural Analysis Button
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                if st.button(
                                    "SyntAI Analysis",
                                    key=f"quantum_neural_{i}_{st.session_state[current_page_key]}",
                                    use_container_width=True,
                                ):
                                    with st.spinner("SyntAI INITIALIZING..."):
                                        current_page = st.session_state[
                                            current_page_key
                                        ]
                                        image_info = scraper.extract_single_page_image(
                                            paper, current_page
                                        )

                                        if (
                                            image_info
                                            and Path(image_info["path"]).exists()
                                        ):
                                            vlm_analysis = code_generation_service.deep_dive_page_analysis(
                                                image_info["path"]
                                            )

                                            if vlm_analysis.get("success"):
                                                st.session_state[
                                                    f"vlm_analysis_{i}_{current_page}"
                                                ] = vlm_analysis
                                                st.session_state[
                                                    f"show_vlm_results_{i}_{current_page}"
                                                ] = True
                                                st.rerun()
                                            else:
                                                st.error(
                                                    f"⚠️ VLM analysis failed: {vlm_analysis.get('error', 'Unknown error')}"
                                                )
                                        else:
                                            st.error(
                                                "⚠️ Page image not found for VLM analysis"
                                            )

                            current_page = st.session_state[current_page_key]
                            if st.session_state.get(
                                f"show_vlm_results_{i}_{current_page}", False
                            ):
                                vlm_data = st.session_state.get(
                                    f"vlm_analysis_{i}_{current_page}"
                                )
                                if vlm_data:
                                    st.markdown(
                                        """
                                        <div style="
                                            background: linear-gradient(135deg, 
                                                rgba(6, 255, 165, 0.1) 0%, 
                                                rgba(0, 212, 255, 0.1) 50%,
                                                rgba(139, 69, 19, 0.05) 100%);
                                            border: 2px solid rgba(6, 255, 165, 0.4);
                                            border-radius: 20px;
                                            padding: 2rem;
                                            margin: 2rem 0;
                                            backdrop-filter: blur(25px);
                                        ">
                                            <h3 style="
                                                color: #06ffa5;
                                                text-align: center;
                                                font-family: 'Space Grotesk', sans-serif;
                                                text-shadow: 0 0 20px rgba(6, 255, 165, 0.5);
                                                margin-bottom: 2rem;
                                            ">SyntAI Analysis Results</h3>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    # Display VLM analysis content
                                    col1, col2 = st.columns([1, 1])

                                    with col1:
                                        st.markdown(
                                            f"""
                                            <div style="
                                            background: linear-gradient(135deg, 
                                                rgba(6, 255, 165, 0.1) 0%, 
                                                rgba(0, 212, 255, 0.1) 50%,
                                                rgba(139, 69, 19, 0.05) 100%);
                                            border: 2px solid rgba(6, 255, 165, 0.4);
                                            border-radius: 20px;
                                            padding: 2rem;
                                            margin: 2rem 0;
                                            backdrop-filter: blur(25px);
                                        ">
                                            Content Type: {vlm_data.get('content_type', 'Unknown')}
                                        </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                        # Quantum analysis display with enhanced styling
                                        st.markdown(
                                            f"""
                                            <div style="
                                                background: linear-gradient(135deg, rgba(6, 255, 165, 0.1), rgba(0, 212, 255, 0.1));
                                                border: 1px solid rgba(6, 255, 165, 0.3);
                                                border-radius: 12px;
                                                padding: 1rem;
                                                margin: 0.5rem 0;
                                                backdrop-filter: blur(10px);
                                            ">
                                                <span style="color: #06ffa5; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                                                    Has Diagram: {'Yes' if vlm_data.get('has_diagram') else 'No'}
                                                </span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                        if vlm_data.get("explanation"):
                                            st.markdown(
                                                """
                                                <div style="
                                                    background: rgba(0, 212, 255, 0.05);
                                                    border-left: 3px solid #00d4ff;
                                                    border-radius: 8px;
                                                    padding: 1rem;
                                                    margin: 1rem 0;
                                                ">
                                                    <h4 style="color: #00d4ff; margin: 0 0 0.5rem 0; font-family: 'JetBrains Mono', monospace;">Detailed Analysis</h4>
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )
                                            st.markdown(
                                                f"""
                                                <div style="
                                                    background: rgba(15, 15, 35, 0.6);
                                                    border: 1px solid rgba(0, 212, 255, 0.2);
                                                    border-radius: 8px;
                                                    padding: 1rem;
                                                    color: #e0e6ed;
                                                    line-height: 1.6;
                                                ">
                                                    {vlm_data["explanation"]}
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )

                                    with col2:
                                        if vlm_data.get("insights"):
                                            st.markdown(
                                                """
                                                <div style="
                                                    background: rgba(91, 33, 182, 0.05);
                                                    border-left: 3px solid #5b21b6;
                                                    border-radius: 8px;
                                                    padding: 1rem;
                                                    margin: 1rem 0;
                                                ">
                                                    <h4 style="color: #5b21b6; margin: 0 0 0.5rem 0; font-family: 'JetBrains Mono', monospace;">Key Insights</h4>
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )
                                            for i, insight in enumerate(
                                                vlm_data["insights"], 1
                                            ):
                                                st.markdown(
                                                    f"""
                                                    <div style="
                                                        background: rgba(91, 33, 182, 0.08);
                                                        border: 1px solid rgba(91, 33, 182, 0.2);
                                                        border-radius: 6px;
                                                        padding: 0.75rem;
                                                        margin: 0.5rem 0;
                                                        color: #e0e6ed;
                                                        transition: all 0.3s ease;
                                                    ">
                                                        <span style="color: #5b21b6; font-weight: 600;">#{i}</span> {insight}
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True,
                                                )

                                    # Additional sections with quantum styling
                                    if vlm_data.get("diagram_analysis"):
                                        st.markdown(
                                            """
                                            <div style="
                                                background: rgba(6, 255, 165, 0.05);
                                                border-left: 3px solid #06ffa5;
                                                border-radius: 8px;
                                                padding: 1rem;
                                                margin: 1rem 0;
                                            ">
                                                <h4 style="color: #06ffa5; margin: 0 0 0.5rem 0; font-family: 'JetBrains Mono', monospace;">Diagram Analysis</h4>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(
                                            f"""
                                            <div style="
                                                background: rgba(15, 15, 35, 0.6);
                                                border: 1px solid rgba(6, 255, 165, 0.2);
                                                border-radius: 8px;
                                                padding: 1rem;
                                                color: #e0e6ed;
                                                line-height: 1.6;
                                            ">
                                                {vlm_data["diagram_analysis"]}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                    if vlm_data.get("technical_elements"):
                                        st.markdown(
                                            """
                                            <div style="
                                                background: rgba(255, 193, 7, 0.05);
                                                border-left: 3px solid #ffc107;
                                                border-radius: 8px;
                                                padding: 1rem;
                                                margin: 1rem 0;
                                            ">
                                                <h4 style="color: #ffc107; margin: 0 0 0.5rem 0; font-family: 'JetBrains Mono', monospace;"> Technical Elements</h4>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(
                                            f"""
                                            <div style="
                                                background: rgba(15, 15, 35, 0.6);
                                                border: 1px solid rgba(255, 193, 7, 0.2);
                                                border-radius: 8px;
                                                padding: 1rem;
                                                color: #e0e6ed;
                                                line-height: 1.6;
                                            ">
                                                {vlm_data["technical_elements"]}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                    except Exception as e:
                        st.error(f"⚠️ Quantum interface error: {str(e)}")

                else:
                    st.warning("🔋 Quantum document not materialized in storage matrix")

            except Exception as e:
                st.error(f"⚠️ Neural interface connection failed: {str(e)}")

        # Neural Analysis Deep Dive
        if st.session_state.get(f"show_analysis_{i}", False):
            sections_data = extract_paper_sections_simple(paper, i)
            if sections_data:
                display_paper_deep_dive(sections_data, paper, i)

        # Quantum separator
        st.markdown(
            """
            <div style="
                height: 2px;
                background: linear-gradient(90deg, 
                    transparent, 
                    rgba(6, 255, 165, 0.5), 
                    rgba(0, 212, 255, 0.5), 
                    transparent);
                margin: 3rem 0;
                border-radius: 1px;
            "></div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def extract_paper_sections_simple(paper: dict, paper_index: int) -> dict:
    """Extract paper sections with AI-enhanced explanations (always AI-powered)"""

    if not paper.get("pdf_filename"):
        return {}

    pdf_path = Path("data") / paper["pdf_filename"]

    # Show spinner while processing
    spinner_placeholder = st.empty()
    with spinner_placeholder.container():
        show_syntai_spinner("SyntAI Processing", "Extracting paper sections...")

    try:
        # Extract PDF sections
        sections_result = pdf_text_extractor.extract_paper_sections(str(pdf_path))

        if not sections_result["success"]:
            spinner_placeholder.empty()
            st.error(
                f"Failed to extract sections: {sections_result.get('error', 'Unknown error')}"
            )
            return {}

        # Clear spinner on success
        spinner_placeholder.empty()

        return sections_result

    except Exception as e:
        spinner_placeholder.empty()
        st.error(f"Error extracting paper sections: {str(e)}")
        return {}


def create_simple_abstract_explanation(abstract_data: dict) -> dict:
    """Create simple explanation of abstract"""
    text = abstract_data.get("text", "")

    # Break down into simple components
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

    # Identify key components
    problem_sentences = []
    solution_sentences = []
    result_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(
            word in sentence_lower
            for word in ["problem", "challenge", "issue", "difficult"]
        ):
            problem_sentences.append(sentence)
        elif any(
            word in sentence_lower
            for word in ["propose", "introduce", "develop", "method", "approach"]
        ):
            solution_sentences.append(sentence)
        elif any(
            word in sentence_lower
            for word in ["result", "achieve", "performance", "improve", "better"]
        ):
            result_sentences.append(sentence)

    return {
        "original_text": text,
        "word_count": len(text.split()),
        "problem_context": problem_sentences[:2],  # First 2 problem statements
        "solution_approach": solution_sentences[:2],  # First 2 solution statements
        "key_results": result_sentences[:2],  # First 2 result statements
        "simple_summary": f"This paper tackles research challenges and proposes new methods, achieving improved results in {abstract_data.get('word_count', 0)} words.",
    }


def create_simple_contributions_explanation(contributions_data: dict) -> dict:
    """Create simple explanation of contributions"""
    items = contributions_data.get("items", [])
    text = contributions_data.get("text", "")

    # Categorize contributions
    categorized = {"technical": [], "methodological": [], "empirical": []}

    for item in items:
        item_lower = item.lower()
        if any(
            word in item_lower
            for word in ["algorithm", "architecture", "model", "system"]
        ):
            categorized["technical"].append(item)
        elif any(
            word in item_lower
            for word in ["method", "approach", "framework", "technique"]
        ):
            categorized["methodological"].append(item)
        elif any(
            word in item_lower
            for word in ["evaluation", "experiment", "analysis", "study"]
        ):
            categorized["empirical"].append(item)
        else:
            categorized["technical"].append(item)  # Default to technical

    return {
        "original_text": text,
        "all_items": items,
        "categorized": categorized,
        "total_count": len(items),
        "why_important": "These contributions advance the field by introducing novel techniques and demonstrating their effectiveness.",
    }


def create_simple_methodology_breakdown(methodology_data: dict) -> dict:
    """Create simple breakdown of methodology/architecture"""
    text = methodology_data.get("text", "")
    steps = methodology_data.get("steps", [])

    # Extract architectural components
    architecture_keywords = [
        "encoder",
        "decoder",
        "attention",
        "transformer",
        "cnn",
        "rnn",
        "lstm",
        "gru",
        "layer",
        "block",
        "module",
        "component",
        "network",
        "model",
        "architecture",
    ]

    components = []
    for sentence in text.split("."):
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in architecture_keywords):
            if len(sentence.strip()) > 30:  # Only substantial sentences
                components.append(sentence.strip())

    # Process steps into clear breakdown
    processed_steps = []
    for i, step in enumerate(steps, 1):
        # Clean up step text
        clean_step = step.replace(f"Step {i}:", "").replace(f"{i}.", "").strip()
        if len(clean_step) > 10:
            processed_steps.append(
                {
                    "step_number": i,
                    "description": clean_step,
                    "type": (
                        "process"
                        if any(
                            word in clean_step.lower()
                            for word in ["process", "compute", "calculate"]
                        )
                        else "architecture"
                    ),
                }
            )

    return {
        "original_text": text[:500],  # First 500 chars
        "architecture_components": components[:5],  # Top 5 components
        "process_steps": processed_steps[:6],  # Max 6 steps
        "step_count": len(processed_steps),
        "overview": f"The methodology involves {len(processed_steps)} main steps with {len(components)} key architectural components.",
    }


def create_simple_results_explanation(results_data: dict) -> dict:
    """Create simple explanation of results"""
    text = results_data.get("text", "")
    findings = results_data.get("findings", [])

    # Extract performance metrics and comparisons
    metrics = []
    comparisons = []

    for sentence in text.split("."):
        sentence_lower = sentence.lower()
        if any(
            word in sentence_lower
            for word in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "score",
                "performance",
            ]
        ):
            metrics.append(sentence.strip())
        elif any(
            word in sentence_lower
            for word in [
                "better",
                "outperform",
                "superior",
                "improve",
                "higher",
                "lower",
            ]
        ):
            comparisons.append(sentence.strip())

    # Process findings
    key_outcomes = []
    for finding in findings:
        if len(finding) > 20:  # Only substantial findings
            key_outcomes.append(finding)

    return {
        "original_text": text[:500],
        "performance_metrics": metrics[:3],
        "comparisons": comparisons[:3],
        "key_findings": key_outcomes[:4],
        "summary": f"Results show {len(metrics)} performance metrics with {len(comparisons)} comparative improvements.",
    }


def display_paper_deep_dive(sections_data: dict, paper: dict, paper_index: int):
    """Display comprehensive AI-powered deep dive analysis with enhanced UX"""

    # Enhanced Hero Section with gradient
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        ">
            <h1 style="color: white; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🧠 AI Deep Dive Analysis
            </h1>
            <p style="color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.2em;">
                Intelligent breakdown powered by SyntAI • {len(sections_data.get('processed_sections', {}))} sections analyzed
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    processed = sections_data.get("processed_sections", {})

    if not processed:
        st.error(
            "🚫 No sections could be analyzed. The PDF might have formatting issues."
        )
        return

    # Create interactive dashboard-style layout
    col1, col2 = st.columns([2, 1])

    with col2:
        # Analysis Overview Card
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(45deg, #1e3a8a, #3b82f6);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                color: white;
            ">
                <h3 style="margin: 0; color: #60a5fa;">📊 Analysis Overview</h3>
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span>Sections Found:</span>
                        <strong>{len(processed)}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span>AI Analysis:</span>
                        <strong style="color: #34d399;">Active</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span>Processing Time:</span>
                        <strong>< 30s</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Section Health Check
        st.markdown("### 🎯 Section Quality Check")
        for section_name in ["abstract", "contributions", "methodology", "results"]:
            if section_name in processed:
                quality_score = (
                    "🟢 Excellent"
                    if processed[section_name].get("narration")
                    or processed[section_name].get("explanations")
                    else "🟡 Basic"
                )
                st.markdown(f"**{section_name.title()}:** {quality_score}")
            else:
                st.markdown(f"**{section_name.title()}:** 🔴 Not Found")

    with col1:
        # Interactive Section Explorer
        st.markdown("### 🔍 Interactive Section Explorer")

        # Create tabs for better navigation
        available_sections = list(processed.keys())

        if len(available_sections) == 1:
            # If only one section, show it directly
            selected_section = available_sections[0]
            display_enhanced_section_content(
                selected_section, processed[selected_section]
            )
        else:
            # Multiple sections - use tabs
            tabs = st.tabs([f"📖 {section.title()}" for section in available_sections])

            for i, (tab, section) in enumerate(zip(tabs, available_sections)):
                with tab:
                    display_enhanced_section_content(section, processed[section])

    total_insights = 0
    key_takeaways = []

    for section_name, section_data in processed.items():
        if section_data.get("key_points"):
            total_insights += len(section_data["key_points"])
            key_takeaways.extend(
                section_data["key_points"][:2]
            )  # Top 2 from each section
        elif section_data.get("key_insights"):
            total_insights += len(section_data["key_insights"])
            key_takeaways.extend(section_data["key_insights"][:2])
        elif section_data.get("explanations"):
            total_insights += len(section_data["explanations"])
            key_takeaways.extend(section_data["explanations"][:2])

    if key_takeaways:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #10b981, #059669);
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
                color: white;
            ">
                <h3 style="margin: 0 0 15px 0; color: #d1fae5;">🧠 AI-Generated Key Takeaways</h3>
                <p style="margin: 0; font-size: 1.1em; line-height: 1.6;">
                    Based on {total_insights} insights extracted across {len(processed)} sections, 
                    here are the most important points to understand this research:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for i, takeaway in enumerate(key_takeaways[:5], 1):  # Show top 5
            st.markdown(
                f"""
                <div style="
                    background: rgba(16, 185, 129, 0.1);
                    border-left: 4px solid #10b981;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 0 8px 8px 0;
                ">
                    <strong>💡 Insight {i}:</strong> {takeaway[:200]}{'...' if len(takeaway) > 200 else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_enhanced_section_content(section_name: str, section_data: dict):
    """Display individual section with enhanced styling and better information architecture"""

    # Section header with icon and AI badge
    section_icons = {
        "abstract": "📄",
        "contributions": "🚀",
        "methodology": "🔬",
        "results": "📊",
    }

    icon = section_icons.get(section_name, "")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #1f2937, #374151);
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 5px solid #3b82f6;
        ">
            <h3 style="color: white; margin: 0; display: flex; align-items: center; gap: 10px;">
                {icon} {section_name.title()} Analysis
                <span style="
                    background: #3b82f6;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 15px;
                    font-size: 0.7em;
                    font-weight: normal;
                ">AI Enhanced</span>
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Content based on section type
    if section_name == "abstract":
        if section_data.get("narration"):
            st.markdown("**AI Understanding:**")
            st.markdown(
                f"""
                <div style="
                    background: rgba(59, 130, 246, 0.1);
                    border: 1px solid rgba(59, 130, 246, 0.2);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    font-size: 1.1em;
                    line-height: 1.6;
                ">
                    {section_data['narration']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if section_data.get("key_points"):
            st.markdown("**Key Points Extracted:**")
            for i, point in enumerate(section_data["key_points"], 1):
                st.markdown(
                    f"""
                    <div style="
                        background: white;
                        border: 1px solid #e5e7eb;
                        padding: 15px;
                        margin: 8px 0;
                        border-radius: 8px;
                        border-left: 4px solid #3b82f6;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <strong>Point {i}:</strong> {point}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    elif section_name == "contributions":
        if section_data.get("why_it_matters"):
            st.markdown("**Impact Analysis:**")
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #fbbf24, #f59e0b);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    font-size: 1.1em;
                    line-height: 1.6;
                ">
                    {section_data['why_it_matters']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if section_data.get("explanations"):
            st.markdown("**Detailed Contribution Analysis:**")
            for i, explanation in enumerate(section_data["explanations"], 1):
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(251, 191, 36, 0.1);
                        border: 1px solid rgba(251, 191, 36, 0.2);
                        padding: 18px;
                        margin: 12px 0;
                        border-radius: 10px;
                        border-left: 4px solid #fbbf24;
                    ">
                        <strong>Contribution {i}:</strong><br>
                        {explanation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    elif section_name == "methodology":
        if section_data.get("overview"):
            st.markdown("**Approach Overview:**")
            st.markdown(
                f"""
                <div style="
                    background: rgba(16, 185, 129, 0.1);
                    border: 1px solid rgba(16, 185, 129, 0.2);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    font-size: 1.1em;
                    line-height: 1.6;
                ">
                    {section_data['overview']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if section_data.get("walkthrough"):
            st.markdown("**Implementation Steps:**")
            for i, step in enumerate(section_data["walkthrough"], 1):
                st.markdown(
                    f"""
                    <div style="
                        background: white;
                        border: 1px solid #d1fae5;
                        padding: 18px;
                        margin: 12px 0;
                        border-radius: 10px;
                        border-left: 5px solid #10b981;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    ">
                        <div style="
                            background: #10b981;
                            color: white;
                            width: 30px;
                            height: 30px;
                            border-radius: 50%;
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            margin-right: 15px;
                            font-weight: bold;
                        ">{i}</div>
                        <strong>Step {i}:</strong> {step}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    elif section_name == "results":
        if section_data.get("interpretation"):
            st.markdown("**Results Interpretation:**")
            st.markdown(
                f"""
                <div style="
                    background: rgba(139, 92, 246, 0.1);
                    border: 1px solid rgba(139, 92, 246, 0.2);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    font-size: 1.1em;
                    line-height: 1.6;
                ">
                    {section_data['interpretation']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if section_data.get("key_insights"):
            st.markdown("**Key Insights:**")
            for i, insight in enumerate(section_data["key_insights"], 1):
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(139, 92, 246, 0.1);
                        border: 1px solid rgba(139, 92, 246, 0.2);
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 8px;
                        border-left: 4px solid #8b5cf6;
                    ">
                        <strong>Insight {i}:</strong> {insight}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def display_section_content(section_name: str, section_data: dict):
    """Display content for a specific section with AI-enhanced analysis"""

    st.markdown(
        f"""
    <div class="tab-container">
        <div class="tab-header">{section_name.title()} Analysis</div>
        <div class="tab-content">
    """,
        unsafe_allow_html=True,
    )

    # Always show AI-enhanced indicator
    st.markdown(
        """
        <div style="background: linear-gradient(45deg, #1e3a8a, #3b82f6); padding: 8px 12px; border-radius: 6px; margin-bottom: 15px;">
            <span style="color: #60a5fa;">AI-Enhanced Analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if section_name == "abstract":
        # AI-enhanced abstract display
        if section_data.get("narration"):
            st.markdown(
                f"""
            <div class="narration-box">
                <strong>AI Summary:</strong><br>
                {section_data['narration']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        if section_data.get("key_points"):
            st.markdown("**Key Points:**")
            for i, point in enumerate(section_data["key_points"], 1):
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(59, 130, 246, 0.1);
                        border: 1px solid rgba(59, 130, 246, 0.2);
                        padding: 12px;
                        margin: 8px 0;
                        border-radius: 6px;
                        border-left: 4px solid #3b82f6;
                        color: #1f2937;
                    ">
                        <strong style="color: #1e40af;">Point {i}:</strong> <span style="color: #374151;">{point}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Show original text in expander
        if section_data.get("original_text"):
            with st.expander("Original Abstract Text"):
                st.text_area(
                    "Original Text",
                    section_data["original_text"],
                    height=100,
                    disabled=True,
                )

    elif section_name == "contributions":
        # AI-enhanced contributions display
        if section_data.get("why_it_matters"):
            st.markdown(
                f"""
            <div class="why-matters-box">
                <h4>Why This Matters:</h4>
                {section_data['why_it_matters']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        if section_data.get("explanations"):
            st.markdown("**Contribution Analysis:**")
            for i, explanation in enumerate(section_data["explanations"], 1):
                st.markdown(
                    f"""
                <div style="
                    background: rgba(16, 185, 129, 0.1);
                    border: 1px solid rgba(16, 185, 129, 0.2);
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    border-left: 4px solid #10b981;
                    color: #1f2937;
                ">
                    <strong style="color: #065f46;">Contribution {i}:</strong> <span style="color: #374151;">{explanation}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Show original items in expander
        if section_data.get("original_items"):
            with st.expander("Original Contribution Items"):
                for item in section_data["original_items"]:
                    st.markdown(f"• {item}")

    elif section_name == "methodology":
        # AI-enhanced methodology display
        if section_data.get("overview"):
            st.markdown(
                f"""
            <div class="narration-box">
                <strong>Approach Overview:</strong><br>
                {section_data['overview']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        if section_data.get("walkthrough"):
            st.markdown("**Step-by-Step Walkthrough:**")
            for i, step in enumerate(section_data["walkthrough"], 1):
                st.markdown(
                    f"""
                <div style="
                    background: rgba(249, 115, 22, 0.1);
                    border: 1px solid rgba(249, 115, 22, 0.2);
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    border-left: 4px solid #f97316;
                    color: #1f2937;
                ">
                    <strong style="color: #c2410c;">Step {i}:</strong> <span style="color: #374151;">{step}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Show original steps in expander
        if section_data.get("original_steps"):
            with st.expander("Original Methodology Steps"):
                for step in section_data["original_steps"]:
                    st.markdown(f"• {step}")

    elif section_name == "results":
        # AI-enhanced results display
        if section_data.get("interpretation"):
            st.markdown(
                f"""
            <div class="interpretation-box">
                <h4>AI Interpretation:</h4>
                {section_data['interpretation']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        if section_data.get("key_insights"):
            st.markdown("**Key Insights:**")
            for insight in section_data["key_insights"]:
                st.markdown(
                    f"""
                <div class="results-insight">
                    {insight}
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Show original findings in expander
        if section_data.get("original_findings"):
            with st.expander("Original Research Findings"):
                for finding in section_data["original_findings"]:
                    st.markdown(f"• {finding}")

    # Close the tab container
    st.markdown("</div></div>", unsafe_allow_html=True)


def display_interactive_tabs(sections_data: dict, paper: dict, paper_index: int):
    """Display interactive explanation tabs"""
    available_tabs = []
    tab_data = {}

    if sections_data.get("sections", {}).get("abstract"):
        available_tabs.append("Abstract")
        tab_data["Abstract"] = sections_data["sections"]["abstract"]

    if sections_data.get("sections", {}).get("contributions"):
        available_tabs.append("Contributions")
        tab_data["Contributions"] = sections_data["sections"]["contributions"]

    if sections_data.get("sections", {}).get("methodology"):
        available_tabs.append("Methodology")
        tab_data["Methodology"] = sections_data["sections"]["methodology"]

    if sections_data.get("sections", {}).get("results"):
        available_tabs.append("Results")
        tab_data["Results"] = sections_data["sections"]["results"]

    if not available_tabs:
        st.warning("No sections could be extracted from this paper")
        return

    selected_tab = st.selectbox(
        "Select Section to Explore", available_tabs, key=f"tab_selector_{paper_index}"
    )

    if selected_tab in tab_data:
        display_tab_content(selected_tab, tab_data[selected_tab])


def display_tab_content(tab_name: str, tab_data: dict):
    """Display content for a specific tab"""

    st.markdown(
        f"""
    <div class="tab-container">
        <div class="tab-header">{tab_name}</div>
        <div class="tab-content">
    """,
        unsafe_allow_html=True,
    )

    if tab_name == "Abstract":
        # AI narration + text version
        if tab_data.get("narration"):
            st.markdown(
                f"""
            <div class="narration-box">
                <strong>AI Narration:</strong><br>
                {tab_data['narration']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        if tab_data.get("key_points"):
            st.markdown(
                """
            <div class="key-points">
                <h4>Key Points:</h4>
                <ul>
            """,
                unsafe_allow_html=True,
            )
            for point in tab_data["key_points"]:
                st.markdown(f"<li>{point}</li>", unsafe_allow_html=True)
            st.markdown("</ul></div>", unsafe_allow_html=True)

        if tab_data.get("original_text"):
            with st.expander("Original Abstract", expanded=False):
                st.text(tab_data["original_text"])

    elif tab_name == "Contributions":
        # Bullet points + AI "why it matters"
        if tab_data.get("why_it_matters"):
            st.markdown(
                f"""
            <div class="why-matters-box">
                <h4>Why It Matters:</h4>
                {tab_data['why_it_matters']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        if tab_data.get("explanations"):
            st.markdown("<h4>Contribution Breakdown:</h4>", unsafe_allow_html=True)
            for explanation in tab_data["explanations"]:
                if ":" in explanation:
                    title, desc = explanation.split(":", 1)
                    st.markdown(
                        f"""
                    <div class="contribution-item">
                        <div class="contribution-title">{title.strip()}</div>
                        <div class="contribution-explanation">{desc.strip()}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="contribution-item">
                        <div class="contribution-explanation">{explanation}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    elif tab_name == "Methodology":
        # Step-by-step diagram + voice walkthrough
        if tab_data.get("overview"):
            st.markdown(
                f"""
            <div class="narration-box">
                <strong>Approach Overview:</strong><br>
                {tab_data['overview']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        if tab_data.get("walkthrough"):
            st.markdown("<h4>Step-by-Step Walkthrough:</h4>", unsafe_allow_html=True)
            for i, step in enumerate(tab_data["walkthrough"], 1):
                step_text = step.replace(f"Step {i}:", "").strip()
                st.markdown(
                    f"""
                <div class="methodology-step">
                    <span class="step-number">{i}</span>
                    {step_text}
                </div>
                """,
                    unsafe_allow_html=True,
                )

    elif tab_name == "Results":
        # Graphs extracted + AI interpretation
        if tab_data.get("interpretation"):
            st.markdown(
                f"""
            <div class="interpretation-box">
                <h4>Results Interpretation:</h4>
                {tab_data['interpretation']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        if tab_data.get("key_insights"):
            st.markdown("<h4>Key Insights:</h4>", unsafe_allow_html=True)
            for insight in tab_data["key_insights"]:
                st.markdown(
                    f"""
                <div class="results-insight">
                    • {insight}
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div></div>", unsafe_allow_html=True)


# Alias for backward compatibility
display_paper_viewer = display_enhanced_paper_viewer


def process_paper_from_url(paper_url: str, progress_callback) -> list:
    """
    Process a single research paper from an ArXiv URL.

    This function extracts paper information from an ArXiv URL and processes it
    through the SyntAI pipeline. It handles:
    - Paper ID extraction from various ArXiv URL formats
    - Metadata retrieval and validation
    - PDF download and processing
    - Integration with the enhanced workflow system

    Args:
        paper_url (str): The ArXiv URL (e.g., https://arxiv.org/abs/2401.12345)
        progress_callback (callable): Function to report processing progress
                                    Called with (step, total_steps, message)

    Returns:
        list: List containing the processed paper dictionary with metadata,
              or empty list if processing fails

    Raises:
        Exception: If URL format is invalid or paper cannot be processed
    """
    try:
        paper_id_match = re.search(r"(\d{4}\.\d{4,5})", paper_url)
        if not paper_id_match:
            st.error("Could not extract paper ID from URL")
            return []

        paper_id = paper_id_match.group(1)
        progress_callback(1, 5, f"Extracting paper ID: {paper_id}")

        scraper = ArxivScraper("data")
        progress_callback(2, 5, "Fetching paper metadata...")
        paper_info = {
            "arxiv_id": paper_id,
            "title": "Loading...",
            "authors": ["Loading..."],
            "published_date": "2024-01-01",
            "abstract": "Loading...",
            "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
        }

        try:
            # Query ArXiv API
            api_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            progress_callback(3, 5, "Fetching from ArXiv API...")

            with urllib.request.urlopen(api_url) as response:
                feed_data = response.read()

            feed = feedparser.parse(feed_data)

            if feed.entries:
                entry = feed.entries[0]
                paper_info.update(
                    {
                        "title": entry.title,
                        "authors": [author.name for author in entry.authors],
                        "published_date": entry.published,
                        "abstract": entry.summary,
                    }
                )
                progress_callback(3, 5, f"Retrieved: {entry.title[:50]}...")

        except Exception as e:
            print(f"Warning: Could not fetch metadata: {e}")
            # Continue with basic info

        # Download PDF
        progress_callback(4, 5, "Downloading PDF...")

        try:
            pdf_filename = f"{paper_id}.pdf"
            pdf_path = Path("data") / pdf_filename

            # Download if not exists
            if not pdf_path.exists():
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                urllib.request.urlretrieve(pdf_url, pdf_path)

            paper_info.update(
                {
                    "pdf_filename": pdf_filename,
                    "downloaded": True,
                    "pdf_path": str(pdf_path),
                }
            )

            progress_callback(5, 5, "Paper ready for analysis!")

        except Exception as e:
            st.error(f"Failed to download PDF: {e}")
            paper_info["downloaded"] = False

        return [paper_info]

    except Exception as e:
        st.error(f"Error processing paper URL: {e}")
        return []


def main():
    """Main application with comprehensive agent behavior"""
    load_css()
    if "papers" not in st.session_state:
        st.session_state.papers = []
    if "searching" not in st.session_state:
        st.session_state.searching = False

    # Search interface
    search_btn, domain, max_results, clear_existing, search_method, paper_url = (
        display_search_interface()
    )

    # Handle search with real-time progress
    if search_btn and not st.session_state.searching:
        if search_method == "URL Import":
            if not paper_url:
                st.error("Please enter a valid ArXiv URL")
                return
            if not (
                "arxiv.org" in paper_url
                and ("abs/" in paper_url or "pdf/" in paper_url)
            ):
                st.error("Please enter a valid ArXiv URL")
                return

        st.session_state.searching = True
        st.session_state.papers = []

        # Progress containers
        progress_container = st.empty()
        sub_progress_container = st.empty()

        def progress_callback(step, total_steps, message, sub_progress=0, sub_total=0):
            """Real-time progress callback"""
            with progress_container:
                show_progress_bar(step, total_steps, f"Step {step}/{total_steps}")

            if sub_progress > 0 and sub_total > 0:
                with sub_progress_container:
                    show_progress_bar(sub_progress, sub_total, message)
            else:
                sub_progress_container.empty()

        try:
            # Clear cache if requested
            if clear_existing:
                progress_callback(1, 6, "Clearing cache...")
                clear_data_folder()
                time.sleep(0.5)

            if search_method == "Domain Search":
                # Execute SyntAI mission for domain search
                papers = enhanced_workflow.run(
                    domain=domain,
                    max_results=max_results,
                    deep_research=True,
                    progress_callback=progress_callback,
                )
            else:
                # Process single paper from URL
                papers = process_paper_from_url(paper_url, progress_callback)

            if papers:
                st.session_state.papers = papers

                # Success metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("SyntAI Papers Acquired", len(papers))
                with col2:
                    downloaded = sum(1 for p in papers if p.get("downloaded"))
                    st.metric("Processed", downloaded)
                with col3:
                    if search_method == "Domain Search":
                        st.metric("Domain", domain)
                    else:
                        st.metric("Source", "ArXiv URL")

            else:
                if search_method == "Domain Search":
                    st.warning(
                        "No research targets identified. Adjust parameters and try again."
                    )
                else:
                    st.warning("Could not process the paper from the provided URL.")

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            st.session_state.searching = False
            time.sleep(1)
            progress_container.empty()
            sub_progress_container.empty()
            time.sleep(0.5)
            st.rerun()

    # Results dashboard
    if st.session_state.papers and not st.session_state.searching:
        st.markdown("---")
        display_paper_viewer(st.session_state.papers)

    elif st.session_state.searching:
        # Enhanced loading state using SyntAI futuristic spinner
        st.markdown("---")
        st.markdown("## SyntAI Processing...")
        st.markdown("*Acquiring research intelligence from ArXiv database*")

        # SyntAI Futuristic Spinner
        st.markdown(
            """
            <div style="text-align: center; margin: 3rem 0;">
                <div class="futuristic-spinner"></div>
                <span style="color: #00d4ff; font-family: 'Space Grotesk', sans-serif; margin-left: 1rem; font-weight: 600; font-size: 1.1rem;">
                    SyntAI is analyzing your request…
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
