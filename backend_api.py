"""
SyntAI Backend API Server
FastAPI server that bridges the new HTML/Tailwind frontend with existing backend services
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import existing backend services
try:
    from agents.research_agent import ArxivScraper
    from service.llm import code_generation_service, paper_summarization_service
    from service.pdf_parser import pdf_text_extractor
    from service.rag_service import get_rag_service
except ImportError as e:
    print(f"Warning: Could not import backend services: {e}")
    print("Make sure you're running from the project root directory")
    ArxivScraper = None
    code_generation_service = None
    paper_summarization_service = None
    pdf_text_extractor = None
    get_rag_service = None

app = FastAPI(title="SyntAI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request validation
class SearchRequest(BaseModel):
    domain: str
    max_results: int = 10
    clear_existing: bool = False


class ImportRequest(BaseModel):
    url: str
    clear_existing: bool = False


class AnalysisRequest(BaseModel):
    paper_index: int
    page_number: int


class ChatRequest(BaseModel):
    message: str


# Global state
current_papers: List[Dict[str, Any]] = []
scraper_instance = None

# Ensure data directory exists
Path("data").mkdir(exist_ok=True)
Path("data/images").mkdir(exist_ok=True)


def get_scraper():
    """Get or create ArxivScraper instance"""
    global scraper_instance
    if scraper_instance is None:
        if ArxivScraper is None:
            raise HTTPException(status_code=503, detail="ArxivScraper not available")
        scraper_instance = ArxivScraper("data")
    return scraper_instance


def clear_data_folder():
    """Clear all PDFs from data folder"""
    data_folder = Path("data")
    if data_folder.exists():
        for file in data_folder.glob("*.pdf"):
            try:
                file.unlink()
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")


@app.get("/api/papers")
async def get_papers():
    """Get current papers list"""
    return {"success": True, "papers": current_papers}


@app.post("/api/search")
async def search_papers(request: SearchRequest):
    """Stream search results for domain-based paper search"""
    global current_papers

    if request.clear_existing:
        clear_data_folder()
        current_papers = []

    async def generate_search_stream():
        try:
            scraper = get_scraper()

            # Send initial progress
            yield f"data: {json.dumps({'type': 'progress', 'current': 0, 'total': request.max_results, 'message': 'Searching ArXiv database...'})}\n\n"

            # Real search using ArxivScraper
            papers = scraper.search_papers(
                domain=request.domain,
                max_results=request.max_results,
                sort_by="submittedDate",
                sort_order="descending",
            )

            if not papers:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No papers found'})}\n\n"
                return

            # Send progress update for download phase
            yield f"data: {json.dumps({'type': 'progress', 'current': 0, 'total': len(papers), 'message': 'Starting PDF downloads...'})}\n\n"

            # Download PDFs
            for i, paper_data in enumerate(papers):
                # Send progress update
                paper_title = paper_data.get("title", "Unknown")[:50]
                progress_message = (
                    f"Downloading paper {i + 1}/{len(papers)}: {paper_title}..."
                )
                yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': len(papers), 'message': progress_message})}\n\n"

                try:
                    # Download individual PDF
                    scraper.download_pdfs([paper_data])

                    # Add to current papers
                    current_papers.append(paper_data)

                    # Process for RAG asynchronously
                    if get_rag_service:
                        rag_service = get_rag_service()
                        rag_service.process_paper_async(paper_data)

                    # Send paper update
                    yield f"data: {json.dumps({'type': 'paper', 'paper': paper_data})}\n\n"

                except Exception as e:
                    print(f"Error downloading paper {i+1}: {e}")
                    paper_data["downloaded"] = False
                    current_papers.append(paper_data)
                    
                    # Still try to process for RAG if PDF exists
                    if get_rag_service:
                        rag_service = get_rag_service()
                        rag_service.process_paper_async(paper_data)
                    
                    yield f"data: {json.dumps({'type': 'paper', 'paper': paper_data})}\n\n"

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)

            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'papers': current_papers})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate_search_stream(), media_type="text/plain")


@app.post("/api/import")
async def import_paper(request: ImportRequest):
    """Import a single paper from ArXiv URL"""
    global current_papers

    try:
        if request.clear_existing:
            clear_data_folder()
            current_papers = []

        scraper = get_scraper()

        # Use real ArxivScraper to download from URL
        # Check if download_from_url method exists, if not use alternative approach
        if hasattr(scraper, "download_from_url"):
            paper_data = scraper.download_from_url(request.url)
        else:
            # Alternative: extract arxiv ID and use search + download
            import re

            arxiv_id_match = re.search(
                r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", request.url
            )
            if arxiv_id_match:
                arxiv_id = arxiv_id_match.group(1)
                # Create paper data manually and download
                paper_data = {
                    "arxiv_id": arxiv_id,
                    "title": f"Imported Paper {arxiv_id}",
                    "authors": [],
                    "abstract": "",
                    "methodology": [],
                    "categories": [],
                    "published_date": datetime.now().isoformat(),
                    "updated_date": datetime.now().isoformat(),
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "arxiv_url": request.url,
                    "downloaded": False,
                    "pdf_filename": None,
                }
                # Download the PDF
                scraper.download_pdfs([paper_data])
            else:
                raise HTTPException(status_code=400, detail="Invalid ArXiv URL format")

        if paper_data:
            current_papers.append(paper_data)
            
            # Process for RAG asynchronously
            if get_rag_service:
                rag_service = get_rag_service()
                rag_service.process_paper_async(paper_data)
            
            return {"success": True, "papers": current_papers}
        else:
            raise HTTPException(status_code=400, detail="Failed to import paper")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/{paper_index}/info")
async def get_paper_info(paper_index: int):
    """Get paper information including total pages"""
    if paper_index >= len(current_papers):
        raise HTTPException(status_code=404, detail="Paper not found")

    paper = current_papers[paper_index]

    try:
        scraper = get_scraper()

        # Get PDF info using real ArxivScraper logic
        total_pages = scraper.get_pdf_page_count(paper)

        return {
            "success": True,
            "total_pages": total_pages,
            "title": paper.get("title", ""),
            "filename": paper.get("pdf_filename", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/{paper_index}/page/{page_number}")
async def get_paper_page(paper_index: int, page_number: int):
    """Get a specific page image from a paper"""
    if paper_index >= len(current_papers):
        raise HTTPException(status_code=404, detail="Paper not found")

    paper = current_papers[paper_index]

    try:
        scraper = get_scraper()

        # Extract page image using real ArxivScraper
        image_info = scraper.extract_single_page_image(paper, page_number)

        if image_info and Path(image_info["path"]).exists():
            # Return the actual image file directly
            return FileResponse(image_info["path"], media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="Page image not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper/{paper_index}/analyze/{page_number}")
async def analyze_paper_page(paper_index: int, page_number: int):
    """Analyze a specific page using VLM deep dive analysis"""
    if paper_index >= len(current_papers):
        raise HTTPException(status_code=404, detail="Paper not found")

    paper = current_papers[paper_index]

    try:
        scraper = get_scraper()

        # Get page image first
        image_info = scraper.extract_single_page_image(paper, page_number)

        if not image_info or not Path(image_info["path"]).exists():
            raise HTTPException(status_code=404, detail="Page image not found")

        # Use VLM for deep dive analysis using the code_generation_service
        if code_generation_service and code_generation_service.is_available():
            analysis_result = code_generation_service.deep_dive_page_analysis(
                image_info["path"]
            )

            if analysis_result.get("success"):
                return {
                    "success": True,
                    "analysis": {
                        "content_type": analysis_result.get("content_type", "Unknown"),
                        "has_diagram": analysis_result.get("has_diagram", False),
                        "explanation": analysis_result.get("explanation", ""),
                        "insights": analysis_result.get("insights", []),
                        "diagram_analysis": analysis_result.get("diagram_analysis", ""),
                        "technical_elements": analysis_result.get(
                            "technical_elements", ""
                        ),
                        "page_number": page_number,
                        "paper_title": paper.get("title", ""),
                    },
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=analysis_result.get("error", "Analysis failed"),
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="VLM service not available. Please check your GROQ_API_KEY configuration.",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_all_data():
    """Clear all papers and data"""
    global current_papers

    try:
        clear_data_folder()
        current_papers = []

        # Clear RAG data as well
        if get_rag_service:
            rag_service = get_rag_service()
            rag_service.clear_all_data()

        return {"success": True, "message": "All data cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_with_papers(request: ChatRequest):
    """Chat with papers using RAG"""
    try:
        if not get_rag_service:
            raise HTTPException(status_code=503, detail="RAG service not available")
        
        rag_service = get_rag_service()
        
        # Get chat response
        response = rag_service.chat(request.message)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/status")
async def get_rag_status():
    """Get RAG service status"""
    try:
        if not get_rag_service:
            return {"success": False, "message": "RAG service not available"}
        
        rag_service = get_rag_service()
        status = rag_service.get_status()
        
        return {"success": True, "status": status}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/process")
async def process_papers_for_rag():
    """Process current papers for RAG (manual trigger)"""
    try:
        if not get_rag_service:
            raise HTTPException(status_code=503, detail="RAG service not available")
        
        rag_service = get_rag_service()
        
        # Process all current papers
        processed_count = 0
        for paper in current_papers:
            rag_service.process_paper_async(paper)
            processed_count += 1
        
        return {
            "success": True, 
            "message": f"Started processing {processed_count} papers for RAG",
            "processed_count": processed_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files AFTER all API routes are defined
if Path("data/images").exists():
    app.mount("/images", StaticFiles(directory="data/images"), name="images")
app.mount("/static", StaticFiles(directory="data"), name="static")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting SyntAI Backend Server...")
    print("📊 Frontend available at: http://localhost:8000")
    print("🔧 API documentation at: http://localhost:8000/docs")

    uvicorn.run(
        "backend_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
