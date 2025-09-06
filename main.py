"""
SyntAI Backend API Server
FastAPI server that bridges the new HTML/Tailwind frontend with existing backend services
"""

import os
import sys
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import math
from dotenv import load_dotenv

load_dotenv()

# Global temporary directory for analysis results
TEMP_ANALYSIS_DIR = None


def get_temp_analysis_dir():
    """Get or create a temporary directory for analysis results"""
    global TEMP_ANALYSIS_DIR
    if TEMP_ANALYSIS_DIR is None or not Path(TEMP_ANALYSIS_DIR).exists():
        TEMP_ANALYSIS_DIR = tempfile.mkdtemp(prefix="syntai_analysis_")
    return TEMP_ANALYSIS_DIR

def cleanup_temp_analysis_dir():
    """Clean up the temporary analysis directory"""
    global TEMP_ANALYSIS_DIR
    if TEMP_ANALYSIS_DIR and Path(TEMP_ANALYSIS_DIR).exists():
        shutil.rmtree(TEMP_ANALYSIS_DIR, ignore_errors=True)
        TEMP_ANALYSIS_DIR = None

# Utility function to clean JSON data
def clean_json_data(obj):
    """Clean data structure for JSON serialization by handling NaN values"""
    if isinstance(obj, dict):
        return {k: clean_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_data(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'isna') and obj.isna():  # pandas Series/DataFrame NaN
        return None
    elif str(obj) == 'nan':  # string representation of NaN
        return None
    else:
        return obj

# Create separate thread pool for RAG processing
rag_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="RAG-Worker")

# Import existing backend services
try:
    print("📥 Loading ArxivScraper...")
    from agents.research_agent import ArxivScraper
    print("✅ ArxivScraper loaded")
    
    print("📥 Loading LLM services...")
    from service.llm import code_generation_service, paper_summarization_service
    print("✅ LLM services loaded")
    
    print("📥 Loading PDF parser...")
    from service.pdf_parser import pdf_text_extractor
    print("✅ PDF parser loaded")
    
    print("📥 Loading RAG service...")
    try:
        from service.rag_service import get_rag_service
        print("✅ RAG service loaded (dependencies will be checked on first use)")
    except Exception as e:
        print(f"⚠️ RAG service failed to load: {e}")
        print("Chat functionality will be unavailable")
        get_rag_service = None

    print("📥 Loading comparative analysis service...")
    from service.comparative_analysis_service import ComparativeAnalysisService, run_comparative_analysis
    print("✅ Comparative analysis service loaded")
    
except ImportError as e:
    print(f"Warning: Could not import backend services: {e}")
    print("Make sure you're running from the project root directory")
    ArxivScraper = None
    code_generation_service = None
    paper_summarization_service = None
    pdf_text_extractor = None
    get_rag_service = None
    ComparativeAnalysisService = None
    run_comparative_analysis = None

app = FastAPI(title="SyntAI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Print startup information and clean up any previous temp files"""
    cleanup_temp_analysis_dir() 
    print("🚀 SyntAI Backend API Server Started")
    print("⚡ Performance optimized with lazy loading:")
    print("  - SentenceTransformer models load only when needed")
    print("  - RAG service initializes on first use")
    print("  - Faster startup times!")
    print("📡 API ready at http://localhost:8000")


# Pydantic models for request validation
class SearchRequest(BaseModel):
    domain: str
    max_results: int = 10
    clear_existing: bool = False


class KeywordSearchRequest(BaseModel):
    keywords: str
    max_results: int = 10
    clear_existing: bool = False
    time_range_days: Optional[int] = 180  # 6 months default, None for custom dates
    sort_by: str = "relevance"  # relevance, submittedDate, lastUpdatedDate
    start_date: Optional[str] = None  # YYYY-MM-DD format
    end_date: Optional[str] = None  # YYYY-MM-DD format


class ImportRequest(BaseModel):
    url: str
    clear_existing: bool = False


class AnalysisRequest(BaseModel):
    paper_index: int
    page_number: int


class ChatRequest(BaseModel):
    message: str


class ComparativeAnalysisRequest(BaseModel):
    paper_indices: List[int]  # Indices of papers from current_papers to compare
    output_dir: Optional[str] = "analysis_output"


class PdfUploadRequest(BaseModel):
    pdf_paths: List[str]  # List of PDF file paths for comparison
    output_dir: Optional[str] = "analysis_output"


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


def schedule_rag_processing(paper_data: Dict[str, Any]) -> None:
    """Schedule RAG processing in a separate thread pool (non-blocking)"""
    def process_for_rag():
        try:
            if get_rag_service:
                rag_service = get_rag_service()
                paper_title = paper_data.get('title', 'Unknown')
                print(f"🔄 [RAG-Thread] Starting processing for: {paper_title[:50]}...")
                rag_service.process_paper_async(paper_data)
        except Exception as e:
            paper_title = paper_data.get('title', 'Unknown')
            print(f"❌ [RAG-Thread] Failed to process {paper_title[:50]}: {e}")
    
    # Submit to separate thread pool
    rag_thread_pool.submit(process_for_rag)


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


@app.get("/health")
async def health_check():
    """Health check endpoint for server readiness"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/papers")
async def get_papers():
    """Get current papers list"""
    return {"success": True, "papers": current_papers}


@app.post("/api/search/keywords")
async def search_papers_by_keywords(request: KeywordSearchRequest):
    """Stream search results for keyword-based paper search"""
    global current_papers

    if request.clear_existing:
        clear_data_folder()
        current_papers = []

    async def generate_keyword_search_stream():
        try:
            scraper = get_scraper()

            # Send initial progress
            yield f"data: {json.dumps({'type': 'progress', 'current': 0, 'total': request.max_results, 'message': f'Searching ArXiv for: {request.keywords}...'})}\n\n"

            # Enhanced keyword search using ArxivScraper
            papers = scraper.search_papers_by_keywords(
                keywords=request.keywords,
                max_results=request.max_results,
                sort_by=request.sort_by,
                sort_order="descending",
                time_range_days=request.time_range_days,
                start_date=request.start_date,
                end_date=request.end_date
            )

            if not papers:
                yield f"data: {json.dumps({'type': 'error', 'message': f'No papers found for keywords: {request.keywords} in the specified date range. Try adjusting your search terms or date range.'})}\n\n"
                return

            # Send progress update for download phase
            yield f"data: {json.dumps({'type': 'progress', 'current': 0, 'total': len(papers), 'message': 'Starting parallel PDF downloads...'})}\n\n"

            # Download PDFs in parallel for better performance
            if hasattr(scraper, 'download_pdfs_parallel'):
                print(f"🚀 Starting parallel download of {len(papers)} papers...")
                
                # Since we can't yield from callback in async generator, 
                # we'll process papers one by one after parallel download
                scraper.download_pdfs_parallel(papers, max_workers=3)

                # Add papers and send updates as they're processed
                for i, paper_data in enumerate(papers):
                    current_papers.append(paper_data)
                    
                    # Send progress update
                    progress_message = f"Processed {i + 1}/{len(papers)} papers..."
                    yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': len(papers), 'message': progress_message})}\n\n"
                    
                    # Send paper update
                    yield f"data: {json.dumps({'type': 'paper', 'paper': paper_data})}\n\n"
                    
                    # Small delay to show progress
                    await asyncio.sleep(0.1)

                # Send download complete event
                yield f"data: {json.dumps({'type': 'download_complete', 'message': f'Successfully downloaded {len(papers)} papers. RAG processing will continue in background.'})}\n\n"
            else:
                # Fallback to sequential download
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

                        # Send paper update
                        yield f"data: {json.dumps({'type': 'paper', 'paper': paper_data})}\n\n"

                    except Exception as e:
                        print(f"Error downloading paper {i+1}: {e}")
                        paper_data["downloaded"] = False
                        current_papers.append(paper_data)

                        # Send paper update even if download failed
                        yield f"data: {json.dumps({'type': 'paper', 'paper': paper_data})}\n\n"

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)

            # Send download completion event (downloads done, RAG processing continues in background)
            yield f"data: {json.dumps({'type': 'download_complete', 'total_papers': len(current_papers)})}\n\n"

            # Start RAG processing in background (non-blocking)
            if get_rag_service:
                rag_service = get_rag_service()
                for paper_data in current_papers:
                    if not paper_data.get('rag_processed', False):
                        schedule_rag_processing(paper_data)

            # Send completion (papers are ready to display)
            yield f"data: {json.dumps({'type': 'complete', 'papers': current_papers})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate_keyword_search_stream(), media_type="text/plain")


@app.post("/api/search")
async def search_papers(request: SearchRequest):
    """Stream search results for domain-based paper search (legacy)"""
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
            yield f"data: {json.dumps({'type': 'progress', 'current': 0, 'total': len(papers), 'message': 'Starting parallel PDF downloads...'})}\n\n"

            # Download PDFs in parallel for better performance
            if hasattr(scraper, 'download_pdfs_parallel'):
                print(f"🚀 Starting parallel download of {len(papers)} papers...")
                scraper.download_pdfs_parallel(papers, max_workers=3)

                # Add all papers to current papers after parallel download
                for paper_data in papers:
                    current_papers.append(paper_data)
                    yield f"data: {json.dumps({'type': 'paper', 'paper': paper_data})}\n\n"

                # Send download complete event
                yield f"data: {json.dumps({'type': 'download_complete', 'message': f'Successfully downloaded {len(papers)} papers. RAG processing will continue in background.'})}\n\n"
            else:
                # Fallback to sequential download
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

                        # Send paper update
                        yield f"data: {json.dumps({'type': 'paper', 'paper': paper_data})}\n\n"

                    except Exception as e:
                        print(f"Error downloading paper {i+1}: {e}")
                        paper_data["downloaded"] = False
                        current_papers.append(paper_data)

                        # Send paper update even if download failed
                        yield f"data: {json.dumps({'type': 'paper', 'paper': paper_data})}\n\n"

                # Send download complete event
                yield f"data: {json.dumps({'type': 'download_complete', 'message': f'Successfully downloaded {len(papers)} papers. RAG processing will continue in background.'})}\n\n"

            # Background RAG processing for all papers
            def schedule_rag_processing():
                rag_service = get_rag_service()
                for paper_data in papers:
                    try:
                        rag_service.process_pdf(paper_data)
                    except Exception as e:
                        print(f"Error processing {paper_data.get('title', 'Unknown')}: {e}")

            # Start RAG processing in background thread
            import threading
            rag_thread = threading.Thread(target=schedule_rag_processing)
            rag_thread.daemon = True
            rag_thread.start()

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)

            # Start background RAG processing for all papers
            # This happens asynchronously so it doesn't block the UI
            async def process_rag_for_papers():
                try:
                    print(f"🧠 Starting background RAG processing for {len(current_papers)} papers...")
                    
                    # Initialize RAG service if needed
                    from service.rag_service import RAGService
                    rag_service = RAGService()
                    
                    # Process all downloaded papers through RAG
                    rag_service.add_papers_to_vector_store(current_papers)
                    print(f"✅ RAG processing complete for {len(current_papers)} papers")
                    
                except Exception as e:
                    print(f"❌ Error in background RAG processing: {e}")
            
            # Schedule background task (fire and forget)
            import asyncio
            asyncio.create_task(process_rag_for_papers())

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
            
            # Schedule RAG processing in background (non-blocking)
            schedule_rag_processing(paper_data)
            
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
            return {
                "success": False,
                "error": "service_unavailable",
                "response": "RAG service is not available. Please check your installation."
            }
        
        # Try to get RAG service with timeout
        try:
            rag_service = get_rag_service()
        except Exception as e:
            return {
                "success": False,
                "error": "initialization_failed",
                "response": f"Failed to initialize RAG service: {str(e)}"
            }
        
        # Check if RAG service is ready
        try:
            is_ready = rag_service.is_ready()
            status = rag_service.get_status()
        except Exception as e:
            return {
                "success": False,
                "error": "status_check_failed",
                "response": f"Failed to check RAG service status: {str(e)}"
            }
        
        if not is_ready:
            # Check if models are initializing
            if not status.get("models_initialized", False):
                return {
                    "success": False,
                    "error": "initializing",
                    "response": "Chat service is initializing... Please wait a moment.",
                    "status": "Models are loading, this may take a few moments."
                }
            else:
                return {
                    "success": False,
                    "error": "not_ready",
                    "response": "Chat service is not ready. Please check your configuration.",
                    "status": status
                }
        
        # Get chat response with timeout protection
        try:
            response = rag_service.chat(request.message)
            return response
        except Exception as chat_error:
            return {
                "success": False,
                "error": "chat_failed",
                "response": f"Chat processing failed: {str(chat_error)}"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/status")
async def get_chat_status():
    """Get chat status and greeting message"""
    try:
        if not get_rag_service:
            return {
                "success": False,
                "message": "Chat service is not available. Please check your installation."
            }
        
        rag_service = get_rag_service()
        status = rag_service.get_status()
        processed_papers = rag_service.get_processed_papers()
        
        total_chunks = status.get("total_chunks", 0)
        papers_count = len(processed_papers)
        
        if not status.get("is_ready", False):
            if not status.get("models_initialized", False):
                return {
                    "success": False,
                    "message": "Chat service is initializing... Please wait a moment.",
                    "status": "initializing"
                }
            else:
                return {
                    "success": False,
                    "message": "Chat service is not ready. Please check your configuration.",
                    "status": "not_ready"
                }
        
        if papers_count == 0 and total_chunks == 0:
            return {
                "success": False,
                "message": "No papers are available for chat yet. Please add some papers and I'll process them automatically.",
                "status": "no_papers"
            }
        
        if papers_count > 0:
            return {
                "success": True,
                "message": f"Ready to chat! I have processed {papers_count} papers with {total_chunks} text segments.",
                "status": "ready",
                "papers_count": papers_count,
                "total_chunks": total_chunks
            }
        else:
            return {
                "success": True,
                "message": f"Ready to chat! I have processed papers with {total_chunks} text segments.",
                "status": "ready",
                "papers_count": 0,
                "total_chunks": total_chunks
            }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error checking chat status: {str(e)}",
            "status": "error"
        }


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
        already_processed = 0
        
        print(f"📋 Starting manual RAG processing for {len(current_papers)} papers...")
        
        for i, paper in enumerate(current_papers):
            paper_id = paper.get('arxiv_id', '')
            paper_title = paper.get('title', 'Unknown')
            
            # Check if already processed
            processed_papers = rag_service.get_processed_papers()
            if paper_id in processed_papers:
                already_processed += 1
                print(f"📋 Paper {i+1}/{len(current_papers)} already processed: {paper_title[:50]}")
            else:
                print(f"📥 Processing paper {i+1}/{len(current_papers)}: {paper_title[:50]}")
                rag_service.process_paper_async(paper)
                processed_count += 1
        
        return {
            "success": True, 
            "message": f"Started processing {processed_count} new papers for RAG. {already_processed} already processed.",
            "processed_count": processed_count,
            "already_processed": already_processed,
            "total_papers": len(current_papers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# COMPARATIVE ANALYSIS ENDPOINTS
# ===============================

@app.post("/api/analysis/compare")
async def compare_papers(request: ComparativeAnalysisRequest):
    """Compare selected papers from current papers list"""
    try:
        if not ComparativeAnalysisService:
            raise HTTPException(status_code=503, detail="Comparative analysis service not available")
        
        # Validate paper indices
        if not request.paper_indices or len(request.paper_indices) < 2:
            raise HTTPException(status_code=400, detail="At least 2 papers required for comparison")
        
        if any(idx >= len(current_papers) or idx < 0 for idx in request.paper_indices):
            raise HTTPException(status_code=400, detail="Invalid paper indices")
        
        # Get PDF paths for selected papers
        pdf_paths = []
        for idx in request.paper_indices:
            paper = current_papers[idx]
            
            # Check if paper is downloaded and has pdf_filename
            if not paper.get('downloaded') or not paper.get('pdf_filename'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"PDF not available for paper at index {idx}: {paper.get('title', 'Unknown')}"
                )
            
            # Construct full PDF path
            pdf_filename = paper.get('pdf_filename')
            pdf_path = Path("data") / pdf_filename
            
            # Check if file actually exists
            if not pdf_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"PDF file not found for paper at index {idx}: {paper.get('title', 'Unknown')} (Path: {pdf_path})"
                )
                
            pdf_paths.append(str(pdf_path))
        
        # Clean up previous analysis and get temp directory
        cleanup_temp_analysis_dir()
        temp_dir = get_temp_analysis_dir()
        output_dir = os.path.join(temp_dir, "comparison_results")
        
        # Run comparative analysis
        results = run_comparative_analysis(pdf_paths, output_dir)
        
        if results.get("success"):
            # Clean results to remove NaN values before JSON serialization
            clean_results = clean_json_data(results)
            
            response_data = {
                "success": True,
                "message": f"Comparative analysis completed for {clean_results['paper_count']} papers",
                "analysis_results": clean_results
            }
            
            # Add warnings if some papers failed
            if clean_results.get("failed_papers"):
                response_data["warnings"] = clean_results.get("warnings", [])
            
            return response_data
        else:
            error_message = results.get("error", "Analysis failed")
            
            # If we have partial results, include them (cleaned)
            if results.get("profiles"):
                clean_results = clean_json_data(results)
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": error_message,
                        "processed_papers": clean_results.get("processed_papers", 0),
                        "failed_papers": clean_results.get("failed_papers", []),
                        "partial_results": True
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=error_message)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analysis/compare-files")
async def compare_pdf_files(request: PdfUploadRequest):
    """Compare papers from provided PDF file paths"""
    try:
        if not ComparativeAnalysisService:
            raise HTTPException(status_code=503, detail="Comparative analysis service not available")
        
        # Validate input
        if not request.pdf_paths or len(request.pdf_paths) < 2:
            raise HTTPException(status_code=400, detail="At least 2 PDF files required for comparison")
        
        # Check if files exist
        missing_files = []
        for pdf_path in request.pdf_paths:
            if not Path(pdf_path).exists():
                missing_files.append(pdf_path)
        
        if missing_files:
            raise HTTPException(
                status_code=400, 
                detail=f"PDF files not found: {', '.join(missing_files)}"
            )
        
        # Clean up previous analysis and get temp directory
        cleanup_temp_analysis_dir()
        temp_dir = get_temp_analysis_dir()
        output_dir = os.path.join(temp_dir, "comparison_results")
        
        # Run comparative analysis
        results = run_comparative_analysis(request.pdf_paths, output_dir)
        
        if results.get("success"):
            # Clean results to remove NaN values before JSON serialization
            clean_results = clean_json_data(results)
            
            response_data = {
                "success": True,
                "message": f"Comparative analysis completed for {clean_results['paper_count']} papers",
                "analysis_results": clean_results
            }
            
            # Add warnings if some papers failed
            if clean_results.get("failed_papers"):
                response_data["warnings"] = clean_results.get("warnings", [])
            
            return response_data
        else:
            error_message = results.get("error", "Analysis failed")
            
            # If we have partial results, include them (cleaned)
            if results.get("profiles"):
                clean_results = clean_json_data(results)
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": error_message,
                        "processed_papers": clean_results.get("processed_papers", 0),
                        "failed_papers": clean_results.get("failed_papers", []),
                        "partial_results": True
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=error_message)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/results/{output_dir}")
async def get_analysis_results(output_dir: str):
    """Get previously generated analysis results"""
    try:
        results_path = Path(output_dir) / "comparative_analysis.json"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"Analysis results not found in {output_dir}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return {
            "success": True,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/download/{output_dir}/{file_type}")
async def download_analysis_file(output_dir: str, file_type: str):
    """Download specific analysis result files"""
    try:
        base_path = Path(output_dir)
        
        if not base_path.exists():
            raise HTTPException(status_code=404, detail=f"Output directory not found: {output_dir}")
        
        file_mappings = {
            "json": "comparative_analysis.json",
            "markdown": "analysis_report.md", 
            "csv": "comparison_table.csv",
            "metrics_plot": "metrics_comparison.png",
            "dataset_plot": "dataset_usage.png"
        }
        
        if file_type not in file_mappings:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Available: {', '.join(file_mappings.keys())}"
            )
        
        file_path = base_path / file_mappings[file_type]
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_mappings[file_type]}")
        
        # Determine media type
        media_types = {
            "json": "application/json",
            "markdown": "text/markdown",
            "csv": "text/csv",
            "metrics_plot": "image/png",
            "dataset_plot": "image/png"
        }
        
        return FileResponse(
            path=str(file_path),
            media_type=media_types[file_type],
            filename=file_mappings[file_type]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/status")
async def get_analysis_status():
    """Get status of comparative analysis service"""
    try:
        if not ComparativeAnalysisService:
            return {
                "available": False,
                "error": "Comparative analysis service not available"
            }
        
        # Test service initialization
        service = ComparativeAnalysisService()
        extractor_available = service.extractor.is_available()
        
        return {
            "available": True,
            "extractor_available": extractor_available,
            "current_papers_count": len(current_papers),
            "message": "Comparative analysis service is ready"
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.get("/api/analysis/test-data")
async def get_test_comparison_data():
    """Get test comparison data for debugging"""
    try:
        # Copy the existing comparison data to temp directory for testing
        source_file = Path("analysis_1756912905830") / "comparison.json"
        if source_file.exists():
            # Set up temp directory
            temp_dir = get_temp_analysis_dir()
            output_dir = os.path.join(temp_dir, "comparison_results")
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy the test data
            dest_file = Path(output_dir) / "comparison.json"
            shutil.copy2(source_file, dest_file)
            
            # Read and return the data
            with open(source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                "success": True,
                "analysis_results": data
            }
        else:
            return {
                "success": False,
                "error": "Test data file not found"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/analysis/cleanup")
async def cleanup_analysis_temp():
    """Clean up temporary analysis files"""
    try:
        cleanup_temp_analysis_dir()
        return {
            "success": True,
            "message": "Temporary analysis files cleaned up"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


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
        "main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
