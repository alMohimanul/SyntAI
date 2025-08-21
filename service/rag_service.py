"""
RAG (Retrieval-Augmented Generation) Service for PaperWhisperer
Handles document embedding, vector storage with FAISS, and context retrieval for chat
"""

import os
import json
import faiss
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Import required libraries with error handling
SentenceTransformer = None
Groq = None
PyPDF2 = None
extract_text = None
torch = None

try:
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformer loaded successfully")
except ImportError as e:
    print(f"⚠️ SentenceTransformer not available: {e}")

try:
    from groq import Groq
    print("✅ Groq client loaded successfully")
except ImportError as e:
    print(f"⚠️ Groq client not available: {e}")

try:
    import PyPDF2
    print("✅ PyPDF2 loaded successfully")
except ImportError as e:
    print(f"⚠️ PyPDF2 not available: {e}")

try:
    from pdfminer.high_level import extract_text
    print("✅ PDFMiner loaded successfully")
except ImportError as e:
    print(f"⚠️ PDFMiner not available: {e}")

try:
    import torch
    print("✅ PyTorch loaded successfully")
except ImportError as e:
    print(f"⚠️ PyTorch not available: {e}")


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    text: str
    paper_id: str
    paper_title: str
    page_number: int
    chunk_id: str
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_chunk: Dict[int, DocumentChunk] = {}
        self.chunk_counter = 0
        self.lock = threading.Lock()
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[DocumentChunk]) -> None:
        """Add embeddings and corresponding chunks to the vector store"""
        with self.lock:
            # Add embeddings to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store chunk mapping
            for i, chunk in enumerate(chunks):
                self.id_to_chunk[self.chunk_counter + i] = chunk
            
            self.chunk_counter += len(chunks)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        with self.lock:
            if self.index.ntotal == 0:
                return []
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32').reshape(1, -1), k)
            
            # Return chunks with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx in self.id_to_chunk:
                    results.append((self.id_to_chunk[idx], score))
            
            return results
    
    def clear(self) -> None:
        """Clear all data from the vector store"""
        with self.lock:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_to_chunk.clear()
            self.chunk_counter = 0
    
    def get_total_documents(self) -> int:
        """Get total number of documents in the store"""
        with self.lock:
            return self.index.ntotal


class RAGService:
    """Main RAG service for document processing and question answering"""
    
    def __init__(self, data_folder: str = "data", model_name: str = "all-MiniLM-L6-v2"):
        self.data_folder = Path(data_folder)
        self.model_name = model_name
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        # Initialize components
        self.embedding_model = None
        self.groq_client = None
        self.vector_store = None
        self.processed_papers: Dict[str, Dict] = {}
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.processing_lock = threading.Lock()
        
        # Initialize async
        self._initialize_async()
    
    def _initialize_async(self) -> None:
        """Initialize models asynchronously"""
        def init_models():
            try:
                # Check if required libraries are available
                if not SentenceTransformer:
                    print("⚠️ SentenceTransformer not available - RAG service will be limited")
                    return
                
                if not Groq:
                    print("⚠️ Groq client not available - RAG service will be limited")
                    return
                
                # Initialize embedding model
                print("Loading embedding model...")
                self.embedding_model = SentenceTransformer(self.model_name)
                
                # Initialize vector store with correct dimension
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.vector_store = FAISSVectorStore(dimension=embedding_dim)
                
                # Initialize Groq client
                groq_api_key = os.getenv('GROQ_API_KEY')
                if groq_api_key:
                    self.groq_client = Groq(api_key=groq_api_key)
                    print("✅ RAG Service initialized successfully")
                else:
                    print("⚠️ Warning: GROQ_API_KEY not found. Chat functionality will be limited.")
                
            except Exception as e:
                print(f"Error initializing RAG service: {e}")
                print("RAG service will be available in limited mode")
        
        # Run initialization in background
        self.thread_pool.submit(init_models)
    
    def is_ready(self) -> bool:
        """Check if the service is ready to use"""
        return (self.embedding_model is not None and 
                self.vector_store is not None and 
                self.groq_client is not None)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[int, str]:
        """Extract text from PDF, returning a dictionary mapping page numbers to text"""
        try:
            page_texts = {}
            
            # Check if required libraries are available
            if not extract_text and not PyPDF2:
                print("No PDF extraction libraries available")
                return {}
            
            # Try pdfminer first (better for academic papers)
            if extract_text:
                try:
                    full_text = extract_text(str(pdf_path))
                    if full_text.strip():
                        # For simplicity, treat as single page if pdfminer succeeds
                        page_texts[1] = full_text
                        return page_texts
                except Exception as e:
                    print(f"PDFMiner extraction failed: {e}")
            
            # Fallback to PyPDF2
            if PyPDF2:
                try:
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page_num, page in enumerate(pdf_reader.pages):
                            try:
                                text = page.extract_text()
                                if text.strip():
                                    page_texts[page_num + 1] = text
                            except Exception as e:
                                print(f"Error extracting page {page_num + 1}: {e}")
                except Exception as e:
                    print(f"PyPDF2 extraction failed: {e}")
                        
            return page_texts
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return {}
    
    def chunk_text(self, text: str, paper_id: str, paper_title: str, page_number: int) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        # Calculate chunk size in words
        chunk_words = self.chunk_size // 4  # Approximate 4 chars per word
        overlap_words = self.chunk_overlap // 4
        
        for i in range(0, len(words), chunk_words - overlap_words):
            chunk_text = ' '.join(words[i:i + chunk_words])
            
            if len(chunk_text.strip()) > 50:  # Only include substantial chunks
                chunk_id = f"{paper_id}_p{page_number}_c{len(chunks)}"
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    metadata={
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "created_at": datetime.now().isoformat()
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_paper_async(self, paper_data: Dict[str, Any]) -> None:
        """Process a paper asynchronously for RAG"""
        def process():
            try:
                if not self.embedding_model:
                    print("Embedding model not ready yet")
                    return
                
                paper_id = paper_data.get('arxiv_id', '')
                paper_title = paper_data.get('title', 'Unknown Title')
                pdf_filename = paper_data.get('pdf_filename')
                
                if not pdf_filename:
                    print(f"No PDF filename for paper {paper_id}")
                    return
                
                pdf_path = self.data_folder / pdf_filename
                if not pdf_path.exists():
                    print(f"PDF file not found: {pdf_path}")
                    return
                
                print(f"Processing paper for RAG: {paper_title[:50]}...")
                
                # Extract text from PDF
                page_texts = self.extract_text_from_pdf(pdf_path)
                if not page_texts:
                    print(f"No text extracted from {pdf_path}")
                    return
                
                # Create chunks
                all_chunks = []
                for page_num, page_text in page_texts.items():
                    chunks = self.chunk_text(page_text, paper_id, paper_title, page_num)
                    all_chunks.extend(chunks)
                
                if not all_chunks:
                    print(f"No chunks created for paper {paper_id}")
                    return
                
                # Create embeddings in batches
                batch_size = 32
                chunk_texts = [chunk.text for chunk in all_chunks]
                
                embeddings_list = []
                for i in range(0, len(chunk_texts), batch_size):
                    batch_texts = chunk_texts[i:i + batch_size]
                    batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
                    embeddings_list.append(batch_embeddings)
                
                # Combine all embeddings
                all_embeddings = np.vstack(embeddings_list)
                
                # Add to vector store
                self.vector_store.add_embeddings(all_embeddings, all_chunks)
                
                # Track processed papers
                with self.processing_lock:
                    self.processed_papers[paper_id] = {
                        "title": paper_title,
                        "chunks_count": len(all_chunks),
                        "processed_at": datetime.now().isoformat()
                    }
                
                print(f"✅ Processed {len(all_chunks)} chunks for paper: {paper_title[:50]}")
                
            except Exception as e:
                print(f"Error processing paper {paper_data.get('title', 'Unknown')}: {e}")
        
        # Submit to thread pool
        self.thread_pool.submit(process)
    
    def search_relevant_context(self, query: str, max_chunks: int = 5) -> List[DocumentChunk]:
        """Search for relevant context chunks for a query"""
        try:
            if not self.embedding_model or not self.vector_store:
                return []
            
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k=max_chunks)
            
            # Return only the chunks (without scores for now)
            return [chunk for chunk, score in results]
            
        except Exception as e:
            print(f"Error searching context: {e}")
            return []
    
    def generate_response(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        """Generate response using Groq with retrieved context"""
        try:
            if not self.groq_client:
                return "Chat service is not available. Please check your GROQ_API_KEY configuration."
            
            # Prepare context
            context_text = ""
            for i, chunk in enumerate(context_chunks):
                context_text += f"\n[Context {i+1} - {chunk.paper_title}, Page {chunk.page_number}]:\n{chunk.text}\n"
            
            # Create system prompt
            system_prompt = """You are a research assistant helping users understand academic papers. 
            Use the provided context to answer questions accurately and cite specific papers when relevant.
            If the context doesn't contain enough information, say so clearly.
            Keep responses concise but informative."""
            
            # Create user prompt
            user_prompt = f"""Context from research papers:
{context_text}

Question: {query}

Please provide a helpful answer based on the context above."""
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",  # Fast model for responsive chat
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat interface"""
        try:
            # Check if service is ready
            if not self.is_ready():
                return {
                    "success": False,
                    "message": "RAG service is not available. Missing required dependencies: sentence-transformers, groq, or GROQ_API_KEY not configured.",
                    "context_count": 0
                }
            
            # Check if we have any processed papers
            if self.vector_store.get_total_documents() == 0:
                return {
                    "success": True,
                    "message": "I don't have any papers to search through yet. Please add some papers first, and I'll process them for our conversation.",
                    "context_count": 0
                }
            
            # Search for relevant context
            context_chunks = self.search_relevant_context(query, max_chunks=3)
            
            if not context_chunks:
                return {
                    "success": True,
                    "message": "I couldn't find relevant information in the available papers for your question. Could you try rephrasing or asking about different aspects of the research?",
                    "context_count": 0
                }
            
            # Generate response
            response = self.generate_response(query, context_chunks)
            
            return {
                "success": True,
                "message": response,
                "context_count": len(context_chunks),
                "sources": [
                    {
                        "title": chunk.paper_title,
                        "page": chunk.page_number,
                        "paper_id": chunk.paper_id
                    } for chunk in context_chunks
                ]
            }
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return {
                "success": False,
                "message": f"An error occurred while processing your question: {str(e)}",
                "context_count": 0
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the RAG service"""
        return {
            "is_ready": self.is_ready(),
            "total_chunks": self.vector_store.get_total_documents() if self.vector_store else 0,
            "processed_papers": len(self.processed_papers),
            "papers_list": list(self.processed_papers.values())
        }
    
    def clear_all_data(self) -> None:
        """Clear all processed data"""
        if self.vector_store:
            self.vector_store.clear()
        
        with self.processing_lock:
            self.processed_papers.clear()


# Global RAG service instance
rag_service = RAGService()


def get_rag_service() -> RAGService:
    """Get the global RAG service instance"""
    return rag_service
