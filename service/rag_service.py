"""
RAG (Retrieval-Augmented Generation) Service for PaperWhisperer
Handles document embedding, vector storage with FAISS, and context retrieval for chat
Optimized with lazy loading and improved paper-aware context management
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

# Import required libraries with error handling - DEFERRED IMPORTS
SentenceTransformer = None
Groq = None
PyPDF2 = None
extract_text = None
torch = None
RecursiveCharacterTextSplitter = None
Document = None

# Note: sentence_transformers import is deferred to avoid PyTorch compatibility issues
# It will be imported when actually needed in _ensure_models_loaded()

try:
    from groq import Groq
except ImportError:
    pass

try:
    import PyPDF2
except ImportError:
    pass

try:
    from pdfminer.high_level import extract_text
except ImportError:
    pass

try:
    import torch
except ImportError:
    pass

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    pass


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    text: str
    paper_id: str
    paper_title: str
    page_number: int
    chunk_id: str
    metadata: Dict[str, Any]
    paper_order: int = 0  # Track paper processing order
    chunk_index: int = 0  # Track chunk order within paper
    
    def to_langchain_doc(self):
        """Convert to LangChain Document format"""
        if Document:
            return Document(
                page_content=self.text,
                metadata={
                    'paper_id': self.paper_id,
                    'paper_title': self.paper_title,
                    'page_number': self.page_number,
                    'chunk_id': self.chunk_id,
                    'paper_order': self.paper_order,
                    'chunk_index': self.chunk_index,
                    **self.metadata
                }
            )
        return None


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search with paper-aware capabilities"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_chunk: Dict[int, DocumentChunk] = {}
        self.paper_to_chunk_ids: Dict[str, List[int]] = {}  # Track chunks by paper
        self.chunk_counter = 0
        self.lock = threading.Lock()
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[DocumentChunk]) -> None:
        """Add embeddings and corresponding chunks to the vector store"""
        with self.lock:
            # Add embeddings to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store chunk mapping and paper tracking
            for i, chunk in enumerate(chunks):
                chunk_id = self.chunk_counter + i
                self.id_to_chunk[chunk_id] = chunk
                
                # Track chunks by paper
                if chunk.paper_id not in self.paper_to_chunk_ids:
                    self.paper_to_chunk_ids[chunk.paper_id] = []
                self.paper_to_chunk_ids[chunk.paper_id].append(chunk_id)
            
            self.chunk_counter += len(chunks)
    
    def search(self, query_embedding: np.ndarray, k: int = 5, paper_filter: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks with optional paper filtering"""
        with self.lock:
            if self.index.ntotal == 0:
                return []
            
            if paper_filter and paper_filter in self.paper_to_chunk_ids:
                # Search within specific paper
                paper_chunk_ids = set(self.paper_to_chunk_ids[paper_filter])
                
                # Get all embeddings first
                all_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
                
                # Filter embeddings for this paper
                filtered_embeddings = []
                filtered_ids = []
                for chunk_id in range(self.chunk_counter):
                    if chunk_id in paper_chunk_ids and chunk_id in self.id_to_chunk:
                        filtered_embeddings.append(all_embeddings[chunk_id])
                        filtered_ids.append(chunk_id)
                
                if not filtered_embeddings:
                    return []
                
                # Create temporary index for paper-specific search
                temp_index = faiss.IndexFlatL2(self.dimension)
                temp_index.add(np.array(filtered_embeddings).astype('float32'))
                
                # Search in temporary index
                scores, indices = temp_index.search(query_embedding.astype('float32').reshape(1, -1), min(k, len(filtered_ids)))
                
                # Map back to original chunk IDs
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(filtered_ids):
                        original_chunk_id = filtered_ids[idx]
                        if original_chunk_id in self.id_to_chunk:
                            results.append((self.id_to_chunk[original_chunk_id], float(score)))
                
                return results
            else:
                # Global search across all papers
                scores, indices = self.index.search(query_embedding.astype('float32').reshape(1, -1), k)
                
                # Return chunks with scores
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx in self.id_to_chunk:
                        results.append((self.id_to_chunk[idx], float(score)))
                
                return results
    
    def get_papers(self) -> List[str]:
        """Get list of paper IDs in the vector store"""
        with self.lock:
            return list(self.paper_to_chunk_ids.keys())
    
    def get_paper_chunks(self, paper_id: str) -> List[DocumentChunk]:
        """Get all chunks for a specific paper"""
        with self.lock:
            if paper_id not in self.paper_to_chunk_ids:
                return []
            
            chunks = []
            for chunk_id in self.paper_to_chunk_ids[paper_id]:
                if chunk_id in self.id_to_chunk:
                    chunks.append(self.id_to_chunk[chunk_id])
            
            # Sort by chunk index to maintain order
            chunks.sort(key=lambda x: x.chunk_index)
            return chunks
    
    def get_total_documents(self) -> int:
        """Get total number of chunks"""
        with self.lock:
            return len(self.id_to_chunk)
    
    def clear(self) -> None:
        """Clear all data"""
        with self.lock:
            self.index.reset()
            self.id_to_chunk.clear()
            self.paper_to_chunk_ids.clear()
            self.chunk_counter = 0


class RAGService:
    """RAG service with improved chunking and paper-aware context management"""
    
    def __init__(self, data_folder: str = "data", model_name: str = "all-MiniLM-L6-v2"):
        self.data_folder = Path(data_folder)
        self.model_name = model_name
        
        # Initialize components
        self.embedding_model = None
        self.groq_client = None
        self.vector_store = None
        self.text_splitter = None
        self.processed_papers: Dict[str, Dict] = {}
        self.paper_processing_order: List[str] = []  # Track paper order
        self._models_initialized = False
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.processing_lock = threading.Lock()
        
        # Initialize data folder only
        self.data_folder.mkdir(exist_ok=True)
        
        print("✅ RAG service created (models will be loaded when needed)")
    
    def _ensure_models_loaded(self) -> None:
        """Ensure models are loaded (lazy initialization)"""
        if self._models_initialized:
            return
            
        with self.processing_lock:
            if self._models_initialized:  # Double-check pattern
                return
                
            try:
                # Import sentence_transformers here to avoid module-level import issues
                global SentenceTransformer, Groq
                
                if not SentenceTransformer:
                    try:
                        print("📥 Importing SentenceTransformer (this may take a moment)...")
                        from sentence_transformers import SentenceTransformer
                        print("✅ SentenceTransformer imported successfully")
                    except Exception as e:
                        print(f"⚠️ Failed to import SentenceTransformer: {e}")
                        print("RAG service will be limited - chat functionality disabled")
                        self._models_initialized = True
                        return
                
                if not Groq:
                    try:
                        from groq import Groq
                    except ImportError:
                        print("⚠️ Groq client not available - RAG service will be limited")
                        self._models_initialized = True
                        return
                
                # Initialize embedding model
                print("📥 Loading embedding model (this may take a moment)...")
                self.embedding_model = SentenceTransformer(self.model_name)
                
                # Initialize text splitter with LangChain
                if RecursiveCharacterTextSplitter:
                    self.text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    print("✅ LangChain text splitter initialized")
                else:
                    print("⚠️ LangChain not available, using fallback chunking")
                
                # Initialize vector store with correct dimension
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.vector_store = FAISSVectorStore(dimension=embedding_dim)
                
                # Initialize Groq client
                groq_api_key = os.getenv('GROQ_API_KEY')
                if groq_api_key:
                    self.groq_client = Groq(api_key=groq_api_key)
                    print("✅ RAG Service models loaded successfully")
                else:
                    print("⚠️ Warning: GROQ_API_KEY not found. Chat functionality will be limited.")
                
                self._models_initialized = True
                
            except Exception as e:
                print(f"Error initializing RAG service models: {e}")
                print("RAG service will be available in limited mode")
                self._models_initialized = True
    
    def is_ready(self) -> bool:
        """Check if the service is ready to use"""
        if not self._models_initialized:
            return False
        return (self.embedding_model is not None and 
                self.vector_store is not None and 
                self.groq_client is not None)
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            "models_initialized": self._models_initialized,
            "embedding_model_ready": self.embedding_model is not None,
            "vector_store_ready": self.vector_store is not None,
            "groq_client_ready": self.groq_client is not None,
            "is_ready": self.is_ready(),
            "processed_papers": len(self.processed_papers),
            "total_chunks": self.vector_store.get_total_documents() if self.vector_store else 0
        }
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> Dict[int, str]:
        """Extract text from PDF with fallback methods"""
        text_by_page = {}
        
        # Try PDFMiner first
        if extract_text:
            try:
                full_text = extract_text(str(pdf_path))
                # Simple page splitting (not perfect but functional)
                pages = full_text.split('\n\n')
                for i, page in enumerate(pages):
                    if page.strip():
                        text_by_page[i + 1] = page.strip()
                return text_by_page
            except Exception as e:
                print(f"PDFMiner failed: {e}")
        
        # Fallback to PyPDF2
        if PyPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            text_by_page[page_num + 1] = text.strip()
                return text_by_page
            except Exception as e:
                print(f"PyPDF2 failed: {e}")
        
        print("No PDF extraction method available")
        return {}
    
    def _create_chunks(self, text: str, paper_id: str, paper_title: str, page_number: int, paper_order: int) -> List[DocumentChunk]:
        """Create semantic chunks from text using LangChain or fallback method"""
        chunks = []
        
        if self.text_splitter and Document:
            # Use LangChain recursive text splitter
            try:
                # Create a temporary document
                doc = Document(page_content=text, metadata={
                    'paper_id': paper_id,
                    'paper_title': paper_title,
                    'page_number': page_number
                })
                
                # Split the document
                split_docs = self.text_splitter.split_documents([doc])
                
                for i, split_doc in enumerate(split_docs):
                    chunk_id = f"{paper_id}_p{page_number}_c{i}"
                    chunk = DocumentChunk(
                        text=split_doc.page_content,
                        paper_id=paper_id,
                        paper_title=paper_title,
                        page_number=page_number,
                        chunk_id=chunk_id,
                        paper_order=paper_order,
                        chunk_index=i,
                        metadata={
                            "word_count": len(split_doc.page_content.split()),
                            "char_count": len(split_doc.page_content),
                            "created_at": datetime.now().isoformat(),
                            "chunking_method": "langchain_recursive"
                        }
                    )
                    chunks.append(chunk)
                
                return chunks
                
            except Exception as e:
                print(f"LangChain chunking failed, using fallback: {e}")
        
        # Fallback to simple word-based chunking
        words = text.split()
        chunk_size = 500
        chunk_overlap = 50
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Skip very short chunks
                chunk_id = f"{paper_id}_p{page_number}_c{len(chunks)}"
                chunk = DocumentChunk(
                    text=chunk_text,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    paper_order=paper_order,
                    chunk_index=len(chunks),
                    metadata={
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "created_at": datetime.now().isoformat(),
                        "chunking_method": "fallback_word_based"
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_paper_async(self, paper_data: Dict[str, Any]) -> None:
        """Process a paper asynchronously for RAG with improved chunking"""
        def process():
            try:
                # Ensure models are loaded
                self._ensure_models_loaded()
                
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
                
                # Track paper processing order
                with self.processing_lock:
                    if paper_id not in self.paper_processing_order:
                        self.paper_processing_order.append(paper_id)
                    paper_order = self.paper_processing_order.index(paper_id)
                
                # Extract text from PDF
                text_by_page = self._extract_text_from_pdf(pdf_path)
                if not text_by_page:
                    print(f"Could not extract text from {pdf_filename}")
                    return
                
                # Create chunks from all pages with improved chunking
                all_chunks = []
                for page_num, text in text_by_page.items():
                    chunks = self._create_chunks(text, paper_id, paper_title, page_num, paper_order)
                    all_chunks.extend(chunks)
                
                if not all_chunks:
                    print(f"No chunks created for paper {paper_id}")
                    return
                
                # Create embeddings in batches
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i + batch_size]
                    batch_texts = [chunk.text for chunk in batch_chunks]
                    batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
                    all_embeddings.append(batch_embeddings)
                
                # Combine all embeddings
                all_embeddings = np.vstack(all_embeddings)
                
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
    
    def _identify_target_paper(self, query: str) -> Optional[str]:
        """Identify which paper the user is asking about"""
        query_lower = query.lower()

        print(f"🔍 Identifying target paper for query: '{query}'")
        print(f"📋 Current paper processing order: {self.paper_processing_order}")
        
        # Show available papers with their titles for debugging
        print(f"📚 Available papers:")
        for i, paper_id in enumerate(self.paper_processing_order, 1):
            title = self.processed_papers.get(paper_id, {}).get('title', 'Unknown')
            print(f"  {i}. {paper_id}: {title[:80]}...")
                # Check for specific paper numbers (more precise matching)
        import re

        # Look for patterns like "paper 1", "paper 2", "paper 3", etc.
        paper_number_match = re.search(r'paper\s+(\d+)', query_lower)
        if paper_number_match:
            paper_num = int(paper_number_match.group(1))
            if 0 < paper_num <= len(self.paper_processing_order):
                target_paper = self.paper_processing_order[paper_num - 1]
                print(f"🎯 Identified target paper {paper_num}: {target_paper}")
                return target_paper
        
        # Check for ordinal references
        if "first paper" in query_lower or "1st paper" in query_lower:
            if self.paper_processing_order:
                target_paper = self.paper_processing_order[0]
                print(f"🎯 Identified first paper: {target_paper}")
                return target_paper
        elif "second paper" in query_lower or "2nd paper" in query_lower:
            if len(self.paper_processing_order) > 1:
                target_paper = self.paper_processing_order[1]
                print(f"🎯 Identified second paper: {target_paper}")
                return target_paper
        elif "third paper" in query_lower or "3rd paper" in query_lower:
            if len(self.paper_processing_order) > 2:
                target_paper = self.paper_processing_order[2]
                print(f"🎯 Identified third paper: {target_paper}")
                return target_paper
        elif "fourth paper" in query_lower or "4th paper" in query_lower:
            if len(self.paper_processing_order) > 3:
                target_paper = self.paper_processing_order[3]
                print(f"🎯 Identified fourth paper: {target_paper}")
                return target_paper
        elif "fifth paper" in query_lower or "5th paper" in query_lower:
            if len(self.paper_processing_order) > 4:
                target_paper = self.paper_processing_order[4]
                print(f"🎯 Identified fifth paper: {target_paper}")
                return target_paper
        elif "last paper" in query_lower or "latest paper" in query_lower:
            if self.paper_processing_order:
                target_paper = self.paper_processing_order[-1]
                print(f"🎯 Identified last paper: {target_paper}")
                return target_paper
        
        # Check for "the paper" - if only one paper, assume that's it
        if ("the paper" in query_lower or "this paper" in query_lower) and len(self.paper_processing_order) == 1:
            target_paper = self.paper_processing_order[0]
            print(f"🎯 Identified the paper: {target_paper}")
            return target_paper
        
        # Check for title keywords (only if no specific paper number was mentioned)
        if not paper_number_match:
            print(f"🔍 Checking title keywords in query: '{query_lower}'")
            
            # First try exact phrase matching
            for paper_id in self.processed_papers:
                paper_title = self.processed_papers[paper_id].get('title', '').lower()
                print(f"📄 Checking paper {paper_id}: '{paper_title[:50]}...'")
                
                # Check if significant parts of the title appear in query
                title_words = paper_title.split()
                
                # Look for sequences of 2-3 words from title in query
                for i in range(len(title_words) - 1):
                    phrase2 = f"{title_words[i]} {title_words[i+1]}"
                    if len(phrase2) > 8 and phrase2 in query_lower:  # Skip very short phrases
                        print(f"🎯 Identified paper by title phrase '{phrase2}': {paper_id}")
                        return paper_id
                    
                    if i < len(title_words) - 2:
                        phrase3 = f"{title_words[i]} {title_words[i+1]} {title_words[i+2]}"
                        if phrase3 in query_lower:
                            print(f"🎯 Identified paper by title phrase '{phrase3}': {paper_id}")
                            return paper_id
            
            # If no phrase match, try individual important keywords
            for paper_id in self.processed_papers:
                paper_title = self.processed_papers[paper_id].get('title', '').lower()
                # Extract key words from title (excluding common words)
                title_words = [word for word in paper_title.split() 
                              if len(word) > 4 and word not in ['paper', 'study', 'analysis', 'the', 'and', 'for', 'with', 'from', 'that', 'this', 'using', 'based', 'approach']]
                
                # Count matching keywords
                matches = 0
                matched_words = []
                for word in title_words:
                    if word in query_lower:
                        matches += 1
                        matched_words.append(word)
                
                # If we have multiple keyword matches, it's likely the right paper
                if matches >= 2:
                    print(f"🎯 Identified paper by multiple title keywords {matched_words}: {paper_id}")
                    return paper_id
        
        # Last resort: try fuzzy matching on title similarity
        print(f"🔄 Trying fuzzy title matching...")
        best_match = None
        best_score = 0
        
        for paper_id in self.processed_papers:
            paper_title = self.processed_papers[paper_id].get('title', '').lower()
            
            # Calculate simple word overlap score
            query_words = set(query_lower.split())
            title_words = set(paper_title.split())
            
            # Remove common words
            common_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'a', 'an'}
            query_words = query_words - common_words
            title_words = title_words - common_words
            
            if title_words:  # Avoid division by zero
                overlap_score = len(query_words & title_words) / len(title_words)
                if overlap_score > best_score and overlap_score > 0.3:  # At least 30% overlap
                    best_score = overlap_score
                    best_match = paper_id
        
        if best_match:
            print(f"🎯 Identified paper by fuzzy matching (score: {best_score:.2f}): {best_match}")
            return best_match
        
        print(f"❓ No specific paper identified for query: '{query[:50]}'")
        return None
    
    def _get_comprehensive_paper_context(self, paper_id: str, query: str, max_chunks: int) -> List[DocumentChunk]:
        """Get comprehensive context from a specific paper"""
        try:
            print(f"🔍 Getting context for paper: {paper_id}")

            # Get ALL chunks from the target paper
            all_paper_chunks = self.vector_store.get_paper_chunks(paper_id)

            if not all_paper_chunks:
                print(f"⚠️ No chunks found for paper {paper_id}")
                # Try alternative paper ID formats
                alternative_ids = [
                    paper_id.replace('v1', ''),  # Remove version
                    paper_id.split('v')[0],       # Remove version
                    f"{paper_id}v1",             # Add version
                ]
                for alt_id in alternative_ids:
                    if alt_id != paper_id:
                        print(f"🔄 Trying alternative paper ID: {alt_id}")
                        all_paper_chunks = self.vector_store.get_paper_chunks(alt_id)
                        if all_paper_chunks:
                            print(f"✅ Found chunks with alternative ID: {alt_id}")
                            paper_id = alt_id  # Update to working ID
                            break

                if not all_paper_chunks:
                    print(f"❌ No chunks found for paper {paper_id} with any ID format")
                    return []

            print(f"📄 Found {len(all_paper_chunks)} total chunks for paper {paper_id}")
            print(f"� Sample chunk paper_id: {all_paper_chunks[0].paper_id if all_paper_chunks else 'None'}")
            print(f"📋 Sample chunk title: {all_paper_chunks[0].paper_title[:50] if all_paper_chunks else 'None'}")
            
            # Strategy 1: If the paper is small (< 20 chunks), return ALL chunks
            if len(all_paper_chunks) <= 20:
                print(f"📋 Small paper detected, returning all {len(all_paper_chunks)} chunks")
                return all_paper_chunks
            
            # Strategy 2: For larger papers, use semantic search within the paper + strategic sampling
            query_embedding = self.embedding_model.encode([query])
            
            # Get semantically relevant chunks from this paper
            semantic_results = self.vector_store.search(query_embedding, k=max_chunks, paper_filter=paper_id)
            semantic_chunks = [chunk for chunk, score in semantic_results]
            
            # Add strategic chunks (beginning, middle, end of paper) for comprehensive coverage
            strategic_chunks = []
            total_chunks = len(all_paper_chunks)
            
            # Add first few chunks (introduction/abstract)
            strategic_chunks.extend(all_paper_chunks[:3])
            
            # Add middle chunks (methodology/results)
            middle_start = total_chunks // 3
            middle_end = 2 * total_chunks // 3
            strategic_chunks.extend(all_paper_chunks[middle_start:middle_start + 2])
            
            # Add last few chunks (conclusion)
            strategic_chunks.extend(all_paper_chunks[-3:])
            
            # Combine semantic + strategic chunks, remove duplicates
            combined_chunks = []
            seen_chunk_ids = set()
            
            # Priority 1: Semantic chunks
            for chunk in semantic_chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    combined_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
            
            # Priority 2: Strategic chunks
            for chunk in strategic_chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    combined_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
            
            # If we still have room and it's a specific paper query, add more chunks
            remaining_slots = max(15, max_chunks * 3) - len(combined_chunks)  # Allow up to 15+ chunks for single paper
            if remaining_slots > 0:
                for chunk in all_paper_chunks:
                    if chunk.chunk_id not in seen_chunk_ids:
                        combined_chunks.append(chunk)
                        remaining_slots -= 1
                        if remaining_slots <= 0:
                            break
            
            print(f"📚 Returning {len(combined_chunks)} chunks from target paper")
            return combined_chunks
            
        except Exception as e:
            print(f"Error getting comprehensive paper context: {e}")
            return []
    
    def _get_global_context(self, query: str, max_chunks: int) -> List[DocumentChunk]:
        """Get context from across all papers when no specific paper is identified"""
        try:
            query_embedding = self.embedding_model.encode([query])
            
            # Search across all papers
            results = self.vector_store.search(query_embedding, k=max_chunks)
            
            # Return chunks sorted by relevance
            sorted_results = sorted(results, key=lambda x: x[1])
            chunks = [chunk for chunk, score in sorted_results]
            
            print(f"🌐 Global search returning {len(chunks)} chunks from multiple papers")
            return chunks
            
        except Exception as e:
            print(f"Error in global context search: {e}")
            return []
    
    def search_relevant_context(self, query: str, max_chunks: int = 5, paper_filter: Optional[str] = None) -> List[DocumentChunk]:
        """Search for relevant context chunks with comprehensive paper-aware strategy"""
        try:
            # Ensure models are loaded
            self._ensure_models_loaded()
            
            if not self.embedding_model or not self.vector_store:
                return []
            
            # Try to identify target paper if not specified
            if not paper_filter:
                paper_filter = self._identify_target_paper(query)
            
            print(f"🔍 Query: '{query[:50]}...'")
            print(f"🎯 Target paper identified: {paper_filter}")
            print(f"📋 Available processed papers: {list(self.processed_papers.keys())}")

            # If a specific paper is identified, use comprehensive paper-focused strategy
            if paper_filter and paper_filter in self.processed_papers:
                print(f"📄 Using comprehensive context for paper: {paper_filter}")
                return self._get_comprehensive_paper_context(paper_filter, query, max_chunks)
            else:
                print(f"🌐 Using global context search (paper_filter: {paper_filter})")
                # Global search across all papers with higher chunk limit
                return self._get_global_context(query, max_chunks * 2)  # Double the chunks for global search
            
        except Exception as e:
            print(f"Error searching context: {e}")
            return []
    
    def chat(self, query: str, max_chunks: int = 5) -> Dict[str, Any]:
        """Chat with the RAG system using comprehensive context strategy"""
        try:
            # Ensure models are loaded
            self._ensure_models_loaded()
            
            if not self.groq_client:
                return {
                    "success": False,
                    "error": "Groq client not available",
                    "response": "Chat functionality is not available. Please check your GROQ_API_KEY."
                }
            
            # Get relevant context with comprehensive strategy
            context_chunks = self.search_relevant_context(query, max_chunks)
            
            if not context_chunks:
                # Try to provide helpful guidance
                available_papers = list(self.processed_papers.keys())
                if available_papers:
                    paper_info = []
                    for i, paper_id in enumerate(self.paper_processing_order[:5], 1):
                        title = self.processed_papers[paper_id].get('title', 'Unknown')[:60]
                        paper_info.append(f"{i}. {title}")
                    
                    guidance = f"I have {len(available_papers)} papers available:\n" + "\n".join(paper_info)
                    if len(available_papers) > 5:
                        guidance += f"\n... and {len(available_papers) - 5} more papers."
                else:
                    guidance = "No papers have been processed yet."
                
                return {
                    "success": False,
                    "error": "No relevant context found",
                    "response": f"I don't have enough information to answer your question.\n\n{guidance}"
                }
            
            # Determine if this is a single-paper or multi-paper query
            target_paper = self._identify_target_paper(query)
            is_single_paper_query = target_paper is not None

            print(f"🎯 Chat target paper: {target_paper}")
            print(f"📊 Context chunks returned: {len(context_chunks)}")
            if context_chunks:
                print(f"📄 First chunk paper_id: {context_chunks[0].paper_id}")
                print(f"📄 First chunk paper_title: {context_chunks[0].paper_title[:50]}")
                print(f"📄 First chunk paper_order: {context_chunks[0].paper_order}")

            # Build context string with adaptive organization
            if is_single_paper_query:
                # Single paper - comprehensive context
                paper_title = context_chunks[0].paper_title if context_chunks else "Unknown"
                paper_order = None
                if target_paper in self.paper_processing_order:
                    paper_order = self.paper_processing_order.index(target_paper) + 1
                
                context_header = f"=== COMPREHENSIVE CONTEXT FROM PAPER {paper_order if paper_order else ''}: {paper_title} ===\n\n"
                
                # Organize chunks by page for better structure
                chunks_by_page = {}
                for chunk in context_chunks:
                    page = chunk.page_number
                    if page not in chunks_by_page:
                        chunks_by_page[page] = []
                    chunks_by_page[page].append(chunk.text)
                
                context_parts = []
                for page in sorted(chunks_by_page.keys()):
                    page_content = f"[Page {page}]\n" + "\n\n".join(chunks_by_page[page])
                    context_parts.append(page_content)
                
                context_text = context_header + "\n\n".join(context_parts)
                
                # Enhanced prompt for single paper
                prompt = f"""You are analyzing a specific research paper. Based on the comprehensive content provided below, please answer the user's question thoroughly and accurately.

{context_text}

User question: {query}

Instructions:
- Focus specifically on this paper since the user asked about it directly
- Provide a comprehensive answer using the full context available
- Include specific details, methodologies, results, and conclusions from this paper
- If the user asked about "paper {paper_order}" or similar, acknowledge which paper you're discussing
- Structure your response clearly with key findings and insights"""

            else:
                # Multi-paper query - organize by paper
                papers_mentioned = {}
                for chunk in context_chunks:
                    if chunk.paper_id not in papers_mentioned:
                        paper_order = None
                        if chunk.paper_id in self.paper_processing_order:
                            paper_order = self.paper_processing_order.index(chunk.paper_id) + 1
                        
                        papers_mentioned[chunk.paper_id] = {
                            'title': chunk.paper_title,
                            'order': paper_order,
                            'chunks': []
                        }
                    papers_mentioned[chunk.paper_id]['chunks'].append(chunk.text)
                
                context_parts = []
                for paper_id, paper_info in papers_mentioned.items():
                    paper_header = f"=== PAPER {paper_info['order'] if paper_info['order'] else '?'}: {paper_info['title']} ==="
                    paper_context = paper_header + "\n\n" + "\n\n".join(paper_info['chunks'])
                    context_parts.append(paper_context)
                
                context_text = "\n\n" + "="*80 + "\n\n".join(context_parts)
                
                # Enhanced prompt for multi-paper
                prompt = f"""Based on the following research paper content from multiple papers, please answer the user's question accurately and comprehensively.

{context_text}

User question: {query}

Instructions:
- Draw insights from all relevant papers provided
- Compare and contrast findings across papers when appropriate
- Clearly indicate which paper specific information comes from
- Provide a well-structured response that synthesizes the information"""

            response = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,  
                temperature=0.4,
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "context_used": len(context_chunks),
                "papers_referenced": len(set(chunk.paper_id for chunk in context_chunks)),
                "papers_referenced_list": list(set(chunk.paper_title for chunk in context_chunks)),
                "target_paper_identified": target_paper,
                "is_single_paper_query": is_single_paper_query,
                "comprehensive_context": is_single_paper_query,
                "sources": [
                    {
                        "title": chunk.paper_title,
                        "page": chunk.page_number,
                        "chunk_id": chunk.chunk_id,
                        "paper_order": self.paper_processing_order.index(chunk.paper_id) + 1 if chunk.paper_id in self.paper_processing_order else None
                    } for chunk in context_chunks[:10]  # Limit sources display to prevent overflow
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while processing your query: {e}"
            }
    
    def get_processed_papers(self) -> Dict[str, Dict]:
        """Get list of processed papers"""
        with self.processing_lock:
            return self.processed_papers.copy()
    
    def clear_all_data(self) -> None:
        """Clear all processed data"""
        try:
            if self.vector_store:
                self.vector_store.clear()
            
            with self.processing_lock:
                self.processed_papers.clear()
                self.paper_processing_order.clear()
            
            print("✅ All RAG data cleared")
            
        except Exception as e:
            print(f"Error clearing data: {e}")
    
    def get_paper_info(self, paper_id: str = None) -> Dict[str, Any]:
        """Get information about a specific paper or all papers"""
        with self.processing_lock:
            if paper_id:
                if paper_id in self.processed_papers:
                    return {
                        "paper": self.processed_papers[paper_id],
                        "order": self.paper_processing_order.index(paper_id) + 1 if paper_id in self.paper_processing_order else 0
                    }
                return {}
            else:
                return {
                    "total_papers": len(self.processed_papers),
                    "processing_order": self.paper_processing_order,
                    "papers": self.processed_papers
                }
    
    def get_paper_chunks_info(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed chunk information for a paper"""
        if not self.vector_store:
            return {}
        
        chunks = self.vector_store.get_paper_chunks(paper_id)
        return {
            "paper_id": paper_id,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page_number,
                    "word_count": chunk.metadata.get("word_count", 0),
                    "preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                } for chunk in chunks
            ]
        }


# Global RAG service instance (lazy initialized)
_rag_service_instance = None


def get_rag_service() -> RAGService:
    """Get the global RAG service instance (lazy initialization)"""
    global _rag_service_instance
    if _rag_service_instance is None:
        print("🔄 Initializing RAG service...")
        _rag_service_instance = RAGService()
    return _rag_service_instance
