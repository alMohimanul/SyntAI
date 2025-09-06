# SyntAI - AI Research Assistant 🤖📚

**SyntAI** is a powerful AI-driven research assistant that helps researchers discover, analyze, and compare academic papers from ArXiv. Built with FastAPI backend and modern HTML/Tailwind CSS frontend, it combines multiple AI services to provide comprehensive paper analysis, intelligent chat capabilities, and comparative analysis across multiple research papers.

## 🌟 Features

### 📖 Paper Discovery & Management
- **ArXiv Integration**: Search and discover papers by domain/category or keywords
- **Advanced Search**: Filter by time range, sort by relevance or date
- **PDF Management**: Automatic PDF download and storage
- **Paper Preview**: Built-in PDF viewer with page navigation
- **Metadata Extraction**: Automatic extraction of titles, authors, abstracts, and categories

### 🤖 AI-Powered Analysis
- **Paper Summarization**: AI-generated summaries using Groq LLM
- **Content Analysis**: Automatic detection of diagrams, methodology, and insights
- **Code Generation**: Extract and generate code snippets from papers
- **Multi-modal Processing**: Handle text, images, and diagrams in research papers

### 💬 Intelligent Chat System
- **RAG (Retrieval-Augmented Generation)**: Chat with your papers using context-aware AI
- **FAISS Vector Search**: Fast semantic search across paper contents
- **Source Attribution**: Track which papers contribute to each answer
- **Context-Aware Responses**: Maintains conversation context for better answers

### 📊 Comparative Analysis
- **Multi-Paper Comparison**: Compare methodologies, results, and approaches across papers
- **Structured Analysis**: Extract key strengths, limitations, and contributions
- **Visual Comparisons**: Generate charts and tables for easy comparison
- **Export Capabilities**: Download analysis results in multiple formats

### 🌐 Modern Web Interface
- **Responsive Design**: Built with Tailwind CSS for modern UI/UX
- **Real-time Updates**: Server-sent events for live progress updates
- **Interactive Elements**: Drag-and-drop, modal dialogs, and smooth animations
- **Multi-tab Interface**: Separate tabs for different search methods

## 🏗️ Architecture

```
SyntAI/
├── main.py                 # FastAPI server and API endpoints
├── requirements.txt        # Python dependencies
├── agents/                 # AI agents and scrapers
│   └── research_agent.py   # ArXiv scraper and paper discovery
├── service/                # Core services
│   ├── llm.py             # Language model services (Groq)
│   ├── pdf_parser.py      # PDF text extraction and processing
│   ├── rag_service.py     # RAG implementation with FAISS
│   └── comparative_analysis_service.py  # Multi-paper analysis
├── frontend/              # Web interface
│   ├── index.html         # Main application UI
│   ├── comparative_analysis.html  # Comparison interface
│   └── app.js            # Frontend JavaScript logic
├── data/                  # Paper storage and metadata
├── docs/                  # Documentation
└── agent-venv/           # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows/Linux/macOS
- 8GB RAM recommended
- Internet connection for ArXiv access

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SyntAI
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv agent-venv

# Activate virtual environment
# On Windows:
agent-venv\Scripts\activate
# On Linux/macOS:
source agent-venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Groq API Key (required for LLM features)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Other configurations
PYTHONPATH=.
```

**Getting a Groq API Key:**
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Create a new API key
4. Copy the key to your `.env` file

### 5. Run the Application

```bash
# Start the FastAPI server
python main.py
```

The server will start on `http://localhost:8000`

### 6. Access the Web Interface

Open your browser and navigate to:
- **Main Interface**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`

## 📋 Detailed Setup Guide

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection

**Recommended Requirements:**
- Python 3.10+
- 8GB RAM
- 5GB free disk space
- SSD storage for better performance

### Dependencies Overview

**Core Dependencies:**
- `fastapi`: Web framework and API server
- `uvicorn`: ASGI server for FastAPI
- `groq`: LLM API client for text generation
- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Text embeddings
- `PyMuPDF`: PDF processing and text extraction
- `pandas`: Data manipulation and analysis
- `matplotlib/seaborn`: Data visualization

**Optional Dependencies:**
- `torch`: Deep learning framework (for advanced features)
- `transformers`: Hugging Face transformer models
- `opencv-python`: Computer vision for image processing
- `plotly`: Interactive visualizations

### Installation Troubleshooting

**Common Issues:**

1. **PyTorch Installation on Windows:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **FAISS Installation Issues:**
   ```bash
   pip install faiss-cpu --no-cache-dir
   ```

3. **PDF Processing Issues:**
   ```bash
   pip install PyMuPDF pdfplumber pdf2image
   ```

4. **Memory Issues:**
   - Reduce batch sizes in configuration
   - Use CPU-only versions of libraries
   - Consider using a machine with more RAM

### Configuration Options

The application can be configured through environment variables:

```env
# API Configuration
GROQ_API_KEY=your_api_key
GROQ_MODEL=qwen/qwen3-32b  # Default LLM model

# Server Configuration
HOST=localhost
PORT=8000
DEBUG=False

# Storage Configuration
DATA_FOLDER=data
TEMP_DIR=temp

# RAG Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_DOCS=5

# Analysis Configuration
MAX_PAPERS_COMPARISON=10
EXPORT_FORMATS=json,csv,html
```

## 💡 Usage Guide

### 1. Searching for Papers

**Domain/Category Search:**
1. Select "Search by Domain" tab
2. Enter ArXiv category (e.g., "cs.AI", "cs.LG")
3. Set maximum results (1-50)
4. Click "Search Papers"

**Keyword Search:**
1. Select "Search by Keywords" tab
2. Enter relevant keywords
3. Configure time range and sorting options
4. Click "Search Papers"

**URL Import:**
1. Select "Import from URL" tab
2. Paste ArXiv paper URL
3. Click "Import Paper"

### 2. Paper Analysis

**Preview Papers:**
1. Click "Preview" on any downloaded paper
2. Navigate through pages using controls
3. Use "Analyze Page" for AI insights

**Generate Summaries:**
- Papers are automatically summarized during download
- View summaries in the paper cards

### 3. Chat with Papers

**Start Chatting:**
1. Click "Chat with Papers" button
2. Wait for RAG system initialization
3. Ask questions about your papers
4. View sources for each response

**Effective Chat Tips:**
- Be specific in your questions
- Reference paper titles or concepts
- Ask for comparisons between papers
- Request explanations of methodologies

### 4. Comparative Analysis

**Multi-Paper Comparison:**
1. Select papers using checkboxes
2. Click "Compare Selected Papers"
3. Wait for analysis completion
4. Review comparison table and visualizations
5. Download results in various formats

**Analysis Features:**
- Methodology comparison
- Results comparison
- Strengths and limitations analysis
- Visual charts and graphs

## 🔧 Advanced Configuration

### Custom Models

**Change LLM Model:**
```python
# In service/llm.py
DEFAULT_MODEL = "groq/llama2-70b-4096"  # Change this line
```

**Custom Embedding Model:**
```python
# In service/rag_service.py
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # Higher quality, slower
```

### Performance Optimization

**For Low-Memory Systems:**
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
MAX_RETRIEVAL_DOCS=3
```

**For High-Performance Systems:**
```env
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=2000
MAX_RETRIEVAL_DOCS=10
```

### Custom Data Storage

```python
# Change data folder location
DATA_FOLDER = "/path/to/your/data/folder"
```

## 🛠️ API Reference

### Core Endpoints

**Paper Search:**
```
POST /api/search
POST /api/search/keywords
POST /api/import/url
```

**Paper Management:**
```
GET /api/papers
GET /api/paper/{index}/info
GET /api/paper/{index}/page/{page}
DELETE /api/clear
```

**Chat System:**
```
GET /api/chat/status
POST /api/chat
```

**Comparative Analysis:**
```
POST /api/compare
GET /api/compare/test
```

### Response Formats

**Paper Object:**
```json
{
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Paper abstract...",
  "arxiv_id": "2024.0001",
  "arxiv_url": "https://arxiv.org/abs/2024.0001",
  "pdf_url": "https://arxiv.org/pdf/2024.0001.pdf",
  "published_date": "2024-01-01",
  "categories": ["cs.AI"],
  "downloaded": true,
  "summary": "AI-generated summary..."
}
```

## 🧪 Development

### Project Structure

**Backend Services:**
- `main.py`: FastAPI application and routing
- `agents/research_agent.py`: ArXiv scraping and paper discovery
- `service/llm.py`: Language model integrations
- `service/rag_service.py`: RAG implementation
- `service/pdf_parser.py`: PDF processing
- `service/comparative_analysis_service.py`: Multi-paper analysis

**Frontend:**
- `frontend/index.html`: Main application interface
- `frontend/app.js`: JavaScript application logic
- `frontend/comparative_analysis.html`: Comparison interface

### Adding New Features

**New LLM Provider:**
1. Add provider client in `service/llm.py`
2. Implement provider-specific methods
3. Update configuration options

**New Search Source:**
1. Create new agent in `agents/`
2. Implement standard paper object format
3. Integrate with main API endpoints

**New Analysis Features:**
1. Extend `ComparativeAnalysisService`
2. Add new visualization types
3. Update frontend interface

### Testing

```bash
# Run basic functionality tests
python -m pytest tests/

# Test specific components
python -c "from service.rag_service import get_rag_service; print('RAG OK')"
python -c "from service.llm import test_groq_connection; test_groq_connection()"
```

## 🔍 Troubleshooting

### Common Issues

**1. Server Won't Start:**
```bash
# Check port availability
netstat -ano | findstr :8000

# Try different port
uvicorn main:app --port 8001
```

**2. Papers Not Downloading:**
- Check internet connection
- Verify ArXiv accessibility
- Check disk space in data folder

**3. Chat Not Working:**
- Verify GROQ_API_KEY in .env file
- Check API key validity
- Monitor server logs for errors

**4. RAG System Issues:**
- Check FAISS installation
- Verify sentence-transformers installation
- Ensure sufficient memory (4GB+)

**5. PDF Preview Issues:**
- Verify PyMuPDF installation
- Check PDF file integrity
- Ensure proper file permissions

### Performance Issues

**Slow Search:**
- Reduce max_results parameter
- Check internet connection speed
- Consider local ArXiv mirror

**High Memory Usage:**
- Reduce embedding model size
- Decrease chunk size
- Limit concurrent operations

**Slow Chat Responses:**
- Check Groq API limits
- Reduce context length
- Optimize retrieval parameters

### Log Analysis

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Server Logs:**
- Monitor console output for errors
- Check temporary file creation
- Verify API call success

## 📊 Usage Examples

### Example 1: Research Literature Review

```python
# 1. Search for papers on a specific topic
search_query = "transformer architecture deep learning"

# 2. Import multiple papers
papers = await search_papers(search_query, max_results=20)

# 3. Chat with papers to understand concepts
chat_response = await chat("What are the main innovations in transformer architectures?")

# 4. Compare approaches across papers
comparison = await compare_papers(selected_paper_indices=[1, 3, 5, 8])
```

### Example 2: Methodology Comparison

```python
# 1. Search for papers on specific methodology
papers = await search_papers("attention mechanism neural networks")

# 2. Analyze methodologies
for paper in papers:
    summary = await analyze_paper(paper)
    
# 3. Generate comparative analysis
comparison_report = await generate_comparison_report(papers)
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/SyntAI.git

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ArXiv**: For providing open access to research papers
- **Groq**: For fast LLM inference capabilities
- **FAISS**: For efficient similarity search
- **Hugging Face**: For transformer models and embeddings
- **FastAPI**: For the excellent web framework
- **Tailwind CSS**: For beautiful UI components

## 📞 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub Issues
- **Questions**: Start a GitHub Discussion
- **Email**: [Your contact email]

## 🗺️ Roadmap

**Upcoming Features:**
- [ ] Support for more academic databases (IEEE, PubMed)
- [ ] Advanced visualization options
- [ ] Collaboration features for research teams
- [ ] Mobile-responsive interface improvements
- [ ] Integration with reference managers (Zotero, Mendeley)
- [ ] Custom AI model fine-tuning
- [ ] Batch paper processing capabilities
- [ ] Advanced search filters and sorting options

---

**Built with ❤️ for the research community**

*SyntAI - Making research literature accessible through AI*
