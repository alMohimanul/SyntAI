# PaperWhisperer - ArXiv Research Assistant

A Python application that uses LangGraph workflows to orchestrate research agents for scraping papers from arXiv, extracting methodology information, and automatically downloading PDFs.

## 🚀 Features

- � **LangGraph Workflows** - Orchestrated multi-agent system
- �🔍 **Smart Search** - Search arXiv by domain/category (cs.AI, cs.CV, cs.LG, etc.)
- 📄 **Metadata Extraction** - Extract paper metadata (title, authors, abstract, categories)
- 🧠 **Methodology Detection** - Automatically identify methodology from abstracts
- 📥 **PDF Downloads** - Download PDFs to local data folder
- 💾 **Data Persistence** - Save all metadata to JSON file
- 📊 **Comprehensive Reports** - Print detailed summaries
- 🔄 **Batch Processing** - Process multiple domains at once
- 🎯 **Interactive Mode** - User-friendly CLI interface

## 🏗️ Architecture

```
PaperWhisperer/
├── main.py                 # LangGraph workflow orchestration
├── Agents/                 # Agent implementations
│   ├── __init__.py
│   └── research_agent.py   # ArXiv scraping agent
├── data/                   # Downloaded PDFs and metadata
├── requirements.txt
└── README.md
```

### LangGraph Workflow

The application uses LangGraph to orchestrate the following workflow:

1. **Collect User Input** - Get research parameters
2. **Search Papers** - Find papers using research agent
3. **Download Papers** - Download PDFs (if requested)
4. **Save Metadata** - Store structured data
5. **Generate Summary** - Display results

## 📦 Installation

1. Make sure you have Python 3.7+ installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Interactive Mode (Default)
```bash
python main.py
```

Follow the prompts to:
- Select arXiv domain
- Choose number of papers
- Decide on PDF downloads

### Batch Mode
```bash
python main.py
# Select option 2 for batch mode
# Enter multiple domains: cs.AI, cs.CV, cs.LG
```

### Programmatic Usage
```python
from main import PaperWhispererWorkflow, WorkflowState

# Create workflow
workflow = PaperWhispererWorkflow()

# Set initial state
initial_state = WorkflowState(
    domain="cs.AI",
    max_results=20,
    download_pdfs=True,
    papers=[],
    metadata_saved=False,
    workflow_complete=False,
    error_message="",
    user_input={}
)

# Run workflow
final_state = workflow.run_workflow(initial_state)
```

## 🎨 Agent Architecture

### Research Agent (`research_agent.py`)
- **Purpose**: ArXiv paper scraping and metadata extraction
- **Key Methods**:
  - `search_papers()` - Search arXiv by domain
  - `download_pdfs()` - Download paper PDFs
  - `extract_methodology()` - Extract methodology from abstracts
  - `save_metadata()` - Save data to JSON

### Adding New Agents
1. Create new agent file in `Agents/` directory
2. Follow the agent pattern (class-based, no main function)
3. Add to workflow in `main.py`
4. Update `Agents/__init__.py`

## 📊 Common ArXiv Domains

| Code | Description |
|------|-------------|
| `cs.AI` | Artificial Intelligence |
| `cs.CV` | Computer Vision and Pattern Recognition |
| `cs.LG` | Machine Learning |
| `cs.CL` | Computation and Language |
| `cs.NE` | Neural and Evolutionary Computing |
| `cs.RO` | Robotics |
| `cs.CR` | Cryptography and Security |
| `stat.ML` | Machine Learning (Statistics) |
| `math.CO` | Combinatorics |
| `physics.data-an` | Data Analysis, Statistics and Probability |

## 📁 Output Structure

### Data Folder
```
data/
├── arxiv_papers_20250804_143022.json  # Metadata file
├── 2401.12345_Deep_Learning_Approach.pdf
├── 2401.12346_Novel_Algorithm_for.pdf
└── ...
```

### Metadata JSON Structure
```json
{
  "search_timestamp": "2025-08-04T14:30:22",
  "total_papers": 10,
  "papers": [
    {
      "arxiv_id": "2401.12345",
      "title": "A Deep Learning Approach to...",
      "authors": ["John Doe", "Jane Smith"],
      "abstract": "Full abstract text...",
      "methodology": [
        "We propose a novel neural network architecture...",
        "The training procedure involves..."
      ],
      "categories": ["cs.CV", "cs.LG"],
      "published_date": "2024-01-23T18:30:00Z",
      "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
      "downloaded": true,
      "pdf_filename": "2401.12345_Deep_Learning_Approach.pdf"
    }
  ]
}
```

## 🧠 Methodology Extraction

The system automatically identifies methodology-related information from abstracts by looking for sentences containing keywords like:
- method, approach, algorithm, technique
- framework, model, architecture, system
- implementation, protocol, procedure
- strategy, solution, mechanism, pipeline
- network, learning, training, optimization

## ⚡ Workflow Features

- **Error Handling**: Robust error handling at each workflow step
- **State Management**: Comprehensive state tracking through workflow
- **Rate Limiting**: Respectful API usage with built-in delays
- **Progress Feedback**: Real-time progress updates with emojis
- **Flexible Configuration**: Support for different search parameters

## 🛠️ Requirements

- Python 3.7+
- requests
- beautifulsoup4
- feedparser
- lxml
- urllib3
- langgraph
- langchain-core
- typing-extensions

## 🔄 Extending the System

The LangGraph architecture makes it easy to:
- Add new agent types
- Modify workflow steps
- Add conditional branching
- Implement parallel processing
- Add new data sources

Each agent is independent and can be reused in different workflows.
