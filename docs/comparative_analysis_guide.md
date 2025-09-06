# Multi-Paper Comparative Analysis Feature

This feature enables comprehensive comparison of multiple research papers using open-source tools. It extracts structured information from PDFs and generates comparative analysis with visualizations.

## Features

### 1. PDF Text and Structure Extraction
- **PyMuPDF**: Extracts text and identifies paper sections (abstract, introduction, methodology, etc.)
- **tabula-py**: Extracts tables as pandas DataFrames
- Automatic metadata extraction from PDF documents

### 2. Structured Profile Generation
- **Groq LLM (Llama 3)**: Analyzes extracted content to generate structured JSON profiles
- Extracted fields include:
  - Title and contributions
  - Architecture description
  - Model parameters (millions)
  - FLOPs (giga operations)
  - Datasets used
  - Key results and metrics
  - Limitations

### 3. Comparative Analysis
- Side-by-side comparison tables (HTML/Markdown/CSV)
- Parameter and complexity analysis
- Dataset overlap detection
- Natural language summary generation

### 4. Visualizations
- **Matplotlib/Seaborn**: Model size comparisons
- Dataset usage frequency charts
- Performance metrics scatter plots
- Computational complexity comparisons

### 5. Export Options
- JSON for programmatic access
- Markdown reports for documentation
- CSV tables for spreadsheet analysis
- PNG visualizations

## API Endpoints

### 1. Compare Selected Papers
```http
POST /api/analysis/compare
Content-Type: application/json

{
    "paper_indices": [0, 1, 2],
    "output_dir": "analysis_output"
}
```

Compares papers from the current papers list by their indices.

### 2. Compare PDF Files
```http
POST /api/analysis/compare-files
Content-Type: application/json

{
    "pdf_paths": [
        "C:/path/to/paper1.pdf",
        "C:/path/to/paper2.pdf",
        "C:/path/to/paper3.pdf"
    ],
    "output_dir": "analysis_output"
}
```

Compares papers from provided PDF file paths.

### 3. Get Analysis Status
```http
GET /api/analysis/status
```

Returns the status of the comparative analysis service.

### 4. Download Results
```http
GET /api/analysis/download/{output_dir}/{file_type}
```

Download specific result files:
- `json`: Complete analysis results
- `markdown`: Human-readable report
- `csv`: Comparison table
- `metrics_plot`: Model metrics visualization
- `dataset_plot`: Dataset usage chart

## Usage Examples

### Python API Usage

```python
from service.comparative_analysis_service import run_comparative_analysis

# Example 1: Compare local PDF files
pdf_paths = [
    "papers/transformer_paper.pdf",
    "papers/bert_paper.pdf", 
    "papers/gpt_paper.pdf"
]

results = run_comparative_analysis(pdf_paths, "my_analysis")

if results.get("success"):
    print(f"Analyzed {results['paper_count']} papers")
    print("Summary:", results['summary'])
    print("Export paths:", results['export_paths'])
else:
    print("Error:", results.get('error'))
```

### Using the Service Class

```python
from service.comparative_analysis_service import ComparativeAnalysisService

service = ComparativeAnalysisService()

# Process individual paper
profile = service._process_single_paper("paper.pdf")

# Analyze multiple papers
results = service.analyze_papers(["paper1.pdf", "paper2.pdf"])

# Export results
export_paths = service.export_results(results, "output_folder")
```

### Frontend Usage

1. Open `http://localhost:8000/comparative_analysis.html`
2. Select papers from the current list OR enter PDF file paths
3. Click "Compare" to start analysis
4. View results with summary, tables, and visualizations
5. Download results in various formats

## Required Dependencies

The following packages are required (install manually):

```bash
pip install tabula-py pandas matplotlib seaborn
```

Already installed in your environment:
- PyMuPDF (fitz)
- groq
- pathlib
- json

## Output Structure

Analysis results are saved in the specified output directory:

```
analysis_output/
├── comparative_analysis.json      # Complete results
├── analysis_report.md            # Markdown report
├── comparison_table.csv          # Data table
├── metrics_comparison.png        # Model metrics plot
└── dataset_usage.png            # Dataset frequency chart
```

## JSON Profile Schema

Each paper is analyzed and structured as:

```json
{
    "title": "Paper Title",
    "contributions": [
        "Main contribution 1",
        "Main contribution 2"
    ],
    "architecture": "Brief architecture description",
    "params_million": 175000,
    "flops_g": 3140,
    "datasets": ["CommonCrawl", "WebText", "Books"],
    "results": {
        "perplexity": "20.0",
        "accuracy": "89.5%"
    },
    "limitations": [
        "High computational cost",
        "Limited multilingual support"
    ],
    "source_file": "/path/to/paper.pdf",
    "processed_at": "2025-08-30T10:30:00"
}
```

## Error Handling

The service includes comprehensive error handling:

- PDF parsing errors
- LLM service unavailability  
- Missing or invalid files
- JSON parsing failures
- Visualization generation errors

All errors are logged and returned in API responses.

## Performance Considerations

- **Caching**: Processed paper profiles are cached to avoid reprocessing
- **Parallel Processing**: Multiple papers can be processed concurrently
- **Memory Management**: Large PDFs are processed in chunks
- **Timeouts**: LLM requests have appropriate timeouts

## Customization

### Adding New Metrics
Modify the profile extraction prompt in `PaperProfileExtractor.generate_paper_profile()` to include additional fields.

### Custom Visualizations
Add new visualization functions in `ComparativeAnalysisService._create_visualizations()`.

### Different LLM Models
Change the model in `PaperProfileExtractor.__init__()` to use different Groq models.

## Troubleshooting

### Common Issues

1. **"Groq service not available"**
   - Check GROQ_API_KEY in environment variables
   - Verify internet connection

2. **"PDF not found"**
   - Ensure PDF files exist at specified paths
   - Check file permissions

3. **"No tables found"**
   - Some PDFs may not have extractable tables
   - tabula-py works best with well-formatted tables

4. **"JSON parsing failed"**
   - LLM response may be malformed
   - Service will return fallback profile with error flag

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check service status via API:
```bash
curl http://localhost:8000/api/analysis/status
```

## Future Enhancements

Potential improvements for the feature:

1. **Enhanced Section Detection**: Better algorithms for identifying paper sections
2. **Citation Analysis**: Extract and analyze citation networks
3. **Multi-language Support**: Support for non-English papers
4. **Real-time Collaboration**: Multiple users comparing papers simultaneously
5. **Advanced Visualizations**: Interactive plots with Plotly
6. **Batch Processing**: Queue system for large-scale comparisons
7. **Export to LaTeX**: Generate LaTeX comparison tables
8. **Semantic Similarity**: Compare papers based on content similarity

## License

This feature uses open-source tools and is part of the SyntAI project.
