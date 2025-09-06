"""
Multi-Paper Comparative Analysis Service
Analyzes and compares multiple research papers using open-source tools.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# PDF parsing imports
import fitz  # PyMuPDF
import tabula
from tabulate import tabulate

# LLM imports
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperProfileExtractor:
    """Extracts structured profiles from research papers using Groq LLM"""
    
    def __init__(self):
        """Initialize the paper profile extractor"""
        self.client = None
        # Try a known working model first for testing
        self.model = "qwen/qwen3-32b"  # Known working model
        # self.model = "openai/gpt-oss-120b"  # Original model - might be the issue
        api_key = os.getenv("GROQ_API_KEY")
        
        # Debug API key loading
        print(f"🔑 API Key loaded: {bool(api_key)}")
        if api_key:
            print(f"🔑 API Key length: {len(api_key)}")
            print(f"🔑 API Key prefix: {api_key[:10]}...")
            print(f"🔑 API Key suffix: ...{api_key[-10:]}")
        else:
            print("❌ No API key found")

        if not api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            return
            
        try:
            self.client = Groq(api_key=api_key)
            logger.info("✅ Groq client initialized successfully")
            print("🤖 Testing client with a simple call...")
            
            # Test the client immediately
            test_response = self.client.chat.completions.create(
                model="qwen/qwen3-32b",  # Use a known working model
                messages=[{"role": "user", "content": "Say 'test' only"}],
                max_tokens=10
            )
            print("✅ Client test successful!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Groq client: {e}")
            print(f"❌ Full error details: {e}")
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.client is not None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and structure from PDF using PyMuPDF
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            print(f"\n📄 Starting PDF extraction for: {pdf_path}")
            doc = fitz.open(pdf_path)
            extracted_data = {
                "full_text": "",
                "sections": {},
                "metadata": {},
                "page_count": len(doc)
            }
            
            # Extract metadata
            extracted_data["metadata"] = doc.metadata
            print(f"📊 PDF Metadata: {doc.metadata}")
            print(f"📖 Total pages: {len(doc)}")
            
            # Extract text by pages and try to identify sections
            total_text_length = 0
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_text_length = len(page_text)
                total_text_length += page_text_length
                extracted_data["full_text"] += page_text + "\n"
                
                print(f"📄 Page {page_num + 1}: Extracted {page_text_length} characters")
                if page_text_length > 0:
                    # Show first 200 chars of each page
                    preview = page_text[:200].replace('\n', ' ').strip()
                    print(f"   Preview: {preview}...")
                
                # Simple section detection based on common patterns
                lines = page_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        # Check for common section headers
                        if any(header in line.lower() for header in 
                               ['abstract', 'introduction', 'methodology', 'results', 
                                'conclusion', 'related work', 'experiments', 'evaluation']):
                            if len(line) < 100 and line.isupper() or line.istitle():
                                section_name = line.lower().replace(' ', '_')
                                if section_name not in extracted_data["sections"]:
                                    extracted_data["sections"][section_name] = ""
                                    print(f"📋 Found section: {section_name}")
            
            doc.close()
            print(f"✅ PDF extraction completed:")
            print(f"   📊 Total text length: {total_text_length} characters")
            print(f"   📋 Sections found: {list(extracted_data['sections'].keys())}")
            print(f"   📄 First 500 characters of full text:")
            print(f"   {extracted_data['full_text'][:500]}")
            
            if total_text_length < 100:
                print("⚠️  WARNING: Very little text extracted - PDF might be image-based or corrupted")
            
            logger.info(f"Successfully extracted {total_text_length} characters from {pdf_path}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return {"error": str(e)}
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables from PDF using tabula-py
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List of pandas DataFrames containing extracted tables
        """
        try:
            # Extract all tables from the PDF with error handling
            tables = tabula.read_pdf(
                pdf_path, 
                pages='all', 
                multiple_tables=True,
                encoding='utf-8',
                silent=True  # Suppress warnings
            )
            logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
            return tables
        except UnicodeDecodeError as e:
            logger.warning(f"UTF-8 encoding error extracting tables from PDF {pdf_path}: {e}")
            # Try with different encoding or skip tables
            try:
                tables = tabula.read_pdf(
                    pdf_path, 
                    pages='all', 
                    multiple_tables=True,
                    encoding='latin-1',
                    silent=True
                )
                logger.info(f"Extracted {len(tables)} tables from {pdf_path} with latin-1 encoding")
                return tables
            except Exception as e2:
                logger.warning(f"Failed to extract tables with alternative encoding: {e2}")
                return []
        except Exception as e:
            logger.error(f"Error extracting tables from PDF {pdf_path}: {e}")
            return []
    
    def generate_paper_profile(self, text_data: Dict[str, Any], tables: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate structured paper profile using Groq LLM
        
        Args:
            text_data (Dict): Extracted text and sections
            tables (List[pd.DataFrame]): Extracted tables
            
        Returns:
            Dict containing structured paper profile
        """
        if not self.is_available():
            return {"error": "Groq service not available"}
        
        # Prepare table summaries
        table_summaries = []
        for i, table in enumerate(tables):
            if not table.empty:
                table_summary = f"Table {i+1}:\n{table.to_string()[:500]}..."
                table_summaries.append(table_summary)
        
        tables_text = "\n\n".join(table_summaries) if table_summaries else "No tables found"
        
        # Create prompt for structured extraction
        prompt = f"""
        Analyze the following research paper and extract comprehensive structured information in valid JSON format.

        Paper Text (first 10000 characters):
        {text_data.get('full_text', '')[:10000]}

        Tables Found:
        {tables_text}

        Extract and return ONLY a valid JSON object with this exact structure:
        {{
            "title": "exact paper title here",
            "contributions": [
                "specific contribution 1 with technical details",
                "specific contribution 2 with technical details",
                "specific contribution 3 with technical details"
            ],
            "architecture": "detailed description of the proposed method, architecture, or approach including key technical components",
            "params_million": null,
            "flops_g": null,
            "datasets": ["Dataset1", "Dataset2", "Dataset3"],
            "results": {{
                "primary_metric_name": "value with unit",
                "secondary_metric_name": "value with unit",
                "dataset1_performance": "best result on dataset1",
                "dataset2_performance": "best result on dataset2"
            }},
            "limitations": [
                "specific limitation 1 with context",
                "specific limitation 2 with context",
                "specific limitation 3 with context"
            ],
            "key_innovations": [
                "main innovation 1",
                "main innovation 2"
            ],
            "comparison_baselines": ["Baseline1", "Baseline2", "Baseline3"],
            "evaluation_metrics": ["metric1", "metric2", "metric3"]
        }}

        EXTRACTION GUIDELINES:
        1. **Title**: Extract the exact paper title as written
        2. **Contributions**: Extract 3-5 specific technical contributions, not generic statements
        3. **Architecture**: Provide detailed technical description of the proposed method
        4. **Parameters**: Look for model size in millions (convert billions to millions: 1.5B = 1500M)
        5. **FLOPs**: Look for computational complexity in gigaFLOPs or similar units
        6. **Datasets**: Extract exact dataset names used for training/evaluation
        7. **Results**: Extract specific numerical results with their context and units
        8. **Limitations**: Extract acknowledged limitations from the paper
        9. **Innovations**: Identify the main novel aspects introduced
        10. **Baselines**: List methods compared against
        11. **Metrics**: List evaluation metrics used

        IMPORTANT RULES:
        - Return ONLY the JSON object, no additional text
        - Use null for numeric fields if values are not found
        - Use empty arrays [] for missing list fields
        - Use double quotes for all strings
        - Escape any quotes within string values
        - Be specific and detailed rather than generic
        - Extract actual values, not approximations when possible
        - Include units in result values (e.g., "95.2% mIoU", "0.023 seconds")
        """
        
        try:
            print(f"\n🤖 Generating paper profile using model: {self.model}")
            full_text = text_data.get('full_text', '')
            print(f"📝 Text length being sent to LLM: {len(full_text)} characters")
            print(f"📄 Text preview (first 300 chars): {full_text[:300]}")
            print(f"📊 Number of tables found: {len(tables)}")
            
            # Debug API call
            print(f"🔧 Making API call with model: {self.model}")
            print(f"🔧 Client status: {self.client is not None}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2048
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"✅ Raw LLM response received ({len(response_text)} chars)")
            print(f"📝 Raw response preview: {response_text[:300]}...")
            
            # Try to parse as JSON
            try:
                profile = json.loads(response_text)
                print("✅ Successfully parsed JSON profile")
                print(f"📊 Profile keys: {list(profile.keys())}")
                print(f"📄 Title: {profile.get('title', 'N/A')}")
                logger.info("Successfully generated paper profile")
                return profile
            except json.JSONDecodeError as json_error:
                print(f"⚠️ Initial JSON parsing failed: {json_error}")
                logger.warning(f"Initial JSON parsing failed: {json_error}")
                # If direct parsing fails, try to extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_text = response_text[json_start:json_end]
                    print(f"🔧 Attempting to extract JSON from position {json_start} to {json_end}")
                    print(f"📝 Extracted JSON preview: {json_text[:200]}...")
                    logger.info(f"Extracted JSON: {json_text[:200]}...")
                    try:
                        profile = json.loads(json_text)
                        print("✅ Successfully extracted and parsed paper profile")
                        logger.info("Successfully extracted and parsed paper profile")
                        return profile
                    except json.JSONDecodeError as json_error2:
                        print(f"❌ JSON extraction also failed: {json_error2}")
                        logger.error(f"JSON extraction also failed: {json_error2}")
                        # Try to fix common JSON issues
                        cleaned_json = self._clean_json_response(json_text)
                        try:
                            profile = json.loads(cleaned_json)
                            print("✅ Successfully parsed cleaned JSON")
                            logger.info("Successfully parsed cleaned JSON")
                            return profile
                        except json.JSONDecodeError as json_error3:
                            print(f"❌ Even cleaned JSON failed: {json_error3}")
                            logger.error(f"Even cleaned JSON failed: {json_error3}")
                            raise json_error3
                else:
                    print("❌ No valid JSON structure found in response")
                    raise json.JSONDecodeError("No valid JSON found", response_text, 0)
                    
        except Exception as e:
            print(f"❌ Error generating paper profile: {e}")
            logger.error(f"Error generating paper profile: {e}")
            return {
                "error": str(e),
                "title": "Unknown",
                "contributions": [],
                "architecture": "Unknown",
                "params_million": None,
                "flops_g": None,
                "datasets": [],
                "results": {},
                "limitations": []
            }
    
    def _clean_json_response(self, json_text: str) -> str:
        """Clean common JSON formatting issues from LLM responses"""
        try:
            # Remove any markdown formatting
            json_text = json_text.replace('```json', '').replace('```', '')
            
            # Fix common issues with quotes
            import re
            
            # Fix unescaped quotes in string values
            # This is a simple fix - for production you'd want more robust handling
            lines = json_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # If line contains a string value, ensure proper escaping
                if ':' in line and '"' in line:
                    # Simple cleanup - remove extra spaces and fix basic formatting
                    line = re.sub(r'\s+', ' ', line.strip())
                cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
        except Exception as e:
            logger.warning(f"JSON cleaning failed: {e}")
            return json_text


class ComparativeAnalysisService:
    """Main service for multi-paper comparative analysis"""
    
    def __init__(self):
        """Initialize the comparative analysis service"""
        self.extractor = PaperProfileExtractor()
        self.profiles_cache = {}
        
    def analyze_papers(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple papers and generate comparative analysis
        
        Args:
            pdf_paths (List[str]): List of paths to PDF files
            
        Returns:
            Dict containing complete analysis results
        """
        logger.info(f"Starting analysis of {len(pdf_paths)} papers")
        
        # Step 1: Extract profiles for each paper
        profiles = []
        failed_papers = []
        
        for i, pdf_path in enumerate(pdf_paths):
            try:
                logger.info(f"Processing paper {i+1}/{len(pdf_paths)}: {pdf_path}")
                profile = self._process_single_paper(pdf_path)
                if profile and "error" not in profile:
                    profiles.append(profile)
                    logger.info(f"Successfully processed: {profile.get('title', 'Unknown')}")
                else:
                    failed_papers.append(pdf_path)
                    logger.warning(f"Failed to process paper: {pdf_path}")
            except Exception as e:
                failed_papers.append(pdf_path)
                logger.error(f"Exception processing paper {pdf_path}: {e}")
        
        if len(profiles) < 1:
            return {
                "error": f"No papers could be processed successfully. Failed papers: {failed_papers}",
                "failed_papers": failed_papers
            }
        
        if len(profiles) < 2:
            return {
                "error": f"Need at least 2 successfully processed papers for comparison. Only got {len(profiles)}. Failed papers: {failed_papers}",
                "processed_papers": len(profiles),
                "failed_papers": failed_papers,
                "profiles": profiles
            }
        
        # Step 2: Generate comparative analysis
        comparison_results = self._generate_comparison(profiles)
        
        # Step 3: Create visualizations
        visualizations = self._create_visualizations(profiles)
        
        # Step 4: Generate summary
        summary = self._generate_summary(profiles, comparison_results)
        
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "paper_count": len(profiles),
            "profiles": profiles,
            "comparison": comparison_results,
            "visualizations": visualizations,
            "summary": summary
        }
        
        if failed_papers:
            result["failed_papers"] = failed_papers
            result["warnings"] = [f"Could not process {len(failed_papers)} papers: {failed_papers}"]
        
        # Clean the result of any NaN values before returning
        clean_result = self._clean_for_json(result)
        
        return clean_result
    
    def _process_single_paper(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single paper and extract its profile"""
        try:
            # Check cache first
            if pdf_path in self.profiles_cache:
                logger.info(f"Using cached profile for {pdf_path}")
                return self.profiles_cache[pdf_path]
            
            logger.info(f"Processing paper: {pdf_path}")
            
            # Extract text and structure
            text_data = self.extractor.extract_text_from_pdf(pdf_path)
            if "error" in text_data:
                return text_data
            
            # Extract tables
            tables = self.extractor.extract_tables_from_pdf(pdf_path)
            
            # Generate profile using LLM
            profile = self.extractor.generate_paper_profile(text_data, tables)
            
            # Add metadata
            profile["source_file"] = pdf_path
            profile["processed_at"] = datetime.now().isoformat()
            
            # Cache the result
            self.profiles_cache[pdf_path] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error processing paper {pdf_path}: {e}")
            return {"error": str(e)}
    
    def _generate_comparison(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate side-by-side comparison of papers"""
        try:
            # Import tabulate for better table formatting
            from tabulate import tabulate
            
            # Create DataFrame for easy comparison
            comparison_data = []
            
            for profile in profiles:
                # Truncate title for better display
                title = profile.get("title", "Unknown")
                if len(title) > 50:
                    title = title[:47] + "..."
                
                # Format architecture description
                arch = profile.get("architecture", "Unknown")
                if len(arch) > 60:
                    arch = arch[:57] + "..."
                
                # Format datasets
                datasets = profile.get("datasets", [])
                dataset_str = ", ".join(datasets[:3])  # Show first 3 datasets
                if len(datasets) > 3:
                    dataset_str += f" (+ {len(datasets) - 3} more)"
                
                row = {
                    "Title": title,
                    "Architecture": arch,
                    "Parameters (M)": profile.get("params_million") if profile.get("params_million") is not None else "N/A",
                    "FLOPs (G)": profile.get("flops_g") if profile.get("flops_g") is not None else "N/A",
                    "Datasets": dataset_str or "Not specified",
                    "Contributions": len(profile.get("contributions", [])),
                    "Limitations": len(profile.get("limitations", []))
                }
                
                # Add key results with better formatting
                results = profile.get("results", {})
                for key, value in results.items():
                    # Clean up result key names
                    clean_key = key.replace("_", " ").title()
                    row[clean_key] = value
                
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            
            # Clean DataFrame of any NaN values
            df = df.fillna("N/A")
            
            # Generate multiple table formats
            html_table = df.to_html(index=False, escape=False, classes="comparison-table table table-striped")
            
            # Use tabulate for better markdown formatting
            markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            
            # Create a detailed comparison table with more information
            detailed_comparison = self._create_detailed_comparison(profiles)
            
            return {
                "dataframe": df.to_dict('records'),
                "html_table": html_table,
                "markdown_table": markdown_table,
                "detailed_comparison": detailed_comparison,
                "comparison_metrics": self._calculate_comparison_metrics(profiles),
                "paper_summaries": self._create_paper_summaries(profiles)
            }
            
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return {"error": str(e)}
    
    def _create_detailed_comparison(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a more detailed comparison with full information"""
        try:
            from tabulate import tabulate
            
            detailed_data = []
            for i, profile in enumerate(profiles, 1):
                # Create a comprehensive comparison entry
                contributions_text = "\n".join([f"• {contrib}" for contrib in profile.get("contributions", [])[:3]])
                if len(profile.get("contributions", [])) > 3:
                    contributions_text += f"\n• ... and {len(profile.get('contributions', [])) - 3} more"
                
                limitations_text = "\n".join([f"• {limit}" for limit in profile.get("limitations", [])[:3]])
                if len(profile.get("limitations", [])) > 3:
                    limitations_text += f"\n• ... and {len(profile.get('limitations', [])) - 3} more"
                
                results_text = ""
                if profile.get("results"):
                    results_list = []
                    for key, value in profile.get("results", {}).items():
                        results_list.append(f"{key}: {value}")
                    results_text = "\n".join(results_list[:5])  # Show top 5 results
                
                detailed_data.append({
                    "Paper": f"Paper {i}",
                    "Title": profile.get("title", "Unknown"),
                    "Key Contributions": contributions_text or "Not specified",
                    "Architecture/Method": profile.get("architecture", "Not specified"),
                    "Model Size": f"{profile.get('params_million', 'N/A')} M parameters" if profile.get('params_million') else "Not specified",
                    "Computational Cost": f"{profile.get('flops_g', 'N/A')} GFLOPs" if profile.get('flops_g') else "Not specified",
                    "Datasets Used": ", ".join(profile.get("datasets", [])) or "Not specified",
                    "Key Results": results_text or "Not available",
                    "Limitations": limitations_text or "Not specified"
                })
            
            # Create formatted table
            table_headers = ["Aspect"] + [f"Paper {i+1}" for i in range(len(profiles))]
            
            # Transpose the data for better comparison view
            comparison_rows = []
            aspects = ["Title", "Key Contributions", "Architecture/Method", "Model Size", 
                      "Computational Cost", "Datasets Used", "Key Results", "Limitations"]
            
            for aspect in aspects:
                row = [aspect]
                for paper_data in detailed_data:
                    value = paper_data.get(aspect, "N/A")
                    # Limit cell content length for readability
                    if len(str(value)) > 200:
                        value = str(value)[:197] + "..."
                    row.append(value)
                comparison_rows.append(row)
            
            detailed_table = tabulate(comparison_rows, headers=table_headers, tablefmt='grid', maxcolwidths=[20, 40, 40])
            
            return {
                "table": detailed_table,
                "data": detailed_data,
                "summary": f"Detailed comparison of {len(profiles)} research papers across {len(aspects)} key aspects"
            }
            
        except Exception as e:
            logger.error(f"Error creating detailed comparison: {e}")
            return {"error": str(e)}
    
    def _create_paper_summaries(self, profiles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create individual summaries for each paper"""
        summaries = []
        
        for i, profile in enumerate(profiles, 1):
            # Create a comprehensive summary for each paper
            title = profile.get("title", "Unknown")
            
            # Format contributions
            contributions = profile.get("contributions", [])
            contrib_summary = f"This paper makes {len(contributions)} key contributions: " + "; ".join(contributions[:2])
            if len(contributions) > 2:
                contrib_summary += f" and {len(contributions) - 2} additional contributions"
            
            # Format technical details
            tech_details = []
            if profile.get("architecture"):
                tech_details.append(f"Architecture: {profile['architecture']}")
            if profile.get("params_million"):
                tech_details.append(f"Model size: {profile['params_million']}M parameters")
            if profile.get("flops_g"):
                tech_details.append(f"Computational cost: {profile['flops_g']}G FLOPs")
            
            tech_summary = ". ".join(tech_details) if tech_details else "Technical details not available"
            
            # Format datasets and results
            datasets = profile.get("datasets", [])
            dataset_summary = f"Evaluated on {len(datasets)} datasets: {', '.join(datasets[:3])}" if datasets else "Evaluation datasets not specified"
            if len(datasets) > 3:
                dataset_summary += f" and {len(datasets) - 3} others"
            
            # Format results
            results = profile.get("results", {})
            results_summary = ""
            if results:
                result_items = [f"{key}: {value}" for key, value in list(results.items())[:3]]
                results_summary = f"Key results include {', '.join(result_items)}"
                if len(results) > 3:
                    results_summary += f" among {len(results)} total metrics"
            else:
                results_summary = "Specific numerical results not available"
            
            # Format limitations
            limitations = profile.get("limitations", [])
            limit_summary = ""
            if limitations:
                limit_summary = f"The authors identify {len(limitations)} main limitations: {'; '.join(limitations[:2])}"
                if len(limitations) > 2:
                    limit_summary += f" and {len(limitations) - 2} additional concerns"
            else:
                limit_summary = "Limitations not explicitly stated"
            
            summary_text = f"{contrib_summary}. {tech_summary}. {dataset_summary}. {results_summary}. {limit_summary}."
            
            summaries.append({
                "paper_number": i,
                "title": title,
                "summary": summary_text,
                "key_strengths": "; ".join(contributions[:3]) if contributions else "Not specified",
                "key_limitations": "; ".join(limitations[:3]) if limitations else "Not specified"
            })
        
        return summaries
    
    def _calculate_comparison_metrics(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate various comparison metrics between papers"""
        metrics = {
            "parameter_comparison": {},
            "dataset_overlap": {},
            "architecture_similarity": {}
        }
        
        # Parameter comparison
        params = []
        for p in profiles:
            param_val = p.get("params_million")
            if param_val is not None and not (isinstance(param_val, float) and (param_val != param_val or param_val == float('inf') or param_val == float('-inf'))):
                params.append(param_val)
        
        if params:
            metrics["parameter_comparison"] = {
                "min": min(params),
                "max": max(params),
                "avg": sum(params) / len(params),
                "range": max(params) - min(params)
            }
        
        # Dataset overlap analysis
        all_datasets = []
        for profile in profiles:
            datasets = profile.get("datasets", [])
            all_datasets.extend(datasets)
        
        unique_datasets = list(set(all_datasets))
        dataset_counts = {dataset: all_datasets.count(dataset) for dataset in unique_datasets}
        
        metrics["dataset_overlap"] = {
            "total_unique_datasets": len(unique_datasets),
            "shared_datasets": [ds for ds, count in dataset_counts.items() if count > 1],
            "dataset_frequency": dataset_counts
        }
        
        return metrics
    
    def _create_visualizations(self, profiles: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create visualizations comparing the papers"""
        visualizations = {}
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Parameters vs Performance visualization (if data available)
            params = []
            titles = []
            flops = []
            
            for profile in profiles:
                param_val = profile.get("params_million")
                if param_val is not None and not (isinstance(param_val, float) and (param_val != param_val or param_val == float('inf') or param_val == float('-inf'))):
                    params.append(float(param_val))
                    titles.append(profile.get("title", "Unknown")[:20] + "...")
                    
                    flop_val = profile.get("flops_g")
                    if flop_val is not None and not (isinstance(flop_val, float) and (flop_val != flop_val or flop_val == float('inf') or flop_val == float('-inf'))):
                        flops.append(float(flop_val))
                    else:
                        flops.append(0)
            
            if len(params) > 1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Parameters comparison
                ax1.bar(range(len(params)), params, color='skyblue', alpha=0.7)
                ax1.set_xlabel('Papers')
                ax1.set_ylabel('Parameters (Millions)')
                ax1.set_title('Model Size Comparison')
                ax1.set_xticks(range(len(titles)))
                ax1.set_xticklabels(titles, rotation=45, ha='right')
                
                # FLOPs comparison (if available)
                if any(f > 0 for f in flops):
                    ax2.bar(range(len(flops)), flops, color='lightcoral', alpha=0.7)
                    ax2.set_xlabel('Papers')
                    ax2.set_ylabel('FLOPs (Giga)')
                    ax2.set_title('Computational Complexity Comparison')
                    ax2.set_xticks(range(len(titles)))
                    ax2.set_xticklabels(titles, rotation=45, ha='right')
                else:
                    ax2.text(0.5, 0.5, 'FLOPs data not available', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('FLOPs Comparison (No Data)')
                
                plt.tight_layout()
                
                # Save plot
                viz_path = "comparison_metrics.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualizations["metrics_comparison"] = viz_path
            
            # 2. Dataset usage visualization
            all_datasets = []
            for profile in profiles:
                datasets = profile.get("datasets", [])
                all_datasets.extend(datasets)
            
            if all_datasets:
                dataset_counts = pd.Series(all_datasets).value_counts()
                
                if len(dataset_counts) > 0:
                    plt.figure(figsize=(12, 8))
                    dataset_counts.head(10).plot(kind='barh', color='lightgreen', alpha=0.7)
                    plt.title('Most Frequently Used Datasets')
                    plt.xlabel('Number of Papers')
                    plt.ylabel('Datasets')
                    plt.tight_layout()
                    
                    viz_path = "dataset_usage.png"
                    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    visualizations["dataset_usage"] = viz_path
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            visualizations["error"] = str(e)
        
        return visualizations
    
    def _generate_summary(self, profiles: List[Dict[str, Any]], comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of the comparison"""
        try:
            # Create both text and structured summary
            summary_text_parts = []
            structured_summary = {}
            
            # Overview section
            summary_text_parts.append(f"## Comparative Analysis Summary")
            summary_text_parts.append(f"")
            summary_text_parts.append(f"Analyzed {len(profiles)} research papers:")
            
            paper_titles = []
            for i, profile in enumerate(profiles, 1):
                title = profile.get("title", "Unknown")
                paper_titles.append(title)
                summary_text_parts.append(f"{i}. {title}")
            
            structured_summary["overview"] = {
                "total_papers": len(profiles),
                "paper_titles": paper_titles,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Key findings section
            summary_text_parts.append("\n## Key Findings")
            summary_text_parts.append("")
            
            # Research contributions analysis
            all_contributions = []
            for profile in profiles:
                all_contributions.extend(profile.get("contributions", []))
            
            total_contributions = len(all_contributions)
            avg_contributions = total_contributions / len(profiles) if profiles else 0
            
            summary_text_parts.append("### Research Contributions:")
            summary_text_parts.append(f"• Total unique research contributions across all papers: **{total_contributions}**")
            summary_text_parts.append(f"• Average contributions per paper: **{avg_contributions:.1f}**")
            
            # Find most common contribution themes
            contribution_themes = self._analyze_contribution_themes(all_contributions)
            if contribution_themes:
                summary_text_parts.append("• Common research themes:")
                for theme, count in contribution_themes[:3]:
                    summary_text_parts.append(f"  - {theme} (mentioned {count} times)")
            
            # Technical analysis
            summary_text_parts.append("\n### Technical Characteristics:")
            
            # Parameter analysis
            metrics = comparison_results.get("comparison_metrics", {})
            param_comp = metrics.get("parameter_comparison", {})
            
            params_with_data = [p for p in profiles if p.get("params_million") is not None]
            
            if param_comp and params_with_data:
                summary_text_parts.append("**Model Complexity:**")
                summary_text_parts.append(f"• Parameter range: **{param_comp.get('min', 'N/A'):.1f}M** to **{param_comp.get('max', 'N/A'):.1f}M** parameters")
                summary_text_parts.append(f"• Average model size: **{param_comp.get('avg', 'N/A'):.1f}M** parameters")
                
                # Classify model sizes
                small_models = [p for p in params_with_data if p.get("params_million", 0) < 50]
                medium_models = [p for p in params_with_data if 50 <= p.get("params_million", 0) < 500]
                large_models = [p for p in params_with_data if p.get("params_million", 0) >= 500]
                
                summary_text_parts.append(f"• Model size distribution: {len(small_models)} small (<50M), {len(medium_models)} medium (50-500M), {len(large_models)} large (≥500M)")
            else:
                summary_text_parts.append("• Model parameter information not available for comparison")
            
            # Architecture analysis
            architectures = [p.get("architecture", "").lower() for p in profiles if p.get("architecture")]
            arch_summary = self._analyze_architectures(architectures)
            if arch_summary:
                summary_text_parts.append("**Architecture Analysis:**")
                summary_text_parts.extend([f"• {point}" for point in arch_summary])
            
            # Dataset analysis
            summary_text_parts.append("\n### Dataset and Evaluation:")
            dataset_info = metrics.get("dataset_overlap", {})
            
            if dataset_info:
                total_datasets = dataset_info.get("total_unique_datasets", 0)
                shared_datasets = dataset_info.get("shared_datasets", [])
                
                summary_text_parts.append(f"• Total unique datasets used: **{total_datasets}**")
                
                if shared_datasets:
                    summary_text_parts.append(f"• Commonly used datasets: **{', '.join(shared_datasets[:5])}**")
                    if len(shared_datasets) > 5:
                        summary_text_parts.append(f"  (and {len(shared_datasets) - 5} others)")
                else:
                    summary_text_parts.append("• No shared datasets identified across papers")
                
                # Dataset frequency analysis
                freq_data = dataset_info.get("dataset_frequency", {})
                if freq_data:
                    most_popular = max(freq_data.items(), key=lambda x: x[1])
                    summary_text_parts.append(f"• Most frequently used dataset: **{most_popular[0]}** (used in {most_popular[1]} papers)")
            
            # Performance and results analysis
            summary_text_parts.append("\n### Performance Analysis:")
            results_analysis = self._analyze_results(profiles)
            if results_analysis:
                summary_text_parts.extend([f"• {point}" for point in results_analysis])
            else:
                summary_text_parts.append("• Detailed performance metrics not available for comparison")
            
            # Limitations analysis
            summary_text_parts.append("\n### Research Limitations:")
            all_limitations = []
            for profile in profiles:
                all_limitations.extend(profile.get("limitations", []))
            
            total_limitations = len(all_limitations)
            avg_limitations = total_limitations / len(profiles) if profiles else 0
            
            summary_text_parts.append(f"• Total identified limitations: **{total_limitations}**")
            summary_text_parts.append(f"• Average limitations per paper: **{avg_limitations:.1f}**")
            
            limitation_themes = self._analyze_limitation_themes(all_limitations)
            if limitation_themes:
                summary_text_parts.append("• Common limitation categories:")
                for theme, count in limitation_themes[:3]:
                    summary_text_parts.append(f"  - {theme} (mentioned {count} times)")
            
            # Research gaps and opportunities
            summary_text_parts.append("\n### Research Gaps and Opportunities:")
            gaps = self._identify_research_gaps(profiles)
            if gaps:
                summary_text_parts.extend([f"• {gap}" for gap in gaps])
            
            # Structured summary for programmatic access
            structured_summary.update({
                "contributions": {
                    "total": total_contributions,
                    "average_per_paper": avg_contributions,
                    "themes": contribution_themes[:5] if contribution_themes else []
                },
                "technical": {
                    "parameter_analysis": param_comp,
                    "architecture_summary": arch_summary[:5] if arch_summary else [],
                    "models_with_params": len(params_with_data)
                },
                "datasets": {
                    "total_unique": dataset_info.get("total_unique_datasets", 0),
                    "shared": shared_datasets[:10] if shared_datasets else [],
                    "most_popular": dict(list(dataset_info.get("dataset_frequency", {}).items())[:5])
                },
                "limitations": {
                    "total": total_limitations,
                    "average_per_paper": avg_limitations,
                    "themes": limitation_themes[:5] if limitation_themes else []
                },
                "research_gaps": gaps[:5] if gaps else []
            })
            
            summary_text = "\n".join(summary_text_parts)
            
            return {
                "text": summary_text,
                "structured": structured_summary,
                "highlights": self._extract_key_highlights(profiles, comparison_results)
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "text": f"Error generating summary: {str(e)}",
                "structured": {"error": str(e)},
                "highlights": []
            }
    
    def _analyze_contribution_themes(self, contributions: List[str]) -> List[Tuple[str, int]]:
        """Analyze common themes in research contributions"""
        # Simple keyword-based theme analysis
        theme_keywords = {
            "Deep Learning/Neural Networks": ["neural", "deep", "network", "cnn", "transformer", "attention"],
            "Computer Vision": ["vision", "image", "visual", "detection", "segmentation", "classification"],
            "Natural Language Processing": ["language", "text", "nlp", "linguistic", "semantic"],
            "Performance Optimization": ["efficient", "fast", "optimization", "speed", "performance"],
            "Novel Architecture": ["novel", "new", "architecture", "design", "framework"],
            "Multi-modal Learning": ["multi", "multimodal", "fusion", "cross-modal"],
            "Transfer Learning": ["transfer", "pretrained", "fine-tuning", "adaptation"],
            "Attention Mechanisms": ["attention", "self-attention", "cross-attention", "focus"]
        }
        
        theme_counts = {}
        for contrib in contributions:
            contrib_lower = contrib.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in contrib_lower for keyword in keywords):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        return sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    
    def _analyze_architectures(self, architectures: List[str]) -> List[str]:
        """Analyze architectural patterns"""
        analysis = []
        
        # Count common architectural patterns
        patterns = {
            "transformer": ["transformer", "attention"],
            "cnn": ["cnn", "convolutional", "convolution"],
            "hybrid": ["hybrid", "combination", "combines"],
            "encoder-decoder": ["encoder", "decoder"],
            "u-net": ["unet", "u-net"],
            "resnet": ["resnet", "residual"],
            "vision transformer": ["vit", "vision transformer"]
        }
        
        pattern_counts = {}
        for arch in architectures:
            arch_lower = arch.lower()
            for pattern, keywords in patterns.items():
                if any(keyword in arch_lower for keyword in keywords):
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        if pattern_counts:
            most_common = max(pattern_counts.items(), key=lambda x: x[1])
            analysis.append(f"Most common architectural pattern: **{most_common[0]}** (found in {most_common[1]} papers)")
            
            if len(pattern_counts) > 1:
                analysis.append(f"Architecture diversity: {len(pattern_counts)} different patterns identified")
        
        return analysis
    
    def _analyze_results(self, profiles: List[Dict[str, Any]]) -> List[str]:
        """Analyze performance results across papers"""
        analysis = []
        
        # Collect all result metrics
        all_metrics = {}
        for profile in profiles:
            results = profile.get("results", {})
            for metric, value in results.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        if all_metrics:
            analysis.append(f"Performance metrics reported: **{len(all_metrics)}** different types")
            
            # Find most commonly reported metrics
            metric_counts = {metric: len(values) for metric, values in all_metrics.items()}
            most_common_metric = max(metric_counts.items(), key=lambda x: x[1])
            analysis.append(f"Most commonly reported metric: **{most_common_metric[0]}** (in {most_common_metric[1]} papers)")
        
        return analysis
    
    def _analyze_limitation_themes(self, limitations: List[str]) -> List[Tuple[str, int]]:
        """Analyze common themes in research limitations"""
        theme_keywords = {
            "Computational Resources": ["computational", "compute", "gpu", "memory", "expensive", "cost"],
            "Dataset Limitations": ["dataset", "data", "limited data", "small dataset", "annotation"],
            "Generalization": ["generalization", "generalize", "domain", "cross-domain", "transfer"],
            "Evaluation": ["evaluation", "benchmark", "metric", "comparison", "baseline"],
            "Scalability": ["scalability", "scale", "large-scale", "scalable"],
            "Real-time Performance": ["real-time", "inference", "speed", "latency", "runtime"],
            "Robustness": ["robust", "robustness", "noise", "adversarial", "stability"]
        }
        
        theme_counts = {}
        for limitation in limitations:
            limitation_lower = limitation.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in limitation_lower for keyword in keywords):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        return sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    
    def _identify_research_gaps(self, profiles: List[Dict[str, Any]]) -> List[str]:
        """Identify potential research gaps and opportunities"""
        gaps = []
        
        # Analyze what's missing or could be improved
        all_limitations = []
        for profile in profiles:
            all_limitations.extend(profile.get("limitations", []))
        
        # Simple gap identification based on common limitations
        if any("computational" in lim.lower() for lim in all_limitations):
            gaps.append("**Computational efficiency** remains a challenge across multiple approaches")
        
        if any("dataset" in lim.lower() or "data" in lim.lower() for lim in all_limitations):
            gaps.append("**Dataset quality and availability** is a recurring concern")
        
        if any("generalization" in lim.lower() for lim in all_limitations):
            gaps.append("**Cross-domain generalization** needs further investigation")
        
        # Check for technical gaps
        architectures = [p.get("architecture", "").lower() for p in profiles]
        if not any("interpretable" in arch or "explainable" in arch for arch in architectures):
            gaps.append("**Model interpretability and explainability** could be better addressed")
        
        return gaps
    
    def _extract_key_highlights(self, profiles: List[Dict[str, Any]], comparison_results: Dict[str, Any]) -> List[str]:
        """Extract key highlights from the analysis"""
        highlights = []
        
        # Best performing model (if we have results)
        best_param_model = None
        min_params = float('inf')
        
        for profile in profiles:
            params = profile.get("params_million")
            if params and params < min_params:
                min_params = params
                best_param_model = profile.get("title", "Unknown")
        
        if best_param_model:
            highlights.append(f"Most parameter-efficient model: {best_param_model} ({min_params}M parameters)")
        
        # Most comprehensive evaluation
        most_datasets = 0
        most_evaluated_paper = None
        
        for profile in profiles:
            dataset_count = len(profile.get("datasets", []))
            if dataset_count > most_datasets:
                most_datasets = dataset_count
                most_evaluated_paper = profile.get("title", "Unknown")
        
        if most_evaluated_paper and most_datasets > 0:
            highlights.append(f"Most comprehensive evaluation: {most_evaluated_paper} (tested on {most_datasets} datasets)")
        
        # Most novel contributions
        max_contributions = 0
        most_innovative_paper = None
        
        for profile in profiles:
            contrib_count = len(profile.get("contributions", []))
            if contrib_count > max_contributions:
                max_contributions = contrib_count
                most_innovative_paper = profile.get("title", "Unknown")
        
        if most_innovative_paper:
            highlights.append(f"Most innovative approach: {most_innovative_paper} ({max_contributions} key contributions)")
        
        return highlights
    
    def export_results(self, analysis_results: Dict[str, Any], output_dir: str = "analysis_output") -> Dict[str, str]:
        """
        Export analysis results to various formats
        
        Args:
            analysis_results (Dict): Results from analyze_papers()
            output_dir (str): Output directory path
            
        Returns:
            Dict containing paths to exported files
        """
        try:
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"{output_dir}_{timestamp}")
            output_path.mkdir(exist_ok=True)
            
            exported_files = {}
            
            # 1. Export comprehensive JSON (clean NaN values first)
            json_path = output_path / "comparative_analysis_full.json"
            clean_results = self._clean_for_json(analysis_results)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=2, default=str)
            exported_files["json"] = str(json_path)
            
            # 2. Export detailed HTML report
            html_path = output_path / "analysis_report.html"
            html_content = self._generate_html_report(analysis_results)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            exported_files["html"] = str(html_path)
            
            # 3. Export Markdown report
            markdown_path = output_path / "analysis_report.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write("# Multi-Paper Comparative Analysis Report\n\n")
                f.write(f"Generated on: {analysis_results.get('timestamp', 'Unknown')}\n\n")
                
                # Add summary
                summary = analysis_results.get('summary', {})
                if isinstance(summary, dict) and 'text' in summary:
                    f.write(summary['text'])
                else:
                    f.write(str(summary))
                
                f.write("\n\n## Side-by-Side Comparison\n\n")
                comparison = analysis_results.get('comparison', {})
                if 'markdown_table' in comparison:
                    f.write(comparison['markdown_table'])
                
                # Add detailed comparison
                if 'detailed_comparison' in comparison:
                    f.write("\n\n## Detailed Comparison\n\n")
                    detailed = comparison['detailed_comparison']
                    if isinstance(detailed, dict) and 'table' in detailed:
                        f.write("```\n")
                        f.write(detailed['table'])
                        f.write("\n```\n")
                
                # Add paper summaries
                if 'paper_summaries' in comparison:
                    f.write("\n\n## Individual Paper Summaries\n\n")
                    for summary in comparison['paper_summaries']:
                        f.write(f"### {summary.get('title', 'Unknown Paper')}\n\n")
                        f.write(f"{summary.get('summary', 'No summary available')}\n\n")
                        f.write(f"**Key Strengths:** {summary.get('key_strengths', 'Not specified')}\n\n")
                        f.write(f"**Key Limitations:** {summary.get('key_limitations', 'Not specified')}\n\n")
                
            exported_files["markdown"] = str(markdown_path)
            
            # 4. Export CSV comparison table
            comparison_data = analysis_results.get('comparison', {}).get('dataframe', [])
            if comparison_data:
                csv_path = output_path / "comparison_table.csv"
                df = pd.DataFrame(comparison_data)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                exported_files["csv"] = str(csv_path)
            
            # 5. Export individual paper profiles
            profiles_path = output_path / "paper_profiles.json"
            profiles = analysis_results.get('profiles', [])
            with open(profiles_path, 'w', encoding='utf-8') as f:
                json.dump(profiles, f, indent=2, default=str)
            exported_files["profiles"] = str(profiles_path)
            
            # 6. Copy visualizations if they exist
            visualizations = analysis_results.get('visualizations', {})
            for viz_name, viz_path in visualizations.items():
                if viz_name != "error" and os.path.exists(viz_path):
                    import shutil
                    dest_path = output_path / f"{viz_name}.png"
                    shutil.copy2(viz_path, dest_path)
                    exported_files[f"viz_{viz_name}"] = str(dest_path)
            
            logger.info(f"Results exported to {output_path}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return {"error": str(e)}
    
    def _generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive HTML report"""
        timestamp = analysis_results.get('timestamp', 'Unknown')
        paper_count = analysis_results.get('paper_count', 0)
        
        # Extract summary information
        summary = analysis_results.get('summary', {})
        if isinstance(summary, dict):
            summary_text = summary.get('text', 'No summary available')
            highlights = summary.get('highlights', [])
        else:
            summary_text = str(summary)
            highlights = []
        
        # Extract comparison data
        comparison = analysis_results.get('comparison', {})
        html_table = comparison.get('html_table', '<p>No comparison table available</p>')
        detailed_comparison = comparison.get('detailed_comparison', {})
        paper_summaries = comparison.get('paper_summaries', [])
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Paper Comparative Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #2c3e50;
        }}
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .meta-info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .section h3 {{
            color: #2c3e50;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        .highlights {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .highlights h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .highlights ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .highlights li {{
            margin-bottom: 8px;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        .comparison-table th,
        .comparison-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
            vertical-align: top;
        }}
        .comparison-table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        .comparison-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .comparison-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .paper-summary {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .paper-summary h4 {{
            color: #495057;
            margin-top: 0;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 10px;
        }}
        .paper-summary .summary-text {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        .paper-summary .strengths,
        .paper-summary .limitations {{
            margin-bottom: 10px;
        }}
        .paper-summary .strengths strong,
        .paper-summary .limitations strong {{
            color: #495057;
        }}
        .summary-content {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
            white-space: pre-line;
        }}
        .detailed-comparison {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .detailed-comparison pre {{
            background-color: white;
            padding: 15px;
            border-radius: 3px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.4;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
            font-size: 14px;
        }}
        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}
            .comparison-table {{
                font-size: 12px;
            }}
            .comparison-table th,
            .comparison-table td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Multi-Paper Comparative Analysis Report</h1>
            <div class="meta-info">
                <strong>Generated:</strong> {timestamp}<br>
                <strong>Papers Analyzed:</strong> {paper_count}<br>
                <strong>Analysis Type:</strong> Comprehensive Multi-Paper Comparison
            </div>
        </div>
        
        {f'''
        <div class="highlights">
            <h3>📊 Key Highlights</h3>
            <ul>
                {"".join([f"<li>{highlight}</li>" for highlight in highlights])}
            </ul>
        </div>
        ''' if highlights else ''}
        
        <div class="section">
            <h2>📋 Executive Summary</h2>
            <div class="summary-content">
                {summary_text.replace('**', '<strong>').replace('**', '</strong>').replace('\n', '<br>')}
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Side-by-Side Comparison</h2>
            {html_table}
        </div>
        
        {f'''
        <div class="section">
            <h2>🔍 Detailed Comparison</h2>
            <div class="detailed-comparison">
                <p>{detailed_comparison.get('summary', 'Detailed comparison analysis')}</p>
                <pre>{detailed_comparison.get('table', 'No detailed comparison available')}</pre>
            </div>
        </div>
        ''' if detailed_comparison else ''}
        
        {f'''
        <div class="section">
            <h2>📄 Individual Paper Summaries</h2>
            {"".join([f"""
            <div class="paper-summary">
                <h4>Paper {summary.get('paper_number', 'N/A')}: {summary.get('title', 'Unknown')}</h4>
                <div class="summary-text">{summary.get('summary', 'No summary available')}</div>
                <div class="strengths"><strong>Key Strengths:</strong> {summary.get('key_strengths', 'Not specified')}</div>
                <div class="limitations"><strong>Key Limitations:</strong> {summary.get('key_limitations', 'Not specified')}</div>
            </div>
            """ for summary in paper_summaries])}
        </div>
        ''' if paper_summaries else ''}
        
        <div class="footer">
            <p>Generated by SyntAI Multi-Paper Comparative Analysis System</p>
            <p>This report provides an automated analysis of research papers. Results should be verified and supplemented with expert domain knowledge.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _clean_for_json(self, obj):
        """Clean data structure for JSON serialization by handling NaN values"""
        import math
        
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
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
    
    def export_results(self, analysis_results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Export analysis results to files"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            exported_files = {}
            
            # 1. Export summary as JSON
            summary_path = output_path / "summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results.get('summary', {}), f, indent=2, ensure_ascii=False)
            exported_files['summary'] = str(summary_path)
            
            # 2. Export comparison table
            comparison = analysis_results.get('comparison', {})
            if comparison:
                comparison_path = output_path / "comparison.json"
                with open(comparison_path, 'w', encoding='utf-8') as f:
                    json.dump(comparison, f, indent=2, ensure_ascii=False)
                exported_files['comparison'] = str(comparison_path)
            
            # 3. Export HTML report
            html_content = self._generate_html_report(analysis_results)
            html_path = output_path / "report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            exported_files['html_report'] = str(html_path)
            
            # 4. Copy visualization files
            visualizations = analysis_results.get('visualizations', {})
            for viz_name, viz_path in visualizations.items():
                if viz_path and Path(viz_path).exists():
                    new_path = output_path / f"{viz_name}.png"
                    import shutil
                    shutil.copy2(viz_path, new_path)
                    exported_files[f"viz_{viz_name}"] = str(new_path)
            
            logger.info(f"Results exported to {output_dir}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return {"error": str(e)}


# Example usage function
def run_comparative_analysis(pdf_paths: List[str], output_dir: str = "analysis_output") -> Dict[str, Any]:
    """
    Main function to run complete comparative analysis
    
    Args:
        pdf_paths (List[str]): List of PDF file paths
        output_dir (str): Output directory for results
        
    Returns:
        Dict containing complete analysis results and export paths
    """
    service = ComparativeAnalysisService()
    
    # Run analysis
    results = service.analyze_papers(pdf_paths)
    
    if results.get("success"):
        # Export results
        export_paths = service.export_results(results, output_dir)
        results["export_paths"] = export_paths
        
        logger.info("Comparative analysis completed successfully!")
        logger.info(f"Results exported to: {output_dir}")
        
        return results
    else:
        logger.error("Comparative analysis failed")
        return results


if __name__ == "__main__":
    # Example usage
    sample_pdfs = [
        "paper1.pdf",
        "paper2.pdf", 
        "paper3.pdf"
    ]
    
    # Run analysis
    results = run_comparative_analysis(sample_pdfs)
    
    if results.get("success"):
        print("Analysis completed successfully!")
        print(f"Analyzed {results['paper_count']} papers")
        print("\nSummary:")
        print(results.get('summary', 'No summary available'))
    else:
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
