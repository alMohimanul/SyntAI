import os
import base64
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
class PaperSummarizationService:
    """Service for summarizing paper sections using Groq LLM with Gemma"""

    def __init__(self):
        """Initialize the summarization service"""
        self.client = None
        self.model = "gemma2-9b-it"  # Using Gemma for text summarization
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY not found in environment variables")
            print("Please set GROQ_API_KEY in your .env file")
            return

        try:
            self.client = Groq(api_key=api_key)
            print("✅ Groq Gemma client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Groq Gemma client: {e}")

    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.client is not None

    def create_story_summary(
        self, paper_data: Dict[str, Any], sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create an AI-generated story summary of the paper

        Args:
            paper_data (dict): Paper metadata
            sections (dict): Extracted paper sections

        Returns:
            Dict[str, Any]: Story summary result
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Groq service not available",
                "summary": "",
            }

        abstract_text = sections.get("abstract", {}).get("text", "")
        title = paper_data.get("title", "")

        prompt = f"""
        Create a compelling, easy-to-understand 1-paragraph story summary of this research paper.
        
        Title: {title}
        
        Abstract: {abstract_text}
        
        Write as if you're explaining this research breakthrough to a curious friend. Focus on:
        - What problem they solved
        - Why it matters
        - What makes their approach interesting
        - The potential impact
        
        Keep it engaging, conversational, and under 150 words. No technical jargon.
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
                top_p=0.9,
            )

            summary = completion.choices[0].message.content.strip()

            return {
                "success": True,
                "error": "",
                "summary": summary,
                "word_count": len(summary.split()),
            }

        except Exception as e:
            print(f"❌ Error creating story summary: {e}")
            return {"success": False, "error": str(e), "summary": ""}

    def summarize_abstract(self, abstract_text: str) -> Dict[str, Any]:
        """
        Create an AI-narrated explanation of the abstract

        Args:
            abstract_text (str): Raw abstract text

        Returns:
            Dict[str, Any]: Narrated abstract result
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Groq service not available",
                "narration": "",
                "key_points": [],
            }

        prompt = f"""
        You are an expert research analyst. Transform this academic abstract into a comprehensive, easy-to-understand explanation that helps users deeply understand the research.
        
        Abstract: {abstract_text}
        
        Provide a detailed analysis with:
        
        1. A conversational narration (4-5 sentences) that explains:
           - What specific problem this research addresses and why it's important
           - What innovative approach or method they used to solve it
           - What they discovered or achieved
           - Why their findings matter in the broader context
        
        2. 4-6 key points that break down:
           - The main research question or challenge
           - The novel technique/approach used
           - Key findings or results
           - Practical implications or applications
           - Significance to the field
           - Future potential or impact
        
        Format your response as:
        NARRATION:
        [Your detailed conversational explanation here - make it engaging and informative]
        
        KEY POINTS:
        • [Research Problem/Challenge]
        • [Novel Approach/Method]
        • [Key Finding 1]
        • [Key Finding 2]
        • [Practical Application]
        • [Broader Impact]
        
        Use simple language but be comprehensive. Help the reader understand not just WHAT was done, but WHY it matters and HOW it advances the field.
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=600,
                top_p=0.8,
            )

            response = completion.choices[0].message.content.strip()

            # Parse the response
            narration = ""
            key_points = []

            if "NARRATION:" in response and "KEY POINTS:" in response:
                parts = response.split("KEY POINTS:")
                narration = parts[0].replace("NARRATION:", "").strip()

                points_text = parts[1].strip()
                # Extract bullet points
                for line in points_text.split("\n"):
                    line = line.strip()
                    if (
                        line.startswith("•")
                        or line.startswith("-")
                        or line.startswith("*")
                    ):
                        key_points.append(line[1:].strip())

            return {
                "success": True,
                "error": "",
                "narration": narration,
                "key_points": key_points,
                "original_text": abstract_text,
            }

        except Exception as e:
            print(f"❌ Error summarizing abstract: {e}")
            return {
                "success": False,
                "error": str(e),
                "narration": "",
                "key_points": [],
            }

    def explain_contributions(
        self, contributions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain why the contributions matter

        Args:
            contributions_data (dict): Extracted contributions data

        Returns:
            Dict[str, Any]: Explained contributions
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Groq service not available",
                "explanations": [],
                "why_it_matters": "",
            }

        contributions_text = contributions_data.get("text", "")
        contributions_items = contributions_data.get("items", [])

        prompt = f"""
        You are a research expert. Analyze and explain why these research contributions are significant and valuable.
        
        Contributions Text: {contributions_text}
        
        Key Contribution Items: {chr(10).join(f"• {item}" for item in contributions_items)}
        
        For each contribution, provide a detailed explanation covering:
        1. What specific problem or limitation it addresses
        2. What novel approach or innovation it introduces
        3. What concrete benefits or improvements it provides
        4. How it advances the current state of the field
        5. What new possibilities it opens up for future research
        6. Why it matters to practitioners or other researchers
        
        Also provide a compelling "Why It Matters" summary (3-4 sentences) that captures the overall significance and potential impact.
        
        Format as:
        EXPLANATIONS:
        • [Contribution 1]: [Detailed explanation covering what it does, why it's innovative, what benefits it provides, and how it advances the field]
        • [Contribution 2]: [Detailed explanation covering what it does, why it's innovative, what benefits it provides, and how it advances the field]
        • [Continue for all contributions...]
        
        WHY IT MATTERS:
        [Compelling 3-4 sentence summary explaining the broader significance, potential applications, and why this work is valuable to the research community and beyond]
        
        Make it comprehensive but accessible. Focus on the real-world impact and significance.
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=700,
                top_p=0.8,
            )

            response = completion.choices[0].message.content.strip()

            # Parse the response
            explanations = []
            why_it_matters = ""

            if "EXPLANATIONS:" in response and "WHY IT MATTERS:" in response:
                parts = response.split("WHY IT MATTERS:")
                explanations_text = parts[0].replace("EXPLANATIONS:", "").strip()
                why_it_matters = parts[1].strip()

                # Extract explanations
                for line in explanations_text.split("\n"):
                    line = line.strip()
                    if line.startswith("•") or line.startswith("-"):
                        explanations.append(line[1:].strip())

            return {
                "success": True,
                "error": "",
                "explanations": explanations,
                "why_it_matters": why_it_matters,
                "original_items": contributions_items,
            }

        except Exception as e:
            print(f"❌ Error explaining contributions: {e}")
            return {
                "success": False,
                "error": str(e),
                "explanations": [],
                "why_it_matters": "",
            }

    def create_methodology_walkthrough(
        self, methodology_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a step-by-step methodology walkthrough

        Args:
            methodology_data (dict): Extracted methodology data

        Returns:
            Dict[str, Any]: Methodology walkthrough
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Groq service not available",
                "walkthrough": [],
                "overview": "",
            }

        method_text = methodology_data.get("text", "")
        steps = methodology_data.get("steps", [])

        prompt = f"""
        You are a research methodology expert. Create a comprehensive, easy-to-understand walkthrough of this research methodology that helps users understand exactly how the research was conducted and why each step matters.
        
        Methodology Text: {method_text}
        
        Identified Steps: {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(steps))}
        
        Provide:
        1. A detailed overview (4-5 sentences) that explains:
           - The overall research approach and strategy
           - Why this particular methodology was chosen
           - What makes it suitable for addressing the research problem
           - How it builds upon or differs from existing approaches
        
        2. A comprehensive step-by-step walkthrough (4-8 steps) that covers:
           - What is done in each step
           - Why this step is necessary
           - How it connects to the previous and next steps
           - What specific techniques, tools, or methods are used
           - What outcomes or outputs are expected from each step
        
        Format as:
        OVERVIEW:
        [Detailed explanation of the research approach, why it was chosen, and how it addresses the research problem effectively]
        
        WALKTHROUGH:
        Step 1: [What is done] - [Why it's important] - [How it works] - [Expected outcome]
        Step 2: [What is done] - [Why it's important] - [How it works] - [Expected outcome]
        Step 3: [What is done] - [Why it's important] - [How it works] - [Expected outcome]
        ...
        
        Make it detailed enough that a researcher could understand the methodology's logic and flow, but accessible enough for non-experts to follow the reasoning.
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=800,
                top_p=0.8,
            )

            response = completion.choices[0].message.content.strip()

            # Parse the response
            overview = ""
            walkthrough_steps = []

            if "OVERVIEW:" in response and "WALKTHROUGH:" in response:
                parts = response.split("WALKTHROUGH:")
                overview = parts[0].replace("OVERVIEW:", "").strip()

                walkthrough_text = parts[1].strip()
                # Extract steps
                for line in walkthrough_text.split("\n"):
                    line = line.strip()
                    if line.startswith("Step"):
                        walkthrough_steps.append(line)

            return {
                "success": True,
                "error": "",
                "overview": overview,
                "walkthrough": walkthrough_steps,
                "original_steps": steps,
            }

        except Exception as e:
            print(f"❌ Error creating methodology walkthrough: {e}")
            return {
                "success": False,
                "error": str(e),
                "overview": "",
                "walkthrough": [],
            }

    def interpret_results(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret and explain research results

        Args:
            results_data (dict): Extracted results data

        Returns:
            Dict[str, Any]: Results interpretation
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Groq service not available",
                "interpretation": "",
                "key_insights": [],
            }

        results_text = results_data.get("text", "")
        findings = results_data.get("findings", [])

        prompt = f"""
        You are a research analysis expert. Interpret and explain these research results in a comprehensive way that helps users understand the significance, implications, and real-world impact.
        
        Results Text: {results_text}
        
        Key Findings: {chr(10).join(f"• {finding}" for finding in findings)}
        
        Provide:
        1. A detailed interpretation (5-6 sentences) that explains:
           - What the results actually demonstrate and prove
           - How these results address the original research question
           - What surprising or expected outcomes emerged
           - How these results compare to previous work in the field
           - What limitations or caveats should be considered
        
        2. 5-7 key insights that cover:
           - The most significant finding and why it matters
           - Performance improvements or new capabilities demonstrated
           - Practical applications and real-world implications
           - What this means for the broader research field
           - Future research directions these results suggest
           - Potential impact on industry or society
           - Any limitations or areas for improvement
        
        Format as:
        INTERPRETATION:
        [Comprehensive analysis of what the results mean, their significance, how they advance our understanding, and their broader implications]
        
        KEY INSIGHTS:
        • [Most significant finding and its importance]
        • [Performance/capability advancement]
        • [Practical application potential]
        • [Impact on research field]
        • [Future research opportunities]
        • [Societal or industry implications]
        • [Limitations or considerations]
        
        Focus on making the significance clear and explaining why these results matter beyond just the technical achievements.
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=700,
                top_p=0.8,
            )

            response = completion.choices[0].message.content.strip()

            # Parse the response
            interpretation = ""
            insights = []

            if "INTERPRETATION:" in response and "KEY INSIGHTS:" in response:
                parts = response.split("KEY INSIGHTS:")
                interpretation = parts[0].replace("INTERPRETATION:", "").strip()

                insights_text = parts[1].strip()
                # Extract insights
                for line in insights_text.split("\n"):
                    line = line.strip()
                    if line.startswith("•") or line.startswith("-"):
                        insights.append(line[1:].strip())

            return {
                "success": True,
                "error": "",
                "interpretation": interpretation,
                "key_insights": insights,
                "original_findings": findings,
            }

        except Exception as e:
            print(f"❌ Error interpreting results: {e}")
            return {
                "success": False,
                "error": str(e),
                "interpretation": "",
                "key_insights": [],
            }


class CodeGenerationService:
    """Service for generating code from paper figures using Groq LLM"""

    def __init__(self):
        """Initialize the code generation service"""
        self.client = None
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY not found in environment variables")
            print("Please set GROQ_API_KEY in your .env file")
            return

        try:
            self.client = Groq(api_key=api_key)
            print("✅ Groq client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Groq client: {e}")

    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.client is not None

    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Encode image to base64 for API transmission

        Args:
            image_path (str): Path to the image file

        Returns:
            Optional[str]: Base64 encoded image or None if error
        """
        try:
            print(f"🔍 Attempting to encode image: {image_path}")
            print(f"📁 File exists: {os.path.exists(image_path)}")

            if not os.path.exists(image_path):
                print(f"❌ File not found at path: {image_path}")
                return None

            # Get file size for verification
            file_size = os.path.getsize(image_path)
            print(f"📊 File size: {file_size} bytes")

            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                print(f"📖 Read {len(image_data)} bytes from file")
                encoded_string = base64.b64encode(image_data).decode("utf-8")
                print(
                    f"✅ Successfully encoded image to base64 ({len(encoded_string)} characters)"
                )
                return encoded_string
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback

            print(f"❌ Traceback: {traceback.format_exc()}")
            return None

    def save_image_to_folder(
        self, image_bytes: bytes, filename: str, folder: str = "inputs/images"
    ) -> Optional[str]:
        """
        Save uploaded image bytes to a specified folder

        Args:
            image_bytes (bytes): The raw image content
            filename (str): Desired image filename (e.g., 'page_1.png')
            folder (str): Target folder to save the image

        Returns:
            Optional[str]: Full path to saved image or None if error
        """
        try:
            os.makedirs(folder, exist_ok=True)
            image_path = os.path.join(folder, filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            print(f"✅ Image saved to: {image_path}")
            return image_path
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return None

    def generate_code_from_image(
        self,
        image_path: str,
        framework: str = "pytorch",
    ) -> Dict[str, Any]:
        """
        Generate code from a paper figure/image

        Args:
            image_path (str): Path to the image file
            framework (str): Target framework ('pytorch' or 'tensorflow')

        Returns:
            Dict[str, Any]: Generated code with metadata
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Groq service not available",
                "code": "",
                "modules": [],
                "explanation": "",
            }

        # Validate image path
        print(f"🔍 Validating image path: {image_path}")
        print(f"📁 Path object exists: {Path(image_path).exists()}")
        print(f"📁 OS path exists: {os.path.exists(image_path)}")
        print(f"📁 Absolute path: {os.path.abspath(image_path)}")

        if not Path(image_path).exists():
            return {
                "success": False,
                "error": f"Image file not found: {image_path}",
                "code": "",
                "modules": [],
                "explanation": "",
            }
        prompt = self._create_image_analysis_prompt(framework, image_path)
        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            return {
                "success": False,
                "error": "Failed to encode image",
                "code": "",
                "modules": [],
                "explanation": "",
            }

        try:
            print(f"🤖 Analyzing image and generating {framework} code...")
            print(f"📁 Image path: {image_path}")
            print(f"📊 Image encoded: {len(image_base64)} characters")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            generated_content = completion.choices[0].message.content
            parsed_result = self._parse_generated_code(generated_content, framework)

            return {
                "success": True,
                "error": "",
                "code": parsed_result["code"],
                "modules": parsed_result["modules"],
                "explanation": parsed_result["explanation"],
                "framework": framework,
                "image_path": image_path,
            }

        except Exception as e:
            print(f"❌ Error generating code: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": "",
                "modules": [],
                "explanation": "",
            }

    def _create_image_analysis_prompt(
        self,
        framework: str,
        image_path: str,
    ) -> str:
        """Create a detailed prompt for image-based code generation"""
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                image_size = img.size
                image_mode = img.mode
        except:
            image_size = "unknown"
            image_mode = "unknown"

        framework_specifics = {
            "pytorch": {
                "imports": "torch, torch.nn, torch.nn.functional, torchvision",
                "style": "PyTorch style with nn.Module classes",
                "example": "class MyModel(nn.Module):",
            },
            "tensorflow": {
                "imports": "tensorflow, keras, tensorflow.keras.layers",
                "style": "TensorFlow/Keras style with functional or sequential API",
                "example": "model = tf.keras.Sequential()",
            },
        }

        specs = framework_specifics.get(framework, framework_specifics["pytorch"])

        prompt = f"""
        Analyze this image and look for any architectural diagrams, flowcharts, or network structures. 

        If you see any diagrams:
        - Implement the diagram components as {framework} code modules
        - Use exact names from diagram labels for classes/functions
        - Create separate modules for each component you can identify
        - Only implement what you can clearly see in the diagram

        If no clear diagrams are visible, return a simple comment explaining what you see.

        Return your response in this format:

        ## Code Implementation
        ```python
        # Your {framework} implementation here
        ```

        ## Diagram Analysis
        - What architectural components did you identify?
        - How do they connect together?

        Focus only on visual diagram elements, ignore any surrounding text.
        """

        return prompt

    def _parse_generated_code(self, content: str, framework: str) -> Dict[str, Any]:
        """
        Parse the generated content to extract code, modules, and explanation

        Args:
            content (str): Generated content from LLM
            framework (str): Target framework

        Returns:
            Dict[str, Any]: Parsed components
        """
        result = {"code": "", "modules": [], "explanation": ""}

        try:
            import re

            code_pattern = r"```python\s*(.*?)\s*```"
            code_matches = re.findall(code_pattern, content, re.DOTALL)

            if code_matches:
                result["code"] = code_matches[0].strip()

            modules_pattern = r"## Module Breakdown\s*(.*?)(?=##|$)"
            modules_match = re.search(modules_pattern, content, re.DOTALL)
            if modules_match:
                modules_text = modules_match.group(1).strip()
                module_lines = [
                    line.strip()
                    for line in modules_text.split("\n")
                    if line.strip().startswith("-")
                ]
                result["modules"] = [line[1:].strip() for line in module_lines]

            explanation_patterns = [
                r"## Diagram Analysis\s*(.*?)(?=##|$)",
                r"## Implementation Notes\s*(.*?)(?=##|$)",
                r"## Analysis\s*(.*?)(?=##|$)",
                r"## Explanation\s*(.*?)$",
            ]

            for pattern in explanation_patterns:
                explanation_match = re.search(pattern, content, re.DOTALL)
                if explanation_match:
                    result["explanation"] = explanation_match.group(1).strip()
                    break

            if not result["explanation"]:
                sections = content.split("##")
                if len(sections) > 1:
                    result["explanation"] = sections[-1].strip()
                else:
                    result["explanation"] = (
                        "Generated code implementation based on diagram analysis."
                    )

        except Exception as e:
            print(f"⚠️ Error parsing generated content: {e}")
            result["code"] = content
            result["explanation"] = "Generated implementation code."

        return result

    def generate_streaming_code(
        self,
        image_path: str,
        framework: str = "pytorch",
    ):
        """
        Generate code with streaming response (generator)

        Args:
            image_path (str): Path to the image file
            framework (str): Target framework

        Yields:
            str: Streaming content chunks
        """
        if not self.is_available():
            yield "❌ Groq service not available"
            return

        prompt = self._create_image_analysis_prompt(framework, image_path)
        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            yield "❌ Failed to encode image"
            return

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"❌ Error generating code: {str(e)}"

    def test_image_analysis(self, image_path: str) -> str:
        """
        Simple test to verify image is being processed correctly

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Simple description of what the model sees
        """
        if not self.is_available():
            return "❌ Groq service not available"

        if not Path(image_path).exists():
            return f"❌ Image file not found: {image_path}"
        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            return "❌ Failed to encode image"

        try:
            print(f"🧪 Testing image analysis for: {image_path}")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What do you see in this image? Describe any diagrams, flowcharts, or architectural elements.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"❌ Error in test: {str(e)}"

    def deep_dive_page_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive deep dive analysis of a paper page

        Args:
            image_path (str): Path to the page image

        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Groq service not available",
                "insights": [],
                "explanation": "",
                "content_type": "",
                "has_diagram": False,
                "code": "",
            }

        if not Path(image_path).exists():
            return {
                "success": False,
                "error": f"Image file not found: {image_path}",
                "insights": [],
                "explanation": "",
                "content_type": "",
                "has_diagram": False,
                "code": "",
            }

        prompt = """
        You are an expert research analyst. Analyze this page from a research paper and provide comprehensive insights.

        Please analyze the page content and provide:

        1. CONTENT TYPE: What type of content is this? (e.g., "Abstract", "Methodology", "Results", "Introduction", "Figure/Diagram", "Equations", "References", "Mixed Content")

        2. KEY INSIGHTS: 5-7 bullet points with the most important insights from this page:
           - What are the main concepts or findings presented?
           - What technical details or innovations are discussed?
           - What methodologies or approaches are described?
           - What results or conclusions are shown?
           - How does this contribute to the research?

        3. DETAILED EXPLANATION: A comprehensive 4-6 sentence explanation of what this page contributes to the research paper and why it's significant.

        4. DIAGRAM ANALYSIS: If there are any diagrams, flowcharts, or architectural elements:
           - Describe what they show
           - Explain their purpose and significance
           - Note if code could be generated from them

        5. TECHNICAL ELEMENTS: Identify any:
           - Mathematical equations or formulas
           - Algorithms or pseudocode
           - Data tables or experimental results
           - Technical specifications

        Format your response as:
        CONTENT_TYPE: [Type of content]
        
        HAS_DIAGRAM: [Yes/No - whether there are implementable diagrams]
        
        KEY_INSIGHTS:
        • [Insight 1]
        • [Insight 2]
        • [Insight 3]
        • [Insight 4]
        • [Insight 5]
        
        DETAILED_EXPLANATION:
        [Your comprehensive explanation here]
        
        DIAGRAM_ANALYSIS:
        [Analysis of any diagrams or visual elements]
        
        TECHNICAL_ELEMENTS:
        [Description of technical content like equations, algorithms, etc.]

        Be thorough and provide actionable insights that help researchers understand the significance of this page.
        """

        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            return {
                "success": False,
                "error": "Failed to encode image",
                "insights": [],
                "explanation": "",
                "content_type": "",
                "has_diagram": False,
                "code": "",
            }

        try:
            print(f"🔍 Performing deep dive analysis on: {image_path}")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=0.9,
                stream=False,
                stop=None,
            )

            response = completion.choices[0].message.content.strip()
            parsed_analysis = self._parse_deep_dive_response(response)

            # If diagram is detected, generate code
            if parsed_analysis["has_diagram"]:
                code_result = self.generate_code_from_image(image_path, "pytorch")
                if code_result.get("success"):
                    parsed_analysis["code"] = code_result.get("code", "")
                else:
                    parsed_analysis["code"] = "# Code generation failed"

            return {
                "success": True,
                "error": "",
                **parsed_analysis,
                "image_path": image_path,
            }

        except Exception as e:
            print(f"❌ Error in deep dive analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "insights": [],
                "explanation": "",
                "content_type": "",
                "has_diagram": False,
                "code": "",
            }

    def _parse_deep_dive_response(self, response: str) -> Dict[str, Any]:
        """Parse the deep dive analysis response"""
        result = {
            "content_type": "",
            "has_diagram": False,
            "insights": [],
            "explanation": "",
            "diagram_analysis": "",
            "technical_elements": "",
        }

        try:
            import re

            # Extract content type
            content_type_match = re.search(r"CONTENT_TYPE:\s*(.*?)(?=\n|$)", response)
            if content_type_match:
                result["content_type"] = content_type_match.group(1).strip()

            # Extract has diagram
            has_diagram_match = re.search(r"HAS_DIAGRAM:\s*(.*?)(?=\n|$)", response)
            if has_diagram_match:
                diagram_text = has_diagram_match.group(1).strip().lower()
                result["has_diagram"] = "yes" in diagram_text

            # Extract key insights
            insights_match = re.search(
                r"KEY_INSIGHTS:\s*(.*?)(?=DETAILED_EXPLANATION:|$)", response, re.DOTALL
            )
            if insights_match:
                insights_text = insights_match.group(1).strip()
                for line in insights_text.split("\n"):
                    line = line.strip()
                    if (
                        line.startswith("•")
                        or line.startswith("-")
                        or line.startswith("*")
                    ):
                        result["insights"].append(line[1:].strip())

            # Extract detailed explanation
            explanation_match = re.search(
                r"DETAILED_EXPLANATION:\s*(.*?)(?=DIAGRAM_ANALYSIS:|$)",
                response,
                re.DOTALL,
            )
            if explanation_match:
                result["explanation"] = explanation_match.group(1).strip()

            # Extract diagram analysis
            diagram_match = re.search(
                r"DIAGRAM_ANALYSIS:\s*(.*?)(?=TECHNICAL_ELEMENTS:|$)",
                response,
                re.DOTALL,
            )
            if diagram_match:
                result["diagram_analysis"] = diagram_match.group(1).strip()

            # Extract technical elements
            technical_match = re.search(
                r"TECHNICAL_ELEMENTS:\s*(.*?)$", response, re.DOTALL
            )
            if technical_match:
                result["technical_elements"] = technical_match.group(1).strip()

        except Exception as e:
            print(f"⚠️ Error parsing deep dive response: {e}")

        return result

    def verify_image_path(self, image_path: str) -> Dict[str, Any]:
        """
        Verify if an image path is valid and can be processed

        Args:
            image_path (str): Path to the image file

        Returns:
            Dict[str, Any]: Verification results
        """
        result = {
            "path_provided": image_path,
            "absolute_path": os.path.abspath(image_path),
            "exists": False,
            "readable": False,
            "size_bytes": 0,
            "can_encode": False,
            "base64_length": 0,
            "error": None,
        }

        try:
            print(f"🔍 Verifying image path: {image_path}")
            result["exists"] = os.path.exists(image_path)
            print(f"📁 Path exists: {result['exists']}")

            if not result["exists"]:
                result["error"] = "File does not exist"
                return result
            try:
                with open(image_path, "rb") as f:
                    data = f.read(100)
                    result["readable"] = True
                    result["size_bytes"] = os.path.getsize(image_path)
                print(
                    f"📖 File readable: {result['readable']}, Size: {result['size_bytes']} bytes"
                )
            except Exception as e:
                result["error"] = f"Cannot read file: {str(e)}"
                return result
            encoded = self.encode_image_to_base64(image_path)
            if encoded:
                result["can_encode"] = True
                result["base64_length"] = len(encoded)
                print(
                    f"✅ Image encoding successful: {result['base64_length']} characters"
                )
            else:
                result["error"] = "Failed to encode image to base64"

        except Exception as e:
            result["error"] = f"Verification error: {str(e)}"

        return result


# Global instances
paper_summarization_service = PaperSummarizationService()
code_generation_service = CodeGenerationService()
