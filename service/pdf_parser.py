#!/usr/bin/env python3
"""
PDF Text Extraction and Analysis Service

Uses PyMuPDF (fitz) to extract and parse text from research papers.
Identifies sections like Abstract, Contributions, Methodology, and Results.
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class PaperTextExtractor:
    """Service for extracting and parsing text from PDF papers"""

    def __init__(self):
        """Initialize the PDF text extractor"""
        # Enhanced section headers patterns (case-insensitive, more comprehensive)
        self.section_patterns = {
            "abstract": [
                r"\babstract\b",
                r"\bsummary\b",
                r"\bexecutive\s+summary\b",
                r"^\s*abstract\s*$",
                r"^\s*summary\s*$",
            ],
            "introduction": [
                r"\bintroduction\b",
                r"\b1\.\s*introduction\b",
                r"\b1\s+introduction\b",
                r"^\s*1\.\s*introduction\s*$",
                r"^\s*introduction\s*$",
                r"\bbackground\b",
                r"\bmotivation\b",
            ],
            "contributions": [
                r"\bcontributions?\b",
                r"\bmain\s+contributions?\b",
                r"\bour\s+contributions?\b",
                r"\bkey\s+contributions?\b",
                r"\bnovel\s+contributions?\b",
                r"\bprimary\s+contributions?\b",
                r"\btechnical\s+contributions?\b",
                r"^\s*contributions?\s*$",
                r"\bcontributions?\s+of\s+this\s+work\b",
                r"\bcontributions?\s+include\b",
            ],
            "methodology": [
                r"\bmethodology\b",
                r"\bmethod\b",
                r"\bmethods\b",
                r"\bapproach\b",
                r"\bproposed\s+method\b",
                r"\bproposed\s+approach\b",
                r"\balgorithm\b",
                r"\bframework\b",
                r"\barchitecture\b",
                r"\bmodel\b",
                r"\btechnique\b",
                r"\bdesign\b",
                r"\bimplementation\b",
                r"\bsystem\s+design\b",
                r"\bsystem\s+architecture\b",
                r"^\s*methodology\s*$",
                r"^\s*method\s*$",
                r"^\s*approach\s*$",
                r"\b\d+\.\s*methodology\b",
                r"\b\d+\.\s*method\b",
                r"\b\d+\.\s*approach\b",
            ],
            "results": [
                r"\bresults?\b",
                r"\bexperiments?\b",
                r"\bexperimental\s+results?\b",
                r"\bevaluation\b",
                r"\bperformance\b",
                r"\bfindings\b",
                r"\banalysis\b",
                r"\bexperimental\s+evaluation\b",
                r"\bperformance\s+evaluation\b",
                r"\bempirical\s+results?\b",
                r"^\s*results?\s*$",
                r"^\s*experiments?\s*$",
                r"^\s*evaluation\s*$",
                r"\b\d+\.\s*results?\b",
                r"\b\d+\.\s*experiments?\b",
                r"\b\d+\.\s*evaluation\b",
            ],
            "conclusion": [
                r"\bconclusion\b",
                r"\bconclusions?\b",
                r"\bsummary\b",
                r"\bfuture\s+work\b",
                r"\blimitations?\b",
                r"\bdiscussion\b",
                r"^\s*conclusion\s*$",
                r"^\s*conclusions?\s*$",
                r"^\s*discussion\s*$",
                r"\b\d+\.\s*conclusion\b",
                r"\b\d+\.\s*conclusions?\b",
            ],
        }

    def extract_pdf_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all text from PDF

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            Dict[str, Any]: Extracted text and metadata
        """
        if not Path(pdf_path).exists():
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}",
                "text": "",
                "pages": [],
                "total_pages": 0,
            }

        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            full_text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                pages_text.append(
                    {
                        "page_number": page_num + 1,
                        "text": page_text,
                        "char_count": len(page_text),
                    }
                )
                full_text += page_text + "\n"

            doc.close()

            return {
                "success": True,
                "error": "",
                "text": full_text,
                "pages": pages_text,
                "total_pages": len(pages_text),
                "char_count": len(full_text),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error extracting PDF text: {str(e)}",
                "text": "",
                "pages": [],
                "total_pages": 0,
            }

    def extract_paper_sections(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract and identify paper sections from PDF

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            Dict[str, Any]: Identified sections and content
        """
        # First extract all text
        extraction_result = self.extract_pdf_text(pdf_path)

        if not extraction_result["success"]:
            return extraction_result

        full_text = extraction_result["text"]

        # Clean and normalize text
        clean_text = self._clean_text(full_text)

        # Extract sections
        sections = self._identify_sections(clean_text)

        # Extract specific content
        result = {
            "success": True,
            "error": "",
            "pdf_path": pdf_path,
            "total_pages": extraction_result["total_pages"],
            "char_count": extraction_result["char_count"],
            "sections": sections,
            "abstract": self._extract_abstract(clean_text),
            "contributions": self._extract_contributions(clean_text, sections),
            "methodology": self._extract_methodology(clean_text, sections),
            "results": self._extract_results(clean_text, sections),
            "introduction": self._extract_introduction(clean_text, sections),
            "conclusion": self._extract_conclusion(clean_text, sections),
        }

        return result

    def _clean_text(self, text: str) -> str:
        """Clean and normalize PDF text"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely page numbers
            if len(line) < 3:
                continue
            # Skip lines that look like page numbers
            if re.match(r"^\d+$", line):
                continue
            cleaned_lines.append(line)

        return " ".join(cleaned_lines)

    def _identify_sections(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Identify section headers and their positions in text"""
        sections = {}
        text_lower = text.lower()

        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower))
                if matches:
                    # Take the first match for each section
                    match = matches[0]
                    sections[section_name] = {
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "pattern_matched": pattern,
                        "header_text": text[match.start() : match.end()],
                    }
                    break  # Found a match, move to next section

        return sections

    def _extract_abstract(self, text: str) -> Dict[str, Any]:
        """Extract abstract from paper text with improved patterns"""
        text_lower = text.lower()

        # Enhanced abstract patterns - more comprehensive
        abstract_patterns = [
            # Standard abstract patterns
            r"\babstract\b\s*[\-\:\.]*\s*(.*?)(?=\b(?:keywords?|introduction|1\.?\s*introduction|background|motivation)\b)",
            r"\babstract\b\s*[\-\:\.]*\s*(.*?)(?=\n\s*\n\s*[A-Z])",
            r"\babstract\b\s*[\-\:\.]*\s*(.*?)(?=\n\s*1\.\s*)",
            r"\babstract\b\s*[\-\:\.]*\s*(.{100,1500}?)(?=\b(?:keywords?|introduction|background)\b)",
            # Alternative patterns for different formats
            r"^\s*abstract\s*[\-\:\.]*\s*(.*?)(?=\n\s*\n)",
            r"abstract\s*[\-\:\.]*\s*\n\s*(.*?)(?=\n\s*keywords?)",
            r"abstract\s*[\-\:\.]*\s*\n\s*(.*?)(?=\n\s*1\.)",
            # Patterns for papers without clear separators
            r"\babstract\b\s*[\-\:\.]*\s*([^\.]*\.(?:[^\.]*\.){1,8})",
            # Summary patterns as fallback
            r"\bsummary\b\s*[\-\:\.]*\s*(.*?)(?=\b(?:keywords?|introduction|1\.?\s*introduction)\b)",
        ]

        for pattern in abstract_patterns:
            match = re.search(
                pattern, text_lower, re.DOTALL | re.IGNORECASE | re.MULTILINE
            )
            if match:
                abstract_text = match.group(1).strip()

                # Clean up the abstract text
                abstract_text = re.sub(
                    r"\s+", " ", abstract_text
                )  # Normalize whitespace
                abstract_text = re.sub(
                    r"^\W+", "", abstract_text
                )  # Remove leading non-word chars

                # Validate abstract length (should be substantial)
                if len(abstract_text) > 50 and len(abstract_text.split()) > 10:
                    # Get the original case version
                    start_pos = text_lower.find(abstract_text.lower())
                    if start_pos != -1:
                        original_abstract = text[
                            start_pos : start_pos + len(abstract_text)
                        ].strip()
                    else:
                        original_abstract = abstract_text

                    return {
                        "found": True,
                        "text": original_abstract,
                        "word_count": len(original_abstract.split()),
                        "char_count": len(original_abstract),
                        "pattern_used": pattern,
                    }

        # Enhanced fallback: look for the first substantial paragraph after "abstract"
        abstract_match = re.search(r"\babstract\b", text_lower)
        if abstract_match:
            # Get text after "abstract" keyword
            start_pos = abstract_match.end()
            remaining_text = text[
                start_pos : start_pos + 2000
            ]  # Increased to 2000 chars

            # Look for meaningful content
            lines = remaining_text.split("\n")
            abstract_lines = []

            for line in lines:
                line = line.strip()
                if len(line) > 20:  # Skip very short lines
                    abstract_lines.append(line)
                    # Stop if we hit a section header
                    if re.match(
                        r"^\d+\.?\s*(introduction|background|method)", line.lower()
                    ):
                        break
                    # Stop after reasonable length
                    if len(" ".join(abstract_lines)) > 800:
                        break

            if abstract_lines:
                abstract_text = " ".join(abstract_lines)
                if len(abstract_text) > 50:
                    return {
                        "found": True,
                        "text": abstract_text,
                        "word_count": len(abstract_text.split()),
                        "char_count": len(abstract_text),
                        "pattern_used": "fallback_paragraph",
                    }

        return {
            "found": False,
            "text": "",
            "word_count": 0,
            "char_count": 0,
            "pattern_used": "none",
        }

    def _extract_contributions(self, text: str, sections: Dict) -> Dict[str, Any]:
        """Extract contributions from paper text with enhanced patterns"""
        # Look for explicit contributions section
        if "contributions" in sections:
            start_pos = sections["contributions"]["start_pos"]
            # Find next section or take reasonable chunk
            next_section_pos = len(text)
            for section_name, section_info in sections.items():
                if section_info["start_pos"] > start_pos:
                    next_section_pos = min(next_section_pos, section_info["start_pos"])

            contrib_text = text[start_pos:next_section_pos]

            # Extract bullet points or numbered items
            contributions = self._extract_bullet_points(contrib_text)

            if contributions:
                return {
                    "found": True,
                    "items": contributions,
                    "text": contrib_text[:800],  # Increased to 800 chars
                    "count": len(contributions),
                    "source": "dedicated_section",
                }

        # Enhanced fallback: look for contribution keywords anywhere in the text
        contrib_patterns = [
            # Direct contribution statements
            r"(?:our|main|key|primary|novel|technical)\s+contributions?\s+(?:are|include|of\s+this\s+work)?\s*[:.]?\s*(.*?)(?=\n\s*\n|\n\s*[A-Z]|\.|$)",
            r"we\s+contribute\s+the\s+following\s*[:.]?\s*(.*?)(?=\n\s*\n|\n\s*[A-Z]|$)",
            r"this\s+(?:paper|work)\s+(?:makes|provides)\s+the\s+following\s+contributions?\s*[:.]?\s*(.*?)(?=\n\s*\n|$)",
            r"specifically,?\s+(?:our|the)\s+contributions?\s+(?:are|include)\s*[:.]?\s*(.*?)(?=\n\s*\n|$)",
            # Numbered/bulleted contributions
            r"contributions?\s+of\s+this\s+(?:paper|work)\s*[:.]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|background|method)|$)",
            r"contributions?\s*[:.]?\s*\n\s*(?:\d+\.|\*|•|-|\(i\)|\(1\))(.*?)(?=\n\s*(?:keywords?|introduction|background)|$)",
        ]

        best_match = None
        best_score = 0

        for pattern in contrib_patterns:
            matches = re.finditer(
                pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE
            )
            for match in matches:
                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 500)
                context = text[context_start:context_end]

                # Extract contributions from this context
                contributions = self._extract_bullet_points(context)

                # Score this match based on number and quality of contributions
                score = len(contributions) * 2

                # Bonus for explicit contribution keywords
                if any(
                    word in match.group(0).lower()
                    for word in ["contribution", "novel", "propose"]
                ):
                    score += 3

                # Bonus for being early in the paper (likely introduction)
                if match.start() < len(text) * 0.3:  # First 30% of paper
                    score += 2

                if score > best_score and contributions:
                    best_score = score
                    best_match = {
                        "found": True,
                        "items": contributions,
                        "text": context,
                        "count": len(contributions),
                        "source": "pattern_match",
                        "pattern": pattern,
                    }

        if best_match:
            return best_match

        # Final fallback: look for any enumerated points near contribution keywords
        contrib_keywords = [
            "contribute",
            "contribution",
            "novel",
            "propose",
            "introduce",
        ]

        for keyword in contrib_keywords:
            # Find keyword mentions
            for match in re.finditer(rf"\b{keyword}\w*\b", text, re.IGNORECASE):
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 800)
                context = text[start:end]

                # Look for enumerated items in this context
                contributions = self._extract_bullet_points(context)

                if len(contributions) >= 2:  # At least 2 contributions
                    return {
                        "found": True,
                        "items": contributions[:5],  # Limit to 5
                        "text": context,
                        "count": len(contributions),
                        "source": "keyword_context",
                    }

        return {"found": False, "items": [], "text": "", "count": 0, "source": "none"}

    def _extract_methodology(self, text: str, sections: Dict) -> Dict[str, Any]:
        """Extract methodology from paper text"""
        methodology_sections = ["methodology", "method", "approach"]

        for method_key in methodology_sections:
            if method_key in sections:
                start_pos = sections[method_key]["start_pos"]
                # Find next section
                next_section_pos = len(text)
                for section_name, section_info in sections.items():
                    if section_info["start_pos"] > start_pos:
                        next_section_pos = min(
                            next_section_pos, section_info["start_pos"]
                        )

                method_text = text[start_pos:next_section_pos]

                # Extract key steps or components
                steps = self._extract_methodology_steps(method_text)

                return {
                    "found": True,
                    "text": method_text[:1000],  # First 1000 chars
                    "steps": steps,
                    "word_count": len(method_text.split()),
                    "step_count": len(steps),
                }

        # Fallback: look for methodology keywords
        method_patterns = [
            r"(our\s+approach\s+.*?[\.!?])",
            r"(we\s+propose\s+.*?[\.!?])",
            r"(the\s+method\s+.*?[\.!?])",
            r"(our\s+framework\s+.*?[\.!?])",
        ]

        for pattern in method_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                methodology_text = " ".join(matches[:3])  # First 3 matches
                return {
                    "found": True,
                    "text": methodology_text,
                    "steps": [m.strip() for m in matches[:5]],
                    "word_count": len(methodology_text.split()),
                    "step_count": len(matches),
                }

        return {
            "found": False,
            "text": "",
            "steps": [],
            "word_count": 0,
            "step_count": 0,
        }

    def _extract_results(self, text: str, sections: Dict) -> Dict[str, Any]:
        """Extract results from paper text"""
        result_sections = ["results", "experiments", "evaluation"]

        for result_key in result_sections:
            if result_key in sections:
                start_pos = sections[result_key]["start_pos"]
                # Find next section
                next_section_pos = len(text)
                for section_name, section_info in sections.items():
                    if section_info["start_pos"] > start_pos:
                        next_section_pos = min(
                            next_section_pos, section_info["start_pos"]
                        )

                results_text = text[start_pos:next_section_pos]

                # Extract key findings
                findings = self._extract_key_findings(results_text)

                return {
                    "found": True,
                    "text": results_text[:1000],  # First 1000 chars
                    "findings": findings,
                    "word_count": len(results_text.split()),
                    "finding_count": len(findings),
                }

        return {
            "found": False,
            "text": "",
            "findings": [],
            "word_count": 0,
            "finding_count": 0,
        }

    def _extract_introduction(self, text: str, sections: Dict) -> Dict[str, Any]:
        """Extract introduction from paper text"""
        if "introduction" in sections:
            start_pos = sections["introduction"]["start_pos"]
            # Find next section
            next_section_pos = len(text)
            for section_name, section_info in sections.items():
                if section_info["start_pos"] > start_pos:
                    next_section_pos = min(next_section_pos, section_info["start_pos"])

            intro_text = text[start_pos:next_section_pos]

            return {
                "found": True,
                "text": intro_text[:1000],  # First 1000 chars
                "word_count": len(intro_text.split()),
                "char_count": len(intro_text),
            }

        return {"found": False, "text": "", "word_count": 0, "char_count": 0}

    def _extract_conclusion(self, text: str, sections: Dict) -> Dict[str, Any]:
        """Extract conclusion from paper text"""
        if "conclusion" in sections:
            start_pos = sections["conclusion"]["start_pos"]
            # Take rest of text or reasonable chunk
            conclusion_text = text[start_pos : start_pos + 1000]

            return {
                "found": True,
                "text": conclusion_text,
                "word_count": len(conclusion_text.split()),
                "char_count": len(conclusion_text),
            }

        return {"found": False, "text": "", "word_count": 0, "char_count": 0}

    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points or numbered items from text with enhanced patterns"""
        bullet_patterns = [
            # Standard bullet points
            r"[•\-\*\+]\s+([^\n\r]+)",  # Bullet points
            r"(?:^|\n)\s*[•\-\*\+]\s+([^\n\r]+)",  # Line-starting bullets
            # Numbered lists
            r"(?:^|\n)\s*\d+[\.\)]\s+([^\n\r]+)",  # Numbered items
            r"\(\d+\)\s+([^\n\r]+)",  # Parenthesized numbers
            # Lettered lists
            r"(?:^|\n)\s*\([a-z]\)\s+([^\n\r]+)",  # Lettered items (a), (b), etc.
            r"(?:^|\n)\s*[a-z][\.\)]\s+([^\n\r]+)",  # a., b., etc.
            # Roman numerals
            r"(?:^|\n)\s*(?:i+\.|[ivx]+\.)\s+([^\n\r]+)",  # Roman numerals
            # Special contribution patterns
            r"(?:^|\n)\s*(?:first|second|third|fourth|fifth)[ly,\s]*[:.]?\s*([^\n\r]+)",
            r"(?:^|\n)\s*(?:we|this\s+work|this\s+paper)\s+(?:propose|introduce|present|develop)\s+([^\n\r]+)",
            # Structured sentences that might be contributions
            r"(?:^|\n)\s*(?:specifically|particularly|notably)[,\s]*([^\n\r\.]+\.)",
            r"(?:^|\n)\s*(?:our|the\s+main|key|primary)\s+(?:contribution|achievement|innovation)\s+(?:is|includes?)\s+([^\n\r\.]+\.)",
        ]

        items = []
        seen_items = set()  # Track duplicates

        for pattern in bullet_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()

                # Filter out short or empty items
                if len(clean_match) < 10:
                    continue

                # Filter out obvious non-contributions (headers, etc.)
                if any(
                    word in clean_match.lower()
                    for word in ["page", "figure", "table", "section", "chapter"]
                ):
                    continue

                # Remove trailing punctuation for comparison
                normalized = re.sub(r"[\.,:;]+$", "", clean_match.lower())

                # Avoid duplicates
                if normalized not in seen_items:
                    seen_items.add(normalized)
                    items.append(clean_match)

        return items[:10]  # Limit to 10 items

    def _extract_methodology_steps(self, text: str) -> List[str]:
        """Extract methodology steps from text"""
        step_patterns = [
            r"step\s+\d+[:\.\-]\s*([^\n]+)",
            r"first[ly]*[,\s]+([^\.]+\.)",
            r"second[ly]*[,\s]+([^\.]+\.)",
            r"third[ly]*[,\s]+([^\.]+\.)",
            r"then[,\s]+([^\.]+\.)",
            r"next[,\s]+([^\.]+\.)",
            r"finally[,\s]+([^\.]+\.)",
        ]

        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 15:
                    steps.append(match.strip())

        return steps[:8]  # Max 8 steps

    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from results text"""
        finding_patterns = [
            r"we\s+found\s+that\s+([^\.]+\.)",
            r"results?\s+show\s+that\s+([^\.]+\.)",
            r"our\s+experiments?\s+demonstrate\s+([^\.]+\.)",
            r"(?:significantly|substantially|considerably)\s+([^\.]+\.)",
            r"performance\s+(?:improvement|gain|increase)\s+([^\.]+\.)",
            r"accuracy\s+(?:of|reaches?)\s+([^\.]+\.)",
        ]

        findings = []
        for pattern in finding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 20:
                    findings.append(match.strip())

        return findings[:6]  # Max 6 findings

    def extract_first_page_image_caption(self, pdf_path: str) -> Optional[str]:
        """
        Extract potential image captions from first page for thumbnail generation

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            Optional[str]: First image caption found or None
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return None

            first_page = doc.load_page(0)
            page_text = first_page.get_text()
            doc.close()

            # Look for figure captions
            caption_patterns = [
                r"figure\s+1[:\.\-]\s*([^\n]+)",
                r"fig\.\s*1[:\.\-]\s*([^\n]+)",
                r"figure\s+\d+[:\.\-]\s*([^\n]+)",
            ]

            for pattern in caption_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

            return None

        except Exception as e:
            print(f"Error extracting image caption: {e}")
            return None


# Global instance
pdf_text_extractor = PaperTextExtractor()
