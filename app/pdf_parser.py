"""
PDF Parser service using PyMuPDF and pdfplumber.
Extracts text, tables, and metadata from PDF files.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. PDF parsing will be limited.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed. Table extraction will be disabled.")


class PDFParser:
    """Service for parsing PDF documents."""

    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables and PDFPLUMBER_AVAILABLE

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a PDF file and extract text, tables, and metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing parsed content
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF is not installed")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Parsing PDF: {file_path.name}")

        result = {
            "filename": file_path.name,
            "filepath": str(file_path),
            "pages": [],
            "total_pages": 0,
            "metadata": {}
        }

        try:
            # Open with PyMuPDF for text and basic metadata
            doc = fitz.open(file_path)
            
            # Extract metadata
            metadata = doc.metadata
            result["metadata"] = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", "")
            }

            result["total_pages"] = len(doc)

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                page_data = {
                    "page_number": page_num + 1,
                    "text": "",
                    "tables": []
                }

                # Extract text
                page_data["text"] = page.get_text("text")

                # Extract tables if enabled
                if self.extract_tables and PDFPLUMBER_AVAILABLE:
                    try:
                        with pdfplumber.open(file_path) as pdf:
                            if page_num < len(pdf.pages):
                                pdf_page = pdf.pages[page_num]
                                tables = pdf_page.extract_tables()
                                
                                for table_idx, table in enumerate(tables):
                                    if table:
                                        # Convert table to markdown-like format
                                        table_text = self._table_to_text(table)
                                        page_data["tables"].append({
                                            "table_index": table_idx,
                                            "content": table_text,
                                            "rows": len(table),
                                            "cols": max(len(row) for row in table) if table else 0
                                        })
                    except Exception as e:
                        logger.warning(f"Table extraction failed for page {page_num + 1}: {e}")

                result["pages"].append(page_data)

            doc.close()
            logger.info(f"Successfully parsed {file_path.name}: {result['total_pages']} pages")

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path.name}: {e}")
            raise

        return result

    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert a table to text format."""
        lines = []
        for row in table:
            if row:
                # Filter out None values and join cells
                cleaned_row = [cell if cell else "" for cell in row]
                lines.append(" | ".join(cleaned_row))
        return "\n".join(lines)

    def get_full_text(self, parsed_data: Dict[str, Any]) -> str:
        """
        Extract full text content from parsed PDF data.
        
        Args:
            parsed_data: Dictionary from parse_pdf()
            
        Returns:
            Full text content with page markers
        """
        texts = []
        
        for page in parsed_data["pages"]:
            page_text = f"\n\n--- Page {page['page_number']} ---\n\n"
            page_text += page["text"]
            
            # Append tables after page text
            for table in page.get("tables", []):
                page_text += f"\n\n[Table {table['table_index'] + 1}]\n{table['content']}\n"
            
            texts.append(page_text)
        
        return "".join(texts)

    def get_pages_with_metadata(
        self, 
        parsed_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get list of pages with their text and metadata.
        
        Args:
            parsed_data: Dictionary from parse_pdf()
            
        Returns:
            List of page dictionaries with text and metadata
        """
        pages = []
        
        for page in parsed_data["pages"]:
            # Combine text and tables
            content = page["text"]
            for table in page.get("tables", []):
                content += f"\n\n[Table]: {table['content']}"
            
            pages.append({
                "page_number": page["page_number"],
                "text": content,
                "source_file": parsed_data["filename"],
                "has_tables": len(page.get("tables", [])) > 0
            })
        
        return pages


def parse_pdf_file(file_path: str, extract_tables: bool = True) -> Dict[str, Any]:
    """
    Convenience function to parse a PDF file.
    
    Args:
        file_path: Path to the PDF file
        extract_tables: Whether to extract tables
        
    Returns:
        Parsed PDF data
    """
    parser = PDFParser(extract_tables=extract_tables)
    return parser.parse_pdf(file_path)
