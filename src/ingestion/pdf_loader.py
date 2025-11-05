# src/ingestion/pdf_loader.py
from PyPDF2 import PdfReader
from pathlib import Path

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file.
    Args:
        file_path: Path to the PDF file.
    Returns:
        Combined text from all pages as a single string.
    """
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Invalid PDF path: {file_path}")

    reader = PdfReader(str(path))
    text_pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_pages.append(text)

    return "\n".join(text_pages).strip()
