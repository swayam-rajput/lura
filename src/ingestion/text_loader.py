from pathlib import Path
import PyPDF2
import docx
import os


def load_text(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    ext = path.suffix.lower()

    if ext == ".txt" or ext == ".md":
        return _load_txt(path)

    elif ext == ".pdf":
        return _load_pdf(path)

    elif ext == ".docx":
        return _load_docx(path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _load_txt(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return " ".join(text.split())

def _load_pdf(path: Path):
    reader = PyPDF2.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return " ".join(" ".join(pages).split())

def _load_docx(path: Path):
    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return " ".join(" ".join(paragraphs).split())
