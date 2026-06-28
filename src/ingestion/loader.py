import os
from pypdf import PdfReader


def load_documents(data_dir: str) -> list:
    """Load all PDF and .txt files from data_dir. Returns list of page dicts."""
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if filename.lower().endswith(".pdf"):
            documents.extend(_load_pdf(filepath, filename))
        elif filename.lower().endswith(".txt"):
            documents.extend(_load_txt(filepath, filename))
    return documents


def _load_pdf(filepath: str, filename: str) -> list:
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "source": filename, "page": i + 1})
    return pages


def _load_txt(filepath: str, filename: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return [{"text": text, "source": filename, "page": 1}]
