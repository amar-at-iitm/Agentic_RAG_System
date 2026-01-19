from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from PyPDF2 import PdfReader


def load_pdfs_from_directory(directory: Path | str) -> List[Document]:
    directory = Path(directory)
    documents: List[Document] = []
    for pdf_path in directory.glob("*.pdf"):
        documents.extend(load_single_pdf(pdf_path))
    return documents


def load_single_pdf(path: Path | str) -> List[Document]:
    path = Path(path)
    reader = PdfReader(str(path))
    docs: List[Document] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "page": idx + 1,
                },
            )
        )
    return docs
