from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from tools.config_loader import load_yaml_file
from tools.embedding_utils import build_text_splitter
from tools.pdf_loader import load_pdfs_from_directory
from tools.retrieval_tools import persist_vector_store

CONFIG_PATH = Path("config/retriever_config.yaml")
RAW_DOCS = Path("data/raw_docs")
PROCESSED_DIR = Path("data/processed")


class IngestionPipeline:
    def __init__(self) -> None:
        self.config = load_yaml_file(CONFIG_PATH)
        self.text_splitter = build_text_splitter(CONFIG_PATH)

    def ingest(self, source_dir: Path | str = RAW_DOCS) -> None:
        documents = load_pdfs_from_directory(source_dir)
        if not documents:
            raise RuntimeError("No PDF documents found to ingest.")
        chunks = self._chunk_documents(documents)
        persist_vector_store(chunks, self.config)

    def _chunk_documents(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for idx, chunk in enumerate(self.text_splitter.split_documents(list(documents))):
            chunk.metadata.setdefault("chunk_id", f"chunk-{idx}")
            chunks.append(chunk)
        self._persist_processed(chunks)
        return chunks

    def _persist_processed(self, chunks: List[Document]) -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_file = PROCESSED_DIR / "chunks.txt"
        with output_file.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(f"{chunk.metadata.get('chunk_id')}: {chunk.page_content}\n\n")
