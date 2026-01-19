from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

from tools.config_loader import load_yaml_file


def build_text_splitter(config_path: Path | str) -> RecursiveCharacterTextSplitter:
    cfg = load_yaml_file(config_path)
    chunking = cfg.get("chunking", {})
    return RecursiveCharacterTextSplitter(
        chunk_size=chunking.get("chunk_size", 800),
        chunk_overlap=chunking.get("chunk_overlap", 120),
        separators=[chunking.get("separator", "\n"), "\n\n", " \n", " "],
    )


def build_embeddings(model_config_path: Path | str) -> OllamaEmbeddings:
    cfg = load_yaml_file(model_config_path)
    embeddings_cfg = cfg.get("embeddings", {})
    model = embeddings_cfg.get("model", "nomic-embed-text")
    return OllamaEmbeddings(
        model=model,
        base_url=embeddings_cfg.get("base_url", "http://localhost:11434"),
    )
