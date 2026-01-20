from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from tools.config_loader import load_yaml_file
from tools.embedding_utils import build_embeddings

MODEL_CONFIG_PATH = Path("config/model_config.yaml")
RETRIEVER_PATH = Path("config/retriever_config.yaml")


def load_vector_store(retriever_config: dict | None = None) -> Optional[Chroma]:
    cfg = retriever_config or load_yaml_file(RETRIEVER_PATH)
    persist_directory = Path(cfg["vector_store"]["persist_directory"])
    if not persist_directory.exists():
        return None
    embeddings = build_embeddings(MODEL_CONFIG_PATH)
    return Chroma(
        collection_name=cfg["vector_store"]["collection_name"],
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def persist_vector_store(documents: Iterable[Document], retriever_config: dict | None = None) -> Chroma:
    cfg = retriever_config or load_yaml_file(RETRIEVER_PATH)
    embeddings = build_embeddings(MODEL_CONFIG_PATH)
    persist_directory = Path(cfg["vector_store"]["persist_directory"])
    persist_directory.parent.mkdir(parents=True, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=list(documents),
        collection_name=cfg["vector_store"]["collection_name"],
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    vectordb.persist()
    return vectordb
