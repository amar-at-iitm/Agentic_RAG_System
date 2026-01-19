from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

from agents.base_agent import BaseAgent
from tools.config_loader import load_yaml_file
from tools.retrieval_tools import load_vector_store

PROMPT_PATH = Path("config/agent_prompts.yaml")
MODEL_PATH = Path("config/model_config.yaml")
RETRIEVER_PATH = Path("config/retriever_config.yaml")


class RetrieverAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("retriever", PROMPT_PATH, MODEL_PATH)
        self.retriever_config = load_yaml_file(RETRIEVER_PATH)
        self.vector_store = load_vector_store(self.retriever_config)

    def retrieve(self, question: str, plan: List[Dict[str, Any]], top_k: int | None = None) -> List[Document]:
        if self.vector_store is None:
            raise RuntimeError("Vector store missing. Run the ingestion pipeline first.")
        top_k = top_k or self.retriever_config["retrieval"]["top_k"]
        docs = self.vector_store.similarity_search(question, k=top_k * 2)
        formatted_candidates = self._format_docs(docs)
        selection = self.invoke(
            question=question,
            plan=json.dumps(plan, indent=2),
            top_k=top_k,
            candidates=formatted_candidates,
        )
        doc_ids = self._parse_selection(selection, docs, top_k)
        filtered = [doc for doc in docs if doc.metadata.get("chunk_id") in doc_ids]
        return filtered[:top_k]

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        lines = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id", "chunk-unknown")
            snippet = doc.page_content[:300].replace("\n", " ")
            lines.append(f"{chunk_id}: {snippet}...")
        return "\n".join(lines)

    @staticmethod
    def _parse_selection(selection: str, docs: List[Document], top_k: int) -> List[str]:
        try:
            parsed = json.loads(selection)
            if isinstance(parsed, list):
                chunk_ids = []
                for item in parsed:
                    if isinstance(item, dict) and "chunk_id" in item:
                        chunk_ids.append(str(item["chunk_id"]))
                    elif isinstance(item, str):
                        chunk_ids.append(item)
                return chunk_ids[:top_k]
        except json.JSONDecodeError:
            pass
        default_ids = [doc.metadata.get("chunk_id", f"chunk-{idx}") for idx, doc in enumerate(docs, start=1)]
        return default_ids[:top_k]
