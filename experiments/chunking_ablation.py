from __future__ import annotations

from pathlib import Path
from typing import List

from pipelines import IngestionPipeline

RETRIEVER_CFG = Path("config/retriever_config.yaml")


def run_ablation(chunk_sizes: List[int]) -> None:
    original_text = RETRIEVER_CFG.read_text()
    for size in chunk_sizes:
        print(f"Testing chunk size {size}")
        cfg_text = RETRIEVER_CFG.read_text()
        modified = cfg_text.replace("chunk_size: 800", f"chunk_size: {size}")
        RETRIEVER_CFG.write_text(modified)
        pipeline = IngestionPipeline()
        pipeline.ingest(Path("data/raw_docs"))
    RETRIEVER_CFG.write_text(original_text)


if __name__ == "__main__":
    run_ablation([400, 800, 1200])
