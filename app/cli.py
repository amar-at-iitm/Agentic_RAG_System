from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pipelines import IngestionPipeline, RAGPipeline

app = typer.Typer(help="Multi-agent RAG system CLI")
console = Console()


@app.command()
def ingest(source: Path = typer.Option(Path("data/raw_docs"), help="Directory of PDFs to ingest")) -> None:
    pipeline = IngestionPipeline()
    console.print("[bold green]Starting ingestion...[/bold green]")
    pipeline.ingest(source)
    console.print("[bold green]Vector store updated.[/bold green]")


@app.command()
def ask(question: str, top_k: int = typer.Option(None, help="Override retrieval depth")) -> None:
    pipeline = RAGPipeline()
    console.print(f"[bold cyan]Question:[/bold cyan] {question}")
    result = pipeline.query(question, top_k=top_k)
    console.print("[bold yellow]Plan[/bold yellow]", result["plan"])
    console.print("[bold yellow]Verification[/bold yellow]", result["verification"])
    console.print("[bold green]Answer[/bold green]\n" + result["answer"])


if __name__ == "__main__":
    app()
