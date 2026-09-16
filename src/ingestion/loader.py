"""
Document loader — Step 1 of the ingestion pipeline.

Supports:
  - PDF files (via pypdf)
  - Plain text files
  - A directory of either

Returns a list of LangChain Document objects, each with:
  - page_content  : the raw text of that page/file
  - metadata      : source path, page number, file type
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from rich.console import Console

console = Console()


def load_document(file_path: str | Path) -> List[Document]:
    """Load a single file. Returns a list of Documents (one per page for PDFs)."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        # Attach extra metadata
        for doc in docs:
            doc.metadata["file_type"] = "pdf"
            doc.metadata["file_name"] = path.name
        console.print(f"[green]✓[/green] Loaded PDF: {path.name} ({len(docs)} pages)")
        return docs

    elif path.suffix.lower() in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = path.suffix.lstrip(".")
            doc.metadata["file_name"] = path.name
        console.print(f"[green]✓[/green] Loaded text: {path.name}")
        return docs

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def load_directory(dir_path: str | Path) -> List[Document]:
    """Recursively load all PDFs and text files from a directory."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    all_docs = []
    supported = {".pdf", ".txt", ".md"}

    files = [f for f in dir_path.rglob("*") if f.suffix.lower() in supported]
    console.print(f"[blue]Found {len(files)} files in {dir_path}[/blue]")

    for file in files:
        try:
            docs = load_document(file)
            all_docs.extend(docs)
        except Exception as e:
            console.print(f"[yellow]⚠ Skipped {file.name}: {e}[/yellow]")

    console.print(f"[bold green]Total documents loaded: {len(all_docs)}[/bold green]")
    return all_docs
