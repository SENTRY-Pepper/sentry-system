"""
SENTRY — Knowledge Base Ingestion Script
==========================================
One-time (or on-update) script that:
    1. Reads raw OWASP markdown files and legal PDF documents
    2. Extracts and cleans text from each document
    3. Chunks text using the token-aware Chunker
    4. Embeds each chunk using the local sentence-transformer model
    5. Stores chunks + embeddings + metadata in ChromaDB

Run this from the project root:
    python scripts/ingest_knowledge_base.py

Re-run whenever source documents are updated. The script clears
the existing collection before re-ingesting to avoid duplicates.

Dependencies used:
    pdfplumber  — PDF text extraction
    Chunker     — token-aware text splitting
    Embedder    — sentence-transformer vector generation
    chromadb    — vector store persistence
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
import json
from pathlib import Path

# Add project root to path before importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pdfplumber
import chromadb

from config.settings import settings
from ai_engine.embeddings.chunker import Chunker
from ai_engine.embeddings.embedder import Embedder

# ------------------------------------------------------------------
# Document readers
# ------------------------------------------------------------------

def read_markdown_file(path: Path) -> str:
    """Read a .md file and return its text content."""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber.
    Pages are joined with double newlines to preserve structure.
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)
            else:
                print(f"  [Warning] Page {page_num} in '{path.name}' yielded no text — skipping.")
    return "\n\n".join(pages)


# ------------------------------------------------------------------
# Document discovery
# ------------------------------------------------------------------

def collect_documents() -> list[dict]:
    """
    Walk the raw knowledge base directories and return a list of
    document dicts with keys: path, source, doc_type.
    """
    documents = []

    # OWASP markdown files
    owasp_dir = settings.RAW_OWASP_DIR
    if not owasp_dir.exists():
        print(f"[Warning] OWASP directory not found: {owasp_dir}")
    else:
        md_files = sorted(owasp_dir.glob("*.md"))
        if not md_files:
            print(f"[Warning] No .md files found in {owasp_dir}")
        for path in md_files:
            documents.append({
                "path": path,
                "source": path.name,
                "doc_type": "owasp",
            })
        print(f"[Ingest] Found {len(md_files)} OWASP markdown file(s).")

    # Legal PDF files
    legal_dir = settings.RAW_LEGAL_DIR
    if not legal_dir.exists():
        print(f"[Warning] Legal directory not found: {legal_dir}")
    else:
        pdf_files = sorted(legal_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[Warning] No .pdf files found in {legal_dir}")
        for path in pdf_files:
            documents.append({
                "path": path,
                "source": path.name,
                "doc_type": "legal",
            })
        print(f"[Ingest] Found {len(pdf_files)} legal PDF file(s).")

    return documents


# ------------------------------------------------------------------
# ChromaDB setup
# ------------------------------------------------------------------

def get_or_create_collection(client: chromadb.PersistentClient):
    """
    Delete the existing collection (if any) and create a fresh one.
    This guarantees no stale or duplicate chunks persist across runs.
    """
    existing = [c.name for c in client.list_collections()]
    if settings.CHROMA_COLLECTION_NAME in existing:
        print(
            f"[Ingest] Existing collection '{settings.CHROMA_COLLECTION_NAME}' "
            f"found — deleting for clean re-ingestion."
        )
        client.delete_collection(settings.CHROMA_COLLECTION_NAME)

    collection = client.create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        # cosine similarity is standard for sentence-transformer embeddings
    )
    print(f"[Ingest] Created fresh collection: '{settings.CHROMA_COLLECTION_NAME}'")
    return collection


# ------------------------------------------------------------------
# Core ingestion
# ------------------------------------------------------------------

def ingest_document(
    doc: dict,
    chunker: Chunker,
    embedder: Embedder,
    collection,
    chunk_id_offset: int,
) -> int:
    """
    Process a single document: read → chunk → embed → store.

    Args:
        doc:             Document dict (path, source, doc_type).
        chunker:         Chunker instance.
        embedder:        Embedder instance.
        collection:      ChromaDB collection.
        chunk_id_offset: Running integer to ensure unique IDs across docs.

    Returns:
        Number of chunks ingested from this document.
    """
    path: Path = doc["path"]
    source: str = doc["source"]
    doc_type: str = doc["doc_type"]

    print(f"\n[Ingest] Processing: {source} ({doc_type})")

    # 1. Extract text
    if path.suffix == ".md":
        raw_text = read_markdown_file(path)
    elif path.suffix == ".pdf":
        raw_text = read_pdf_file(path)
    else:
        print(f"  [Skip] Unsupported file type: {path.suffix}")
        return 0

    if not raw_text.strip():
        print(f"  [Skip] No text extracted from {source}.")
        return 0

    print(f"  Extracted {len(raw_text):,} characters of raw text.")

    # 2. Chunk
    chunks = chunker.chunk_text(
        text=raw_text,
        source=source,
        doc_type=doc_type,
    )
    print(f"  Produced {len(chunks)} chunk(s) "
          f"(size={settings.RAG_CHUNK_SIZE}, overlap={settings.RAG_CHUNK_OVERLAP} tokens).")

    if not chunks:
        print(f"  [Skip] No chunks produced for {source}.")
        return 0

    # 3. Embed all chunks in one batch call (efficient)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_many(texts)
    print(f"  Embedded {len(embeddings)} chunk(s).")

    # 4. Prepare ChromaDB inputs
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{chunk_id_offset + i:06d}"
        ids.append(chunk_id)
        metadatas.append({
            "source": chunk["source"],
            "doc_type": chunk["doc_type"],
            "chunk_index": chunk["chunk_index"],
            "token_count": chunk["token_count"],
        })

    # 5. Store in ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"  Stored {len(chunks)} chunk(s) in ChromaDB.")

    return len(chunks)


# ------------------------------------------------------------------
# Ingestion summary — save to processed/ for documentation
# ------------------------------------------------------------------

def save_ingestion_report(report: dict) -> None:
    """
    Save a JSON summary of the ingestion run to knowledge_base/processed/.
    Useful for documenting exactly what was ingested — relevant for
    your evaluation chapter (traceability of knowledge sources).
    """
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report_path = settings.PROCESSED_DIR / "ingestion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Ingest] Report saved to: {report_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("SENTRY — Knowledge Base Ingestion")
    print("=" * 60)

    # Validate settings (checks API key etc.)
    # OPENAI not needed here but good habit to validate early
    if not settings.OPENAI_API_KEY:
        print("[Warning] OPENAI_API_KEY not set — ingestion will continue "
              "but the RAG pipeline will fail at query time.")

    # Initialise tools
    chunker = Chunker()
    embedder = Embedder()

    # Set up ChromaDB persistent client
    chroma_dir = Path(settings.CHROMA_PERSIST_DIR)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Ingest] ChromaDB persistence directory: {chroma_dir}")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = get_or_create_collection(client)

    # Discover documents
    documents = collect_documents()
    if not documents:
        print("\n[Ingest] No documents found. "
              "Check knowledge_base/raw/owasp/ and knowledge_base/raw/legal/")
        sys.exit(1)

    print(f"\n[Ingest] Total documents to process: {len(documents)}")

    # Ingest each document
    total_chunks = 0
    report_entries = []

    for doc in documents:
        chunks_added = ingest_document(
            doc=doc,
            chunker=chunker,
            embedder=embedder,
            collection=collection,
            chunk_id_offset=total_chunks,
        )
        total_chunks += chunks_added
        report_entries.append({
            "source": doc["source"],
            "doc_type": doc["doc_type"],
            "chunks_added": chunks_added,
        })

    # Final verification — query ChromaDB for actual stored count
    stored_count = collection.count()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Documents processed : {len(documents)}")
    print(f"  Total chunks stored : {stored_count}")
    print(f"  Collection name     : {settings.CHROMA_COLLECTION_NAME}")
    print(f"  Vector store path   : {settings.CHROMA_PERSIST_DIR}")
    print("=" * 60)

    # Save report
    save_ingestion_report({
        "collection": settings.CHROMA_COLLECTION_NAME,
        "total_documents": len(documents),
        "total_chunks": stored_count,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size_tokens": settings.RAG_CHUNK_SIZE,
        "chunk_overlap_tokens": settings.RAG_CHUNK_OVERLAP,
        "documents": report_entries,
    })


if __name__ == "__main__":
    main()