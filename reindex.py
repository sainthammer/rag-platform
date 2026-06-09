#!/usr/bin/env python3
"""reindex.py — clear a vector collection and re-index documents from a directory.

Usage examples
--------------
# Chroma backend (local, no external service needed):
python reindex.py ./docs --backend chroma --collection my_docs

# Qdrant backend (Qdrant server must be running):
python reindex.py ./docs --backend qdrant --collection my_docs \\
    --qdrant-host localhost --qdrant-port 6333

Supported file types: .txt  .md  .pdf  .docx  .html

NOTE: The default embedding service is FakeEmbeddingService (deterministic,
for development / testing only).  Swap it for SentenceTransformersService or
OpenAIEmbeddingService in production by editing the `_build_embed_fn` helper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clear a vector collection and re-index documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("source_dir", type=Path, help="Directory containing documents")
    p.add_argument(
        "--backend",
        choices=["chroma", "qdrant"],
        default="chroma",
        help="Vector store backend",
    )
    p.add_argument("--collection", default="documents", help="Collection / index name")
    p.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters")
    p.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap in characters")
    p.add_argument(
        "--vector-size",
        type=int,
        default=8,
        help="Embedding dimensionality (must match the embedding service)",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Upsert batch size")

    # ChromaDB
    g_chroma = p.add_argument_group("ChromaDB options")
    g_chroma.add_argument("--chroma-dir", default="./chroma_data", help="Chroma persist path")

    # Qdrant
    g_qdrant = p.add_argument_group("Qdrant options")
    g_qdrant.add_argument("--qdrant-host", default="localhost")
    g_qdrant.add_argument("--qdrant-port", type=int, default=6333)

    # Embedding provider
    g_emb = p.add_argument_group("Embedding options")
    g_emb.add_argument(
        "--embedding-provider",
        choices=["fake", "ollama"],
        default="fake",
        help="Embedding provider (default: fake for dev, ollama for real usage)",
    )
    g_emb.add_argument("--ollama-url", default="http://localhost:11434/v1", help="Ollama base URL")
    g_emb.add_argument("--ollama-model", default="nomic-embed-text", help="Ollama embedding model")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_files(source_dir: Path) -> list[Path]:
    return sorted(
        f for f in source_dir.rglob("*") if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS
    )


def _build_embed_fn(args: argparse.Namespace):
    """Return an embed callable based on --embedding-provider."""
    if args.embedding_provider == "ollama":
        from embeddings.adapters import OllamaEmbeddingService

        svc = OllamaEmbeddingService(
            model=args.ollama_model,
            base_url=args.ollama_url,
            normalize=True,
        )
        return svc.embed

    from embeddings.adapters import FakeEmbeddingService

    svc = FakeEmbeddingService(size=args.vector_size, normalize=True)
    return svc.embed


def _clear_chroma(chroma_dir: str, collection: str) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(collection)
        print(f"  Cleared existing Chroma collection '{collection}'.")
    except Exception:
        pass  # collection did not exist


def _clear_qdrant(host: str, port: int, collection: str) -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(host=host, port=port)
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        client.delete_collection(collection)
        print(f"  Cleared existing Qdrant collection '{collection}'.")


def _build_store(args: argparse.Namespace, vectorizer: Any | None = None) -> Any:
    if args.backend == "chroma":
        from vector_store.adapters import ChromaDB

        return ChromaDB(args.collection, persist_directory=args.chroma_dir)

    from vector_store.adapters import QdrantVectorStore

    return QdrantVectorStore(
        collection=args.collection,
        vector_size=args.vector_size,
        vectorizer=vectorizer,
        host=args.qdrant_host,
        port=args.qdrant_port,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm is required: pip install tqdm", file=sys.stderr)
        return 1

    args = _parse_args(argv)

    if not args.source_dir.exists():
        print(f"Error: '{args.source_dir}' does not exist.", file=sys.stderr)
        return 1

    # 1. Collect files
    files = _collect_files(args.source_dir)
    if not files:
        print(
            f"No supported documents found in '{args.source_dir}'.\n"
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(files)} document(s) in '{args.source_dir}'.")

    # 2. Chunk all documents
    from chunking import ingest

    print("Chunking documents...")
    all_chunks = []
    for f in tqdm(files, desc="Loading", unit="file"):
        try:
            chunks = ingest(
                f,
                strategy="fixed",
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"  Warning: skipped '{f.name}': {exc}", file=sys.stderr)

    if not all_chunks:
        print("No chunks produced. Exiting.", file=sys.stderr)
        return 1

    print(f"Produced {len(all_chunks)} chunk(s) across all documents.")

    # 3. Build embedding function
    embed = _build_embed_fn(args)

    # 4. For Qdrant: fit BM25 vectorizer on the full corpus before clearing
    vectorizer = None
    if args.backend == "qdrant":
        from vector_store.bm25 import BM25SparseVectorizer

        print("Fitting BM25 vectorizer on full corpus...")
        vectorizer = BM25SparseVectorizer()
        vectorizer.fit([c.text for c in all_chunks])
        print(f"  Vocabulary size: {len(vectorizer._vocab)} terms.")

    # 5. Clear existing collection
    print(f"Clearing collection '{args.collection}' (backend={args.backend})...")
    if args.backend == "chroma":
        _clear_chroma(args.chroma_dir, args.collection)
    else:
        _clear_qdrant(args.qdrant_host, args.qdrant_port, args.collection)

    # 6. Build store (will create the collection fresh)
    store = _build_store(args, vectorizer=vectorizer)

    # 7. Index in batches with progress bar
    print(f"Indexing {len(all_chunks)} chunk(s) (batch_size={args.batch_size})...")
    batch_count = (len(all_chunks) + args.batch_size - 1) // args.batch_size

    for i in tqdm(
        range(0, len(all_chunks), args.batch_size),
        total=batch_count,
        desc="Indexing",
        unit="batch",
    ):
        batch = all_chunks[i : i + args.batch_size]
        store.add(
            ids=[c.id for c in batch],
            embeddings=[embed(c.text) for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )

    total = store.count()
    print(f"\nDone. Collection '{args.collection}' now contains {total} vector(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
