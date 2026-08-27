#!/usr/bin/env python3
"""
Build a self-contained, precomputed RAG index for the Oobee VS Code extension.

Walks docs/{frameworks,languages,web}/**, chunks each markdown with the same
MarkdownChunker used by the Pinecone pipeline, and embeds every chunk with
sentence-transformers/all-MiniLM-L6-v2 (the source of the Xenova ONNX model
transformers.js loads at query time in the extension).

Outputs, packed under --out-dir:
  chunks.jsonl   one JSON per line: {id, namespace, sourceFile, heading, text}
  vectors.bin    raw float32 array, row-major, count * dims
  meta.json      {model, dims, count, chunksSha256, vectorsSha256, commitSha, generatedAt}

The extension's ragIndex.ts loads these files verbatim (skipping its own
chunker + embed pass) when they are present under resources/index/. When
absent, it falls back to the current runtime chunk+embed path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import struct
import sys
from pathlib import Path

# Reuse the Pinecone pipeline's chunker so precomputed chunks match what
# search would produce today. Any drift here means query-time and
# build-time diverge silently.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import MarkdownChunker  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMS = 384
EMBED_BATCH_SIZE = 64


# Namespace conventions match ragIndex.ts:
#   frameworks/<fw>/**    -> framework:<fw>
#   languages/<lang>/**   -> lang:<lang>
#   web/<topic>/**        -> web:<topic>
# WCAG clause files (top-level "1.4.3-*.md") are not currently in docs/;
# they live in oobee-dev-suite-AI-pipeline. If they land here later, they
# should be added under docs/wcag/*.md and mapped to wcag:<clause>.
def walk_corpus(docs_dir: Path) -> list[tuple[Path, str]]:
    """Return [(absolute markdown path, namespace)]."""
    out: list[tuple[Path, str]] = []
    for group, prefix in (
        ("frameworks", "framework"),
        ("languages", "lang"),
        ("web", "web"),
    ):
        group_dir = docs_dir / group
        if not group_dir.is_dir():
            continue
        for topic_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            namespace = f"{prefix}:{topic_dir.name}"
            for md in sorted(topic_dir.rglob("*.md")):
                out.append((md, namespace))
            for mdx in sorted(topic_dir.rglob("*.mdx")):
                out.append((mdx, namespace))
    return out


HEADER_RE = None


def first_heading(text: str) -> str:
    """Best-effort heading for a chunk: the first non-blank line starting with #."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def chunk_id(namespace: str, source_file: str, index: int) -> str:
    """Match ragIndex.ts's id format so downstream code can rely on it."""
    return f"{namespace}::{Path(source_file).name}::{index}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--out-dir", default="index-build")
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", ""))
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not docs_dir.is_dir():
        raise SystemExit(f"docs directory not found: {docs_dir}")

    # sentence-transformers imports torch; import lazily so the top-level
    # module still loads in envs without torch (for --help etc.).
    from sentence_transformers import SentenceTransformer

    logger.info("Loading %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    if model.get_sentence_embedding_dimension() != EMBED_DIMS:
        raise SystemExit(
            f"Model dims {model.get_sentence_embedding_dimension()} != expected {EMBED_DIMS}"
        )

    chunker = MarkdownChunker()

    corpus = walk_corpus(docs_dir)
    logger.info("Walking corpus: %d files under %s", len(corpus), docs_dir)

    chunks_path = out_dir / "chunks.jsonl"
    vectors_path = out_dir / "vectors.bin"

    all_texts: list[str] = []
    chunk_records: list[dict] = []

    for md_path, namespace in corpus:
        try:
            content = md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-utf8 file: %s", md_path)
            continue

        pieces = chunker.split_text(content)
        rel = md_path.relative_to(docs_dir).as_posix()
        for i, text in enumerate(pieces):
            record = {
                "id": chunk_id(namespace, rel, i),
                "namespace": namespace,
                "sourceFile": rel,
                "heading": first_heading(text),
                "text": text,
            }
            chunk_records.append(record)
            all_texts.append(text)

    if not chunk_records:
        raise SystemExit("No chunks produced — corpus is empty.")

    logger.info(
        "Chunked %d files into %d chunks; embedding in batches of %d",
        len(corpus),
        len(chunk_records),
        EMBED_BATCH_SIZE,
    )

    with chunks_path.open("w", encoding="utf-8") as f:
        for record in chunk_records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    # Embed and stream to vectors.bin so peak memory stays bounded.
    total = len(all_texts)
    with vectors_path.open("wb") as vf:
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = all_texts[start : start + EMBED_BATCH_SIZE]
            vectors = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if vectors.shape[1] != EMBED_DIMS:
                raise SystemExit(
                    f"Vector width {vectors.shape[1]} != expected {EMBED_DIMS}"
                )
            vf.write(vectors.astype("float32").tobytes(order="C"))
            done = start + len(batch)
            if done % (EMBED_BATCH_SIZE * 8) == 0 or done == total:
                logger.info("  embedded %d / %d", done, total)

    expected_bytes = total * EMBED_DIMS * 4
    actual_bytes = vectors_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"vectors.bin size mismatch: {actual_bytes} != {expected_bytes}"
        )

    meta = {
        "model": MODEL_NAME,
        "dims": EMBED_DIMS,
        "count": total,
        "chunksSha256": sha256_file(chunks_path),
        "vectorsSha256": sha256_file(vectors_path),
        "commitSha": args.commit_sha,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "Wrote %s (chunks=%d, vectors=%.2f MiB)",
        out_dir,
        total,
        actual_bytes / (1024 * 1024),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
