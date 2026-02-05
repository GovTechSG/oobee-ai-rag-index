#!/usr/bin/env python3
"""
Embedding and vector DB operations for documentation indexing.
Chunks markdown files, generates embeddings, and upserts to Pinecone.
"""

import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken
from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CHUNK_SIZE = 1000  # tokens
DEFAULT_CHUNK_OVERLAP = 200  # tokens
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    id: str
    text: str
    metadata: dict


class MarkdownChunker:
    """Chunks markdown content by headers and size."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        model: str = DEFAULT_EMBEDDING_MODEL
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks respecting token limits."""
        chunks = []

        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', text)

        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds chunk size, split it further
            if para_tokens > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)
                    if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        # Keep overlap
                        overlap_text = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_text] if overlap_text else []
                        current_tokens = self.count_tokens(overlap_text) if overlap_text else 0
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
            else:
                # Check if adding paragraph exceeds limit
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep some overlap
                    overlap_text = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = self.count_tokens(overlap_text) if overlap_text else 0

                current_chunk.append(para)
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def chunk_markdown(
        self,
        content: str,
        framework: str,
        file_path: str,
        file_hash: str,
        source_url: str = ""
    ) -> list[Chunk]:
        """
        Chunk markdown content and create Chunk objects with metadata.

        Args:
            content: Markdown content
            framework: Framework name (react, vue, etc.)
            file_path: Relative path to file
            file_hash: Content hash of the file
            source_url: URL to source on GitHub

        Returns:
            List of Chunk objects
        """
        chunks = []
        text_chunks = self.split_text(content)

        for i, text in enumerate(text_chunks):
            # Generate deterministic chunk ID
            chunk_id = self._generate_chunk_id(framework, file_path, i)

            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata={
                    "framework": framework,
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "source_url": source_url,
                    "token_count": self.count_tokens(text)
                }
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, framework: str, file_path: str, index: int) -> str:
        """Generate deterministic chunk ID."""
        content = f"{framework}:{file_path}:{index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class EmbeddingClient:
    """Client for generating embeddings."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    ):
        self.client = OpenAI()
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions
        )

        return [item.embedding for item in response.data]

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class PineconeIndex:
    """Wrapper for Pinecone index operations."""

    def __init__(self, index_name: str, namespace: str = ""):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 100
    ) -> int:
        """
        Upsert chunks with embeddings to Pinecone.

        Args:
            chunks: List of Chunk objects
            embeddings: Corresponding embeddings
            batch_size: Batch size for upsert

        Returns:
            Number of vectors upserted
        """
        vectors = [
            {
                "id": chunk.id,
                "values": embedding,
                "metadata": {**chunk.metadata, "text": chunk.text}
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            total_upserted += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}")

        return total_upserted

    def delete_by_file(self, framework: str, file_path: str) -> None:
        """Delete all chunks for a specific file."""
        self.index.delete(
            filter={
                "framework": {"$eq": framework},
                "file_path": {"$eq": file_path}
            },
            namespace=self.namespace
        )
        logger.debug(f"Deleted chunks for {framework}/{file_path}")

    def delete_by_ids(self, chunk_ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if chunk_ids:
            self.index.delete(ids=chunk_ids, namespace=self.namespace)
            logger.debug(f"Deleted {len(chunk_ids)} chunks by ID")

    def delete_framework(self, framework: str) -> None:
        """Delete all chunks for a framework."""
        self.index.delete(
            filter={"framework": {"$eq": framework}},
            namespace=self.namespace
        )
        logger.info(f"Deleted all chunks for {framework}")


class Embedder:
    """Main class for embedding documentation files."""

    def __init__(
        self,
        index_name: str,
        namespace: str = "",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    ):
        self.chunker = MarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model=embedding_model
        )
        self.embedding_client = EmbeddingClient(
            model=embedding_model,
            dimensions=embedding_dimensions
        )
        self.pinecone = PineconeIndex(index_name, namespace)

    def embed_file(
        self,
        framework: str,
        file_path: str,
        file_hash: str,
        content: str,
        source_url: str = ""
    ) -> list[str]:
        """
        Embed a single file and upsert to Pinecone.

        Args:
            framework: Framework name
            file_path: Relative path to file
            file_hash: Content hash
            content: File content
            source_url: GitHub URL

        Returns:
            List of chunk IDs that were created
        """
        # Chunk the content
        chunks = self.chunker.chunk_markdown(
            content=content,
            framework=framework,
            file_path=file_path,
            file_hash=file_hash,
            source_url=source_url
        )

        if not chunks:
            return []

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_client.embed(texts)

        # Upsert to Pinecone
        self.pinecone.upsert_chunks(chunks, embeddings)

        return [chunk.id for chunk in chunks]

    def delete_file(self, framework: str, file_path: str) -> None:
        """Delete all chunks for a file."""
        self.pinecone.delete_by_file(framework, file_path)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID."""
        self.pinecone.delete_by_ids(chunk_ids)


def create_embed_callback(
    embedder: Embedder,
    frameworks_dir: Path,
    repo_urls: dict[str, str]
):
    """
    Create a callback function for use with sync.apply_changes().

    Args:
        embedder: Embedder instance
        frameworks_dir: Base directory containing framework docs
        repo_urls: Dict mapping framework name to repo URL

    Returns:
        Callback function(framework, file_path, content_hash) -> list[chunk_ids]
    """
    def callback(framework: str, file_path: str, content_hash: str) -> list[str]:
        full_path = frameworks_dir / framework / file_path

        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            return []

        content = full_path.read_text(encoding="utf-8")

        # Build source URL
        repo_url = repo_urls.get(framework, "")
        source_url = ""
        if repo_url:
            # This is approximate - actual path in repo may differ
            source_url = f"{repo_url}/blob/main/{file_path}"

        chunk_ids = embedder.embed_file(
            framework=framework,
            file_path=file_path,
            file_hash=content_hash,
            content=content,
            source_url=source_url
        )

        logger.info(f"Embedded {file_path} -> {len(chunk_ids)} chunks")
        return chunk_ids

    return callback


if __name__ == "__main__":
    # Simple test
    import argparse

    parser = argparse.ArgumentParser(description="Test embedding a single file")
    parser.add_argument("file", type=Path, help="Markdown file to embed")
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--framework", default="test", help="Framework name")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}")
        exit(1)

    content = args.file.read_text()
    file_hash = hashlib.sha256(content.encode()).hexdigest()

    embedder = Embedder(index_name=args.index)
    chunk_ids = embedder.embed_file(
        framework=args.framework,
        file_path=str(args.file.name),
        file_hash=file_hash,
        content=content
    )

    print(f"Created {len(chunk_ids)} chunks: {chunk_ids}")
