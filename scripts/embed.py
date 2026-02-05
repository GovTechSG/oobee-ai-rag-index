#!/usr/bin/env python3
"""
Embedding and vector DB operations for documentation indexing.
Chunks markdown files and upserts to Pinecone using integrated inference.
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pinecone import Pinecone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CHUNK_SIZE = 500  # characters (simpler than tokens for Pinecone inference)
DEFAULT_CHUNK_OVERLAP = 100  # characters


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
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks respecting character limits."""
        chunks = []

        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', text)

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # If single paragraph exceeds chunk size, split it further
            if para_size > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    sent_size = len(sentence)
                    if current_size + sent_size > self.chunk_size and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        # Keep overlap
                        overlap_text = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_text] if overlap_text else []
                        current_size = len(overlap_text) if overlap_text else 0
                    current_chunk.append(sentence)
                    current_size += sent_size
            else:
                # Check if adding paragraph exceeds limit
                if current_size + para_size > self.chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep some overlap
                    overlap_text = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(overlap_text) if overlap_text else 0

                current_chunk.append(para)
                current_size += para_size

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
                    "source_url": source_url
                }
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, framework: str, file_path: str, index: int) -> str:
        """Generate deterministic chunk ID."""
        content = f"{framework}:{file_path}:{index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class PineconeInferenceIndex:
    """
    Wrapper for Pinecone index with integrated inference.
    Uses Pinecone's hosted embedding models - no OpenAI needed.
    """

    def __init__(self, index_name: str, namespace: str = ""):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        # Pinecone API 2025-04 requires "__default__" instead of empty string
        self.namespace = namespace if namespace else "__default__"
        self.index_name = index_name

    def upsert_records(
        self,
        chunks: list[Chunk],
        batch_size: int = 96
    ) -> int:
        """
        Upsert text records to Pinecone. Pinecone handles embedding via integrated inference.

        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for upsert

        Returns:
            Number of records upserted
        """
        # Format records for Pinecone integrated inference
        # The "text" field will be automatically embedded by Pinecone
        records = [
            {
                "_id": chunk.id,
                "text": chunk.text,  # This gets embedded by Pinecone
                **chunk.metadata
            }
            for chunk in chunks
        ]

        total_upserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert_records(self.namespace, batch)
            total_upserted += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}")

        return total_upserted

    def delete_by_file(self, framework: str, file_path: str) -> None:
        """Delete all chunks for a specific file."""
        # Use delete with filter for metadata-based deletion
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

    def search(self, query: str, top_k: int = 10, framework: str = None) -> list[dict]:
        """
        Search for similar documents using integrated inference.

        Args:
            query: Search query text
            top_k: Number of results
            framework: Optional filter by framework

        Returns:
            List of matching records
        """
        search_params = {
            "namespace": self.namespace,
            "query": {
                "inputs": {"text": query},
                "top_k": top_k
            }
        }

        if framework:
            search_params["query"]["filter"] = {"framework": {"$eq": framework}}

        results = self.index.search_records(**search_params)

        # Handle different response formats from Pinecone SDK
        hits = []
        if hasattr(results, 'result') and hasattr(results.result, 'hits'):
            hits = results.result.hits
        elif hasattr(results, 'result') and isinstance(results.result, dict):
            hits = results.result.get("hits", [])
        elif isinstance(results, dict):
            hits = results.get("result", {}).get("hits", [])

        # Convert hits to dicts
        parsed = []
        for hit in hits:
            if hasattr(hit, '_id'):
                # SDK object
                parsed.append({
                    '_id': hit._id,
                    '_score': getattr(hit, '_score', None),
                    'fields': dict(hit.fields) if hasattr(hit, 'fields') else {}
                })
            elif isinstance(hit, dict):
                parsed.append(hit)
            else:
                # Try to convert
                parsed.append({'raw': str(hit)})

        return parsed


class Embedder:
    """Main class for embedding documentation files using Pinecone integrated inference."""

    def __init__(
        self,
        index_name: str,
        namespace: str = "",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        self.chunker = MarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.pinecone = PineconeInferenceIndex(index_name, namespace)

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

        # Upsert to Pinecone (embedding happens automatically)
        self.pinecone.upsert_records(chunks)

        return [chunk.id for chunk in chunks]

    def delete_file(self, framework: str, file_path: str) -> None:
        """Delete all chunks for a file."""
        self.pinecone.delete_by_file(framework, file_path)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID."""
        self.pinecone.delete_by_ids(chunk_ids)

    def search(self, query: str, top_k: int = 10, framework: str = None) -> list[dict]:
        """Search for similar documents."""
        return self.pinecone.search(query, top_k, framework)


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
    # Simple test / search CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Search or test embedding")
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--framework", type=str, help="Filter by framework")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--debug", action="store_true", help="Show raw response")
    args = parser.parse_args()

    embedder = Embedder(index_name=args.index)

    if args.search:
        results = embedder.search(args.search, args.top_k, args.framework)

        if args.debug:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"\nFound {len(results)} results:\n")
            for r in results:
                # Handle different response structures
                fields = r.get('fields', r)
                score = r.get('_score', r.get('score', 'N/A'))
                framework = fields.get('framework', 'unknown')
                file_path = fields.get('file_path', 'unknown')
                text = fields.get('text', '')[:200]

                print(f"- [{framework}/{file_path}]")
                print(f"  Score: {score}")
                print(f"  {text}...")
                print()
