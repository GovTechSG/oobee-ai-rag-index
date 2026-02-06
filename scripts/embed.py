#!/usr/bin/env python3
"""
Embedding and vector DB operations for documentation indexing.
Chunks markdown files and upserts to Pinecone using integrated inference.
"""

import hashlib
import logging
import os
import re
import time
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
DEFAULT_HEADER_LEVEL = 2  # split sections on headings at this level or deeper

HEADER_RE = re.compile(r'^(#{1,6})\s+')
FENCE_RE = re.compile(r'^(`{3,}|~{3,})')


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
        header_level: int = DEFAULT_HEADER_LEVEL
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_level = max(1, header_level)

    def split_text(self, text: str) -> list[str]:
        """
        Split markdown into chunks by header level and size.
        Code blocks are kept intact (never split).
        """
        chunks: list[str] = []
        sections = self._split_by_header(text)
        for section in sections:
            chunks.extend(self._chunk_section(section))
        return [c for c in chunks if c.strip()]

    def _is_header(self, line: str) -> bool:
        match = HEADER_RE.match(line)
        if not match:
            return False
        return len(match.group(1)) >= self.header_level

    def _split_by_header(self, text: str) -> list[str]:
        """Split text into sections based on header level, ignoring headers inside code."""
        lines = text.splitlines()
        sections: list[str] = []
        current: list[str] = []
        in_code = False
        fence = None

        for line in lines:
            stripped = line.lstrip()
            fence_match = FENCE_RE.match(stripped)

            if not in_code and self._is_header(line):
                if current:
                    sections.append("\n".join(current).strip("\n"))
                current = [line]
            else:
                current.append(line)

            if fence_match:
                fence_marker = fence_match.group(1)
                if not in_code:
                    in_code = True
                    fence = fence_marker
                elif fence and stripped.startswith(fence):
                    in_code = False
                    fence = None

        if current:
            sections.append("\n".join(current).strip("\n"))

        return sections

    def _split_section_into_blocks(self, section: str) -> list[tuple[str, str]]:
        """Split a section into text and code blocks."""
        lines = section.splitlines()
        blocks: list[tuple[str, str]] = []
        buffer: list[str] = []
        in_code = False
        fence = None

        for line in lines:
            stripped = line.lstrip()
            fence_match = FENCE_RE.match(stripped)

            if not in_code and fence_match:
                if buffer:
                    blocks.append(("text", "\n".join(buffer)))
                    buffer = []
                in_code = True
                fence = fence_match.group(1)
                buffer.append(line)
                continue

            if in_code:
                buffer.append(line)
                if fence and stripped.startswith(fence):
                    blocks.append(("code", "\n".join(buffer)))
                    buffer = []
                    in_code = False
                    fence = None
                continue

            buffer.append(line)

        if buffer:
            blocks.append(("code" if in_code else "text", "\n".join(buffer)))

        return blocks

    def _split_text_block(self, text: str) -> list[str]:
        """Split a text block into chunk-sized pieces (paragraph/sentence aware)."""
        pieces: list[str] = []
        for para in re.split(r'\n\n+', text.strip()):
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                pieces.append(para)
                continue

            current: list[str] = []
            current_len = 0
            for sentence in re.split(r'(?<=[.!?])\s+', para):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > self.chunk_size:
                    if current:
                        pieces.append(" ".join(current))
                        current = []
                        current_len = 0
                    pieces.append(sentence)
                    continue
                sep = 1 if current else 0
                if current_len + sep + len(sentence) > self.chunk_size and current:
                    pieces.append(" ".join(current))
                    current = [sentence]
                    current_len = len(sentence)
                else:
                    current.append(sentence)
                    current_len += sep + len(sentence)
            if current:
                pieces.append(" ".join(current))

        return pieces

    def _chunk_section(self, section: str) -> list[str]:
        """Chunk a section while keeping code blocks intact."""
        blocks = self._split_section_into_blocks(section)
        chunks: list[str] = []

        current_parts: list[str] = []
        current_len = 0
        current_has_code = False
        last_text_piece = ""

        def flush(with_overlap: bool) -> None:
            nonlocal current_parts, current_len, current_has_code, last_text_piece
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip("\n"))
            overlap_text = ""
            if with_overlap and last_text_piece and self.chunk_overlap > 0:
                overlap_text = last_text_piece[-self.chunk_overlap:]
            current_parts = [overlap_text] if overlap_text else []
            current_len = len(overlap_text) if overlap_text else 0
            current_has_code = False
            if not overlap_text:
                last_text_piece = ""

        def append_piece(piece: str, is_code: bool) -> None:
            nonlocal current_parts, current_len, current_has_code, last_text_piece
            if not piece.strip():
                return
            if not is_code:
                piece = piece.strip()
            piece_len = len(piece)
            sep_len = 2 if current_parts else 0

            if current_parts and current_len + sep_len + piece_len <= self.chunk_size:
                current_parts.append(piece)
                current_len += sep_len + piece_len
                if is_code:
                    current_has_code = True
                else:
                    last_text_piece = piece
                return

            if is_code:
                flush(with_overlap=False)
                if piece_len > self.chunk_size:
                    chunks.append(piece)
                    return
                current_parts.append(piece)
                current_len = piece_len
                current_has_code = True
                return

            # text piece
            flush(with_overlap=(self.chunk_overlap > 0 and not current_has_code))
            if current_parts:
                # If overlap is too big, drop it
                sep_len = 2
                if current_len + sep_len + piece_len > self.chunk_size:
                    current_parts = []
                    current_len = 0
            current_parts.append(piece)
            current_len += (2 if current_len > 0 else 0) + piece_len
            last_text_piece = piece

        for block_type, block in blocks:
            if block_type == "code":
                append_piece(block, is_code=True)
                continue
            for piece in self._split_text_block(block):
                append_piece(piece, is_code=False)

        if current_parts:
            chunks.append("\n\n".join(current_parts).strip("\n"))

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
        batch_size: int = 96,
        max_retries: int = 5,
        base_delay: float = 1.0,
        batch_delay: float = 0.5
    ) -> int:
        """
        Upsert text records to Pinecone. Pinecone handles embedding via integrated inference.

        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for upsert
            max_retries: Maximum retry attempts for rate-limited requests
            base_delay: Base delay in seconds for exponential backoff
            batch_delay: Delay between batches to avoid rate limits

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
        total_batches = (len(records) + batch_size - 1) // batch_size

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Retry loop with exponential backoff
            for attempt in range(max_retries):
                try:
                    self.index.upsert_records(self.namespace, batch)
                    total_upserted += len(batch)
                    logger.debug(f"Upserted batch {batch_num}/{total_batches}")
                    break
                except Exception as e:
                    if self._is_rate_limited(e):
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limited on batch {batch_num}/{total_batches}, "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise
            else:
                # All retries exhausted
                raise RuntimeError(
                    f"Failed to upsert batch {batch_num} after {max_retries} retries"
                )

            # Delay between batches to avoid hitting rate limits
            if i + batch_size < len(records):
                time.sleep(batch_delay)

        return total_upserted

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        msg = str(exc).lower()
        return "429" in msg or "rate" in msg or "too many" in msg

    def delete_by_file(self, framework: str, file_path: str) -> None:
        """Delete all chunks for a specific file."""
        # Use delete with filter for metadata-based deletion
        try:
            self.index.delete(
                filter={
                    "framework": {"$eq": framework},
                    "file_path": {"$eq": file_path}
                },
                namespace=self.namespace
            )
            logger.debug(f"Deleted chunks for {framework}/{file_path}")
        except Exception as e:
            if self._is_namespace_missing(e):
                logger.info(
                    f"Namespace '{self.namespace}' not found; "
                    f"skip delete for {framework}/{file_path}"
                )
                return
            raise

    def delete_by_ids(self, chunk_ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if chunk_ids:
            try:
                self.index.delete(ids=chunk_ids, namespace=self.namespace)
                logger.debug(f"Deleted {len(chunk_ids)} chunks by ID")
            except Exception as e:
                if self._is_namespace_missing(e):
                    logger.info(
                        f"Namespace '{self.namespace}' not found; "
                        "skip delete by IDs"
                    )
                    return
                raise

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

    @staticmethod
    def _is_namespace_missing(exc: Exception) -> bool:
        """Best-effort check for namespace-not-found errors."""
        msg = str(exc).lower()
        return "namespace not found" in msg


class Embedder:
    """Main class for embedding documentation files using Pinecone integrated inference."""

    def __init__(
        self,
        index_name: str,
        namespace: str = "",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        header_level: int = DEFAULT_HEADER_LEVEL
    ):
        self.chunker = MarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            header_level=header_level
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
