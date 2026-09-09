#!/usr/bin/env python3
"""
Markdown chunking and untrusted-content sanitisation used by the local
precomputed-index build (``build_local_index.py``) and the sync/scrape
pipeline.

This module used to live inside ``embed.py`` when the corpus was pushed
to Pinecone. Pinecone has since been removed; the chunker stays because
``build_local_index.py`` still needs deterministic, code-fence-aware
splitting to produce the precomputed RAG index consumed by the VS Code
extension and oobee-desktop.
"""

import hashlib
import re
from dataclasses import dataclass


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_HEADER_LEVEL = 2

HEADER_RE = re.compile(r'^(#{1,6})\s+')
FENCE_RE = re.compile(r'^(`{3,}|~{3,})')

# Zero-width / invisible unicode characters commonly abused for hidden
# prompt-injection payloads inside otherwise-innocent-looking markdown.
_HIDDEN_UNICODE_RE = re.compile(
    r'[­᠎​-‏‪-‮⁠-⁤⁦-⁯﻿]'
)
# HTML comment blocks: also a common injection carrier since they render invisibly.
_HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)

# Well-known prompt-injection triggers that show up verbatim in community-editable
# doc corpora. We can't fully "neutralize" natural-language instructions, but
# defanging the classic phrasings meaningfully raises the bar for a drive-by
# attacker who lands a PR into an upstream docs repo. Any downstream LLM should
# still treat retrieved corpus text as untrusted content, which is why each
# embedded chunk is also wrapped with an UNTRUSTED-CONTENT guard marker below.
_INJECTION_TRIGGER_RE = re.compile(
    r'(?i)('
    r'ignore\s+(?:all\s+)?(?:previous|above|prior|earlier)\s+(?:instructions?|prompts?|rules?)'
    r'|disregard\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?)'
    r'|forget\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|prompts?)'
    r'|you\s+are\s+now\s+(?:a\s+)?(?:different|new|dan|unrestricted)'
    r'|act\s+as\s+(?:if\s+you\s+are\s+)?(?:dan|jailbroken|unrestricted)'
    r'|system\s*:\s*you\s+(?:are|must|will)'
    r'|<\|(?:im_start|im_end|system|assistant|user)\|>'
    r')'
)

# Guard markers wrap every chunk at build time so a downstream LLM prompt can
# unambiguously delimit retrieved doc content from operator instructions.
UNTRUSTED_CONTENT_BEGIN = '[BEGIN UNTRUSTED DOCUMENT CONTENT]'
UNTRUSTED_CONTENT_END = '[END UNTRUSTED DOCUMENT CONTENT]'


def wrap_untrusted(text: str) -> str:
    """Wrap a chunk's text with the sentinel markers.

    Applied uniformly to every chunk, so retrieval-vector distances are
    unaffected (the same fixed prefix/suffix appears on every stored
    document), while every retrieved chunk carries an unmistakable
    outer boundary that a downstream LLM prompt can key off.
    """
    return f"{UNTRUSTED_CONTENT_BEGIN}\n{text}\n{UNTRUSTED_CONTENT_END}"


# Cap per-file ingest size so a single massive doc can't dominate the corpus.
MAX_INGESTED_BYTES = 512 * 1024


def sanitize_ingested_content(content: str) -> str:
    """Strip hidden-payload vectors from third-party markdown before chunking.

    - Removes HTML comments and zero-width/bidi unicode (invisible carriers).
    - Defangs the classic prompt-injection phrasings so a drive-by upstream PR
      cannot embed "IGNORE ALL PREVIOUS INSTRUCTIONS ..." verbatim into a
      chunk that later gets retrieved into an LLM prompt.
    - Caps per-file size so one huge doc cannot dominate the corpus.

    This is not a substitute for treating retrieved corpus text as untrusted
    at prompt-construction time — callers should also wrap chunks with the
    UNTRUSTED_CONTENT_BEGIN / _END markers exported above.
    """
    if not content:
        return content
    cleaned = _HTML_COMMENT_RE.sub('', content)
    cleaned = _HIDDEN_UNICODE_RE.sub('', cleaned)
    cleaned = _INJECTION_TRIGGER_RE.sub(
        lambda m: f'[QUOTED-DOC-TEXT:{m.group(0)}]', cleaned
    )
    if len(cleaned.encode('utf-8', errors='ignore')) > MAX_INGESTED_BYTES:
        cleaned = cleaned.encode('utf-8', errors='ignore')[:MAX_INGESTED_BYTES].decode(
            'utf-8', errors='ignore'
        )
    return cleaned


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
        header_level: int = DEFAULT_HEADER_LEVEL,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_level = max(1, header_level)

    def split_text(self, text: str) -> list[str]:
        """Split markdown into chunks by header level and size. Code blocks are kept intact."""
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

            flush(with_overlap=(self.chunk_overlap > 0 and not current_has_code))
            if current_parts:
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
        source_url: str = "",
    ) -> list[Chunk]:
        """Chunk markdown content and create Chunk objects with metadata.

        Every returned chunk's ``text`` is wrapped with the UNTRUSTED_CONTENT
        sentinels so any downstream LLM prompt can key off the boundary
        without having to opt in.
        """
        chunks = []
        text_chunks = self.split_text(content)

        for i, text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(framework, file_path, i)
            chunk = Chunk(
                id=chunk_id,
                text=wrap_untrusted(text),
                metadata={
                    "framework": framework,
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "source_url": source_url,
                },
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, framework: str, file_path: str, index: int) -> str:
        content = f"{framework}:{file_path}:{index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
