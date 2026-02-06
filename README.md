# oobee-ai-rag-index

Scrape framework documentation from GitHub, chunk it, and index it in Pinecone for RAG workflows. This repo keeps a manifest of file hashes and chunk IDs so each sync only updates what changed.

## Flow

```
GitHub repos
    |
    v
scripts/scrape.py  -> frameworks/<framework>/*
    |
    v
scripts/sync.py (diff vs manifest.json)
    |
    v
scripts/embed.py -> Pinecone index (per-framework namespaces)
```

## What gets tracked

`manifest.json` stores:
- per-file SHA256 hash (change detection)
- chunk IDs (for targeted deletions)
- last synced timestamp
- framework commit SHA

This is the state used to decide NEW / MODIFIED / DELETED / UNCHANGED.

## Setup

Requirements:
- Python 3.10+
- Pinecone index (with integrated inference)

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set env vars:
```bash
export PINECONE_API_KEY="..."
export PINECONE_INDEX_NAME="docs-rag"
```

## Configuration

Edit `config.yaml`:
- `sources`: GitHub repos + docs paths + extensions
- `embedding`: chunk size/overlap + header split level
- `vector_db`: Pinecone settings (namespace template optional)

Namespace template example:
```yaml
vector_db:
  namespace_template: "{framework}-docs"
```

## Usage

Scrape only (no embedding):
```bash
.venv/bin/python scripts/scrape.py
```

Dry-run diff (no changes applied):
```bash
.venv/bin/python scripts/sync.py --dry-run
```

Sync + embed (all frameworks):
```bash
.venv/bin/python scripts/sync.py --embed
```

Sync + embed (single framework):
```bash
.venv/bin/python scripts/sync.py --embed -f react
```

## Chunking behavior

Current chunker (see `scripts/embed.py`):
- Splits by markdown headings at `embedding.header_level` (default `##`).
- Chunks by character size (`embedding.chunk_size`).
- Fenced code blocks are kept intact (never split).
- Overlap is applied only between text-only chunks.

## GitHub Actions

The workflow runs on a schedule and can be triggered manually. It:
1) clones docs
2) embeds updated files
3) commits `manifest.json` and `frameworks/`

Secrets required:
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

Workflow file: `.github/workflows/sync-docs.yml`.

## Repo layout

```
config.yaml
manifest.json
frameworks/
scripts/
  scrape.py
  sync.py
  embed.py
  manifest.py
```
