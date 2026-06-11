# oobee-ai-rag-index

Scrape framework documentation from GitHub, chunk it, and index it in Pinecone for RAG workflows. This repo keeps a manifest of file hashes and chunk IDs so each sync only updates what changed.

## Flow

```
GitHub repos
    |
    v
scripts/scrape.py  -> docs/<framework>/*
    |
    v
scripts/sync.py (diff vs manifest.json)
    |                               |
    v                               v
[Weekly: open PR]           [On merge: embed]
    |                               |
    v                               v
Human reviews PR            scripts/embed_from_diff.py -> Pinecone
```

## How the sync works

The sync is a two-phase process with a human review gate:

1. **Weekly scrape** (GitHub Action: `sync-docs.yml`)
   - Scrapes docs from all configured repos
   - Diffs against `manifest.json` to find new/modified/deleted files
   - Creates a `sync/YYYY-MM-DD` branch with the changes
   - Opens a PR with a summary (per-framework breakdown, estimated chunks, file lists)
   - Closes any previously open sync PR (keeps the branch for history)
   - Tags: `synced/YYYY-MM-DD` + `latest-sync`

2. **Embed on merge** (GitHub Action: `embed-on-merge.yml`)
   - Triggers automatically when a `sync/*` PR is merged to master
   - Compares old manifest (pre-merge) vs new manifest (post-merge)
   - Embeds only the changed files to Pinecone
   - Commits updated manifest with populated chunk IDs
   - Tags: `embedded/YYYY-MM-DD` + `latest-embedded`

## Tags

| Tag | Description |
|-----|-------------|
| `synced/YYYY-MM-DD` | Permanent — marks each weekly scrape |
| `latest-sync` | Floating — most recent scrape |
| `embedded/YYYY-MM-DD` | Permanent — marks each successful embed |
| `latest-embedded` | Floating — most recent embed (what's in Pinecone) |

Useful commands:
```bash
# What's synced but not yet embedded?
git diff latest-embedded latest-sync -- docs/

# What changed between two embed cycles?
git diff embedded/2026-06-01 embedded/2026-06-08 -- docs/

# What's currently in Pinecone?
git show latest-embedded:manifest.json
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

Sync + embed (all frameworks, local):
```bash
.venv/bin/python scripts/sync.py --embed
```

Sync + embed (single framework):
```bash
.venv/bin/python scripts/sync.py --embed -f react
```

Generate a JSON summary of changes:
```bash
.venv/bin/python scripts/sync.py --dry-run --json-summary summary.json
```

## GitHub Actions

### Sync docs (weekly)

Runs every Sunday 2AM SGT. Creates a PR for review.

Manual trigger options:
- `force_resync` — clears manifest, treats all files as new
- `skip_pr` — emergency bypass, embeds directly to Pinecone without PR

```bash
# Normal trigger
gh workflow run "Sync docs"

# Force re-sync (all files re-embedded after merge)
gh workflow run "Sync docs" -f force_resync=true

# Emergency: skip PR and embed directly
gh workflow run "Sync docs" -f skip_pr=true
```

### Embed on merge

Triggers automatically when a `sync/*` PR is merged. Can also be re-triggered manually:

```bash
gh workflow run "Embed on merge"
```

### Secrets required

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

## Chunking behavior

Current chunker (see `scripts/embed.py`):
- Splits by markdown headings at `embedding.header_level` (default `##`).
- Chunks by character size (`embedding.chunk_size`).
- Fenced code blocks are kept intact (never split).
- Overlap is applied only between text-only chunks.

## Repo layout

```
config.yaml
manifest.json
docs/
  frameworks/
    react/
    vue/
    angular/
  languages/
    javascript/
    typescript/
scripts/
  scrape.py            # fetch docs from GitHub
  sync.py              # orchestrate scrape → diff → embed
  embed.py             # chunk + embed + upsert to Pinecone
  embed_from_diff.py   # embed based on manifest diff (used by CI)
  pr_summary.py        # generate PR body from sync summary
  manifest.py          # manifest read/write helpers
.github/workflows/
  sync-docs.yml        # weekly scrape + PR creation
  embed-on-merge.yml   # embed on PR merge
```
