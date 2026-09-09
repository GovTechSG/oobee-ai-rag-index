# oobee-ai-rag-index

Scrape framework documentation from GitHub, chunk it, and publish a precomputed
local RAG index consumed by the Oobee VS Code extension and oobee-desktop.
This repo keeps a manifest of file hashes so each sync only updates what changed.

> **Note:** the corpus used to be pushed to Pinecone. Pinecone has been removed;
> the only retrieval path is now the precomputed `sentence-transformers/all-MiniLM-L6-v2`
> index built by `scripts/build_local_index.py` and released by
> `.github/workflows/release-docs-corpus.yml`.

## Flow

```
GitHub repos
    |
    v
scripts/scrape.py  -> docs/<framework>/*
    |
    v
scripts/sync.py (diff vs manifest.json)
    |
    v
[Weekly: open PR]
    |
    v
Human reviews & merges PR
    |
    v
release-docs-corpus.yml (on push to master)
    |
    v
scripts/build_local_index.py -> chunks.jsonl + vectors.bin + meta.json
    |
    v
GitHub Release (latest-precompute) — downstream consumers pull from here
```

## How the sync works

1. **Weekly scrape** (GitHub Action: `sync-docs.yml`)
   - Scrapes docs from all configured repos
   - Diffs against `manifest.json` to find new/modified/deleted files
   - Creates a `sync/YYYY-MM-DD` branch with the changes
   - Opens a PR with a summary (per-framework breakdown, file lists)
   - Closes any previously open sync PR
   - Tags: `synced/YYYY-MM-DD` + `latest-sync`

2. **Precomputed-index release** (GitHub Action: `release-docs-corpus.yml`)
   - Triggers on push to master when `docs/**`, `manifest.json`, `config.yaml`,
     `scripts/build_local_index.py`, or `scripts/build_wcag_index.py` change.
   - Also triggerable manually via `workflow_dispatch`.
   - Rebuilds the full precomputed index (`chunks.jsonl`, `vectors.bin`, `meta.json`)
     with `sentence-transformers/all-MiniLM-L6-v2`.
   - Publishes two archives to a `precompute/YYYY-MM-DD` GitHub Release:
     - `docs-precompute.zip` — full bundle (index + markdown)
     - `docs-index.zip` — index only (~73 MiB)
   - Force-pushes `latest-precompute` to the same commit.

## Tags

| Tag | Description |
|-----|-------------|
| `synced/YYYY-MM-DD` | Permanent — marks each weekly scrape |
| `latest-sync` | Floating — most recent scrape |
| `precompute/YYYY-MM-DD` | Permanent — marks each precomputed-index release |
| `latest-precompute` | Floating — most recent released index |

## What gets tracked

`manifest.json` stores:
- per-file SHA256 hash (change detection)
- last synced timestamp
- framework commit SHA
- `chunk_ids` — legacy field, unused since Pinecone removal; kept so old
  manifests still round-trip cleanly.

This is the state used to decide NEW / MODIFIED / DELETED / UNCHANGED.

## Setup

Requirements:
- Python 3.10+

Install (for scrape/sync only):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Building the precomputed index locally additionally needs:
```bash
pip install "numpy<2"
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
pip install sentence-transformers==2.7.0 beautifulsoup4 lxml
```

(These are installed by `release-docs-corpus.yml` at build time; they are not
in `requirements.txt` because they aren't needed for scrape/sync.)

## Configuration

Edit `config.yaml`:
- `sources`: GitHub repos + docs paths + extensions
- `embedding`: chunk size/overlap + header split level (read by
  `scripts/chunker.py`)

## Usage

Scrape only:
```bash
.venv/bin/python scripts/scrape.py
```

Dry-run diff (no manifest changes):
```bash
.venv/bin/python scripts/sync.py --dry-run
```

Sync (scrape + diff + write manifest):
```bash
.venv/bin/python scripts/sync.py
```

Sync a single framework:
```bash
.venv/bin/python scripts/sync.py -f react
```

Generate a JSON summary of changes:
```bash
.venv/bin/python scripts/sync.py --dry-run --json-summary summary.json
```

Build the precomputed index locally (requires torch + sentence-transformers):
```bash
.venv/bin/python scripts/build_local_index.py --docs-dir docs --out-dir index-build
```

## GitHub Actions

### Sync docs (weekly)

Runs every Sunday 2AM SGT. Creates a PR for review.

Manual trigger:
```bash
# Normal trigger
gh workflow run "Sync docs"

# Force re-sync (all files treated as new)
gh workflow run "Sync docs" -f force_resync=true
```

### Release precomputed RAG index

Triggers automatically when docs land on master. Also manually dispatchable:
```bash
gh workflow run "Release precomputed RAG index"
gh workflow run "Release precomputed RAG index" -f tag=custom/2026-01-01
```

## Chunking behavior

See `scripts/chunker.py`:
- Splits by markdown headings at `embedding.header_level` (default `##`).
- Chunks by character size (`embedding.chunk_size`).
- Fenced code blocks are kept intact (never split).
- Overlap applied only between text-only chunks.
- Third-party markdown is sanitised (HTML comments stripped, invisible
  unicode removed, classic prompt-injection triggers defanged).
- Every emitted chunk is wrapped with `[BEGIN UNTRUSTED DOCUMENT CONTENT]` /
  `[END UNTRUSTED DOCUMENT CONTENT]` sentinels so downstream LLM prompts can
  key off an unambiguous boundary between operator instructions and retrieved
  doc text.

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
  web/
    html/
    accessibility/
  wcag/
scripts/
  scrape.py                # fetch docs from GitHub
  sync.py                  # orchestrate scrape → diff → manifest update
  chunker.py               # markdown chunker + prompt-injection sanitiser
  build_local_index.py     # produce chunks.jsonl + vectors.bin + meta.json
  build_wcag_index.py      # WCAG / DSS / DETAILS.md corpus builder
  pr_summary.py            # generate PR body from sync summary
  manifest.py              # manifest read/write helpers
.github/workflows/
  sync-docs.yml            # weekly scrape + PR creation
  release-docs-corpus.yml  # rebuild + publish precomputed index on merge
```
