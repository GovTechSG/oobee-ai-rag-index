# oobee-ai-rag-index

A precomputed documentation index for retrieval-augmented generation (RAG), used by Oobee's AI features to ground accessibility fix suggestions in official framework and WCAG documentation.

## Purpose

This repo scrapes documentation from upstream sources (React, Vue, Angular, MDN, TypeScript, WCAG), chunks and sanitises it, and publishes a ready-to-use local index as a GitHub Release. Downstream apps download the release instead of scraping and embedding docs themselves. A manifest of file hashes means each sync only picks up what changed.

## Audience

- **Oobee maintainers** who refresh the docs corpus or change how it is chunked and indexed.
- **Developers of downstream consumers:** the [Oobee Dev Suite](https://github.com/GovTechSG/oobee-dev-suite-vscode-oss) VS Code extension and [Oobee Desktop](https://github.com/GovTechSG/oobee-desktop).

End users of Oobee don't need this repo. Their apps fetch the index automatically.

## Usage

**Consume the index:** download `docs-index.zip` (index only) or `docs-precompute.zip` (index and markdown) from the [`latest-precompute`](https://github.com/GovTechSG/oobee-ai-rag-index/releases/tag/latest-precompute) release. Each archive contains `chunks.jsonl`, `vectors.bin` and `meta.json`.

**Refresh the corpus:** run the *Sync docs* workflow, review and merge the PR it opens, and the *Release precomputed RAG index* workflow publishes a new release. Details are below.

## Status

Active. The corpus is refreshed on demand (manual workflow trigger) and released on every merge to `master`. The repo carries the `govtech-active` lifecycle topic.

## Owner

Maintained by the Oobee team at [GovTech Singapore](https://www.tech.gov.sg/) (Government Technology Agency of Singapore). Report security issues as described in [SECURITY.md](SECURITY.md). Open other issues in this repo.

## Licence

The code in this repo is released under the [MIT Licence](LICENSE). Scraped documentation under `docs/` stays under the licence of its upstream project (see `config.yaml` for sources).

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
[Manual workflow trigger opens PR]
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

1. **Manually triggered scrape** (GitHub Action: `sync-docs.yml`)
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
| `synced/YYYY-MM-DD` | Permanent — marks each scrape |
| `latest-sync` | Floating — most recent scrape |
| `precompute/YYYY-MM-DD` | Permanent — marks each precomputed-index release |
| `latest-precompute` | Floating — most recent released index |

## What gets tracked

`manifest.json` stores:
- per-file SHA256 hash (change detection)
- last synced timestamp
- framework commit SHA

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

### Sync docs (manual)

Run on demand. Creates a PR for review.

Trigger:
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
  sync-docs.yml            # trigger scrape + PR creation
  release-docs-corpus.yml  # rebuild + publish precomputed index on merge
```
