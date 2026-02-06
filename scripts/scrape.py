#!/usr/bin/env python3
"""
Scraper for fetching documentation from GitHub repositories.
Clones repos, filters markdown files, and preserves folder structure.
"""

import argparse
import hashlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def clone_repo(repo_url: str, dest: Path) -> str:
    """Clone repository with shallow depth. Returns commit SHA."""
    logger.info(f"Cloning {repo_url}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
        capture_output=True,
        text=True
    )

    result = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def should_ignore(path: Path, ignore_patterns: list[str]) -> bool:
    """Check if path should be ignored based on patterns."""
    for pattern in ignore_patterns:
        if pattern in path.parts:
            return True
    return False


def find_markdown_files(
    source_dir: Path,
    extensions: list[str],
    ignore_patterns: list[str]
) -> list[Path]:
    """Find all markdown files matching extensions, excluding ignored paths."""
    files = []
    for ext in extensions:
        for file_path in source_dir.rglob(f"*{ext}"):
            if not should_ignore(file_path.relative_to(source_dir), ignore_patterns):
                files.append(file_path)
    return sorted(files)


def copy_files(
    files: list[Path],
    source_base: Path,
    dest_base: Path
) -> dict[str, str]:
    """
    Copy files preserving directory structure.
    Returns dict mapping relative path to content hash.
    """
    file_hashes = {}

    for file_path in files:
        rel_path = file_path.relative_to(source_base)
        dest_path = dest_base / rel_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)

        file_hash = compute_file_hash(file_path)
        file_hashes[str(rel_path)] = file_hash
        logger.debug(f"Copied: {rel_path}")

    return file_hashes


def scrape_source(
    name: str,
    config: dict,
    output_base: Path
) -> dict:
    """
    Scrape a single source repository.
    Returns metadata about scraped files.
    """
    repo_url = config["repo"]
    docs_path = config.get("docs_path", "")
    extensions = config.get("extensions", [".md"])
    ignore_patterns = config.get("ignore", [])

    with tempfile.TemporaryDirectory() as tmp_dir:
        clone_dir = Path(tmp_dir) / "repo"

        commit_sha = clone_repo(repo_url, clone_dir)
        logger.info(f"Cloned {name} at commit {commit_sha[:8]}")

        source_dir = clone_dir / docs_path if docs_path else clone_dir

        if not source_dir.exists():
            raise FileNotFoundError(
                f"docs_path '{docs_path}' not found in {repo_url}"
            )

        files = find_markdown_files(source_dir, extensions, ignore_patterns)
        logger.info(f"Found {len(files)} markdown files in {name}")

        dest_dir = output_base / name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)

        file_hashes = copy_files(files, source_dir, dest_dir)

    return {
        "repo": repo_url,
        "commit": commit_sha,
        "docs_path": docs_path,
        "file_count": len(files),
        "files": file_hashes
    }


def scrape_all(config_path: Path) -> dict:
    """Scrape all sources defined in config."""
    config = load_config(config_path)
    sources = config.get("sources", {})
    output_base = Path(config.get("output", {}).get("base_dir", "frameworks"))

    results = {}

    for name, source_config in sources.items():
        logger.info(f"Processing source: {name}")
        try:
            results[name] = scrape_source(name, source_config, output_base)
            logger.info(f"Successfully scraped {name}: {results[name]['file_count']} files")
        except Exception as e:
            logger.error(f"Failed to scrape {name}: {e}")
            results[name] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Scrape documentation from GitHub repositories"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        help="Scrape only this source (by name in config)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    config = load_config(args.config)

    if args.source:
        if args.source not in config.get("sources", {}):
            logger.error(f"Source '{args.source}' not found in config")
            return 1
        sources = {args.source: config["sources"][args.source]}
        config["sources"] = sources

    output_base = Path(config.get("output", {}).get("base_dir", "frameworks"))
    output_base.mkdir(parents=True, exist_ok=True)

    results = scrape_all(args.config)

    print("\n--- Scrape Summary ---")
    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: FAILED - {result['error']}")
        else:
            print(f"  {name}: {result['file_count']} files (commit: {result['commit'][:8]})")

    return 0


if __name__ == "__main__":
    exit(main())
