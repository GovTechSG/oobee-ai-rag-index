#!/usr/bin/env python3
"""
Sync orchestrator for documentation scraping.
Compares scraped files against manifest to determine what needs updating.

The vector-DB embed step used to live here; it has been removed along with
Pinecone. Downstream consumers now consume the precomputed index built by
``build_local_index.py`` and released by ``release-docs-corpus.yml``.
"""

import argparse
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from manifest import Manifest, FrameworkState, FileState
from scrape import scrape_source, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChangeType(Enum):
    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class FileChange:
    file_path: str
    change_type: ChangeType
    content_hash: str = ""

    def __post_init__(self):
        pass


@dataclass
class SyncResult:
    framework: str
    commit: str
    new_files: list[FileChange]
    modified_files: list[FileChange]
    deleted_files: list[FileChange]
    unchanged_count: int

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.modified_files or self.deleted_files)

    def summary(self) -> str:
        return (
            f"{self.framework}: "
            f"+{len(self.new_files)} new, "
            f"~{len(self.modified_files)} modified, "
            f"-{len(self.deleted_files)} deleted, "
            f"={self.unchanged_count} unchanged"
        )


def diff_framework(
    framework_name: str,
    scraped_files: dict[str, str],
    commit: str,
    manifest: Manifest,
) -> SyncResult:
    """Compare scraped files against manifest to determine changes."""
    new_files = []
    modified_files = []
    deleted_files = []
    unchanged_count = 0

    manifest_state = manifest.get_framework(framework_name)
    manifest_files = manifest_state.files if manifest_state else {}

    for file_path, content_hash in scraped_files.items():
        manifest_file = manifest_files.get(file_path)

        if manifest_file is None:
            new_files.append(FileChange(
                file_path=file_path,
                change_type=ChangeType.NEW,
                content_hash=content_hash,
            ))
        elif manifest_file.content_hash != content_hash:
            modified_files.append(FileChange(
                file_path=file_path,
                change_type=ChangeType.MODIFIED,
                content_hash=content_hash,
            ))
        else:
            unchanged_count += 1

    for file_path, _ in manifest_files.items():
        if file_path not in scraped_files:
            deleted_files.append(FileChange(
                file_path=file_path,
                change_type=ChangeType.DELETED,
            ))

    return SyncResult(
        framework=framework_name,
        commit=commit,
        new_files=new_files,
        modified_files=modified_files,
        deleted_files=deleted_files,
        unchanged_count=unchanged_count,
    )


def sync_framework(
    name: str,
    source_config: dict,
    output_base: Path,
    manifest: Manifest,
    dry_run: bool = False,
) -> SyncResult:
    """Sync a single framework: scrape and diff."""
    logger.info(f"Scraping {name}...")
    scrape_result = scrape_source(name, source_config, output_base)

    if "error" in scrape_result:
        raise RuntimeError(f"Scrape failed: {scrape_result['error']}")

    result = diff_framework(
        framework_name=name,
        scraped_files=scrape_result["files"],
        commit=scrape_result["commit"],
        manifest=manifest,
    )

    logger.info(result.summary())
    return result


def apply_changes(result: SyncResult, manifest: Manifest) -> None:
    """Apply changes to manifest (content-hash tracking only, no embed)."""
    framework = result.framework

    if manifest.get_framework(framework) is None:
        manifest.set_framework(framework, FrameworkState(commit=result.commit))
    else:
        manifest.frameworks[framework].commit = result.commit

    for change in result.deleted_files:
        logger.info(f"Removing {change.file_path}")
        manifest.remove_file(framework, change.file_path)

    for change in result.modified_files:
        manifest.set_file(framework, change.file_path, change.content_hash)
        logger.debug(f"Updated {change.file_path}")

    for change in result.new_files:
        manifest.set_file(framework, change.file_path, change.content_hash)
        logger.debug(f"Added {change.file_path}")


def sync_all(
    config_path: Path,
    manifest_path: Path,
    dry_run: bool = False,
    frameworks: list[str] = None,
) -> dict[str, SyncResult]:
    """Sync all configured sources."""
    config = load_config(config_path)
    sources = config.get("sources", {})
    output_base = Path(config.get("output", {}).get("base_dir", "frameworks"))

    if frameworks:
        sources = {k: v for k, v in sources.items() if k in frameworks}

    manifest = Manifest(manifest_path)
    manifest.load()

    results = {}

    for name, source_config in sources.items():
        try:
            source_output = Path(source_config["output_dir"]) if "output_dir" in source_config else output_base

            result = sync_framework(
                name=name,
                source_config=source_config,
                output_base=source_output,
                manifest=manifest,
                dry_run=dry_run,
            )
            results[name] = result

            if not dry_run:
                apply_changes(result, manifest)

        except Exception as e:
            logger.error(f"Failed to sync {name}: {e}")
            results[name] = None

    if not dry_run:
        manifest.save()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sync documentation from GitHub repositories"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "-m", "--manifest",
        type=Path,
        default=Path("manifest.json"),
        help="Path to manifest file (default: manifest.json)",
    )
    parser.add_argument(
        "-f", "--framework",
        type=str,
        action="append",
        dest="frameworks",
        help="Sync only these frameworks (can be repeated)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying manifest",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default=None,
        metavar="FILE",
        help="Write JSON summary of changes to FILE",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    results = sync_all(
        config_path=args.config,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        frameworks=args.frameworks,
    )

    if args.json_summary:
        import json
        from datetime import date

        summary = {
            "date": date.today().isoformat(),
            "frameworks": {},
            "totals": {"new": 0, "modified": 0, "deleted": 0},
        }
        for name, result in results.items():
            if result is None:
                summary["frameworks"][name] = {"status": "failed"}
            else:
                summary["frameworks"][name] = {
                    "commit": result.commit,
                    "new": [f.file_path for f in result.new_files],
                    "modified": [f.file_path for f in result.modified_files],
                    "deleted": [f.file_path for f in result.deleted_files],
                    "unchanged_count": result.unchanged_count,
                }
                summary["totals"]["new"] += len(result.new_files)
                summary["totals"]["modified"] += len(result.modified_files)
                summary["totals"]["deleted"] += len(result.deleted_files)

        with open(args.json_summary, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote summary to {args.json_summary}")

    print("\n" + "=" * 50)
    print("SYNC SUMMARY")
    print("=" * 50)

    total_new = 0
    total_modified = 0
    total_deleted = 0

    for name, result in results.items():
        if result is None:
            print(f"  {name}: FAILED")
        else:
            print(f"  {result.summary()}")
            total_new += len(result.new_files)
            total_modified += len(result.modified_files)
            total_deleted += len(result.deleted_files)

    print("-" * 50)
    print(f"  TOTAL: +{total_new} new, ~{total_modified} modified, -{total_deleted} deleted")

    if args.dry_run:
        print("\n  (dry run - no changes applied)")

    return 0


if __name__ == "__main__":
    exit(main())
