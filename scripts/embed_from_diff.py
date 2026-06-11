#!/usr/bin/env python3
"""
Embed documentation changes by comparing two manifest states.
Used by the embed-on-merge workflow to process only the delta.
"""

import argparse
import logging
import os
from pathlib import Path

import yaml

from manifest import Manifest, FrameworkState, FileState
from embed import Embedder, create_embed_callback
from sync import SyncResult, FileChange, ChangeType, apply_changes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def diff_manifests(old_manifest: Manifest, new_manifest: Manifest) -> dict[str, SyncResult]:
    """Compare two manifests and return changes per framework."""
    results = {}

    all_frameworks = set(old_manifest.frameworks.keys()) | set(new_manifest.frameworks.keys())

    for fw_name in all_frameworks:
        old_fw = old_manifest.get_framework(fw_name)
        new_fw = new_manifest.get_framework(fw_name)

        old_files = old_fw.files if old_fw else {}
        new_files = new_fw.files if new_fw else {}

        new_changes = []
        modified_changes = []
        deleted_changes = []
        unchanged = 0

        for path, new_state in new_files.items():
            old_state = old_files.get(path)
            if old_state is None:
                new_changes.append(FileChange(
                    file_path=path,
                    change_type=ChangeType.NEW,
                    content_hash=new_state.content_hash
                ))
            elif old_state.content_hash != new_state.content_hash:
                modified_changes.append(FileChange(
                    file_path=path,
                    change_type=ChangeType.MODIFIED,
                    content_hash=new_state.content_hash,
                    old_chunk_ids=old_state.chunk_ids
                ))
            else:
                unchanged += 1

        for path, old_state in old_files.items():
            if path not in new_files:
                deleted_changes.append(FileChange(
                    file_path=path,
                    change_type=ChangeType.DELETED,
                    old_chunk_ids=old_state.chunk_ids
                ))

        commit = new_fw.commit if new_fw else ""
        results[fw_name] = SyncResult(
            framework=fw_name,
            commit=commit,
            new_files=new_changes,
            modified_files=modified_changes,
            deleted_files=deleted_changes,
            unchanged_count=unchanged
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Embed documentation changes by comparing two manifests"
    )
    parser.add_argument(
        "--old-manifest", type=Path, required=True,
        help="Path to the pre-merge manifest"
    )
    parser.add_argument(
        "--new-manifest", type=Path, required=True,
        help="Path to the post-merge manifest"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--embed", action="store_true",
        help="Actually embed to Pinecone (otherwise dry-run)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Clear namespaces before embedding"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    sources = config.get("sources", {})
    output_base = Path(config.get("output", {}).get("base_dir", "docs/frameworks"))
    embedding_config = config.get("embedding", {})
    vector_db_config = config.get("vector_db", {})

    # Load both manifests
    old_manifest = Manifest(args.old_manifest)
    old_manifest.load()

    new_manifest = Manifest(args.new_manifest)
    new_manifest.load()

    # Compute diff
    results = diff_manifests(old_manifest, new_manifest)

    # Print summary
    total_new = 0
    total_mod = 0
    total_del = 0
    for name, result in results.items():
        logger.info(result.summary())
        total_new += len(result.new_files)
        total_mod += len(result.modified_files)
        total_del += len(result.deleted_files)

    if total_new == 0 and total_mod == 0 and total_del == 0:
        logger.info("No changes to embed.")
        return 0

    if not args.embed:
        logger.info("Dry run — no embedding performed.")
        return 0

    # Set up embedding
    index_name = (
        os.environ.get("PINECONE_INDEX_NAME")
        or os.environ.get("VECTOR_DB_INDEX_NAME")
        or vector_db_config.get("index_name")
    )
    if not index_name:
        raise ValueError(
            "PINECONE_INDEX_NAME or VECTOR_DB_INDEX_NAME must be set"
        )

    namespace_template = vector_db_config.get("namespace_template")
    default_namespace = vector_db_config.get("namespace", "")
    repo_urls = {name: cfg.get("repo", "") for name, cfg in sources.items()}

    # Process each framework
    for fw_name, result in results.items():
        if not result.has_changes:
            continue

        source_config = sources.get(fw_name, {})
        source_output = Path(source_config["output_dir"]) if "output_dir" in source_config else output_base

        namespace = (
            namespace_template.format(framework=fw_name)
            if namespace_template
            else default_namespace
        )

        embedder = Embedder(
            index_name=index_name,
            namespace=namespace,
            chunk_size=embedding_config.get("chunk_size", 500),
            chunk_overlap=embedding_config.get("chunk_overlap", 100),
            header_level=embedding_config.get("header_level", 2)
        )

        if args.force:
            logger.info(f"Force: clearing namespace '{namespace or '__default__'}' for {fw_name}")
            embedder.clear_namespace()

        embed_callback = create_embed_callback(embedder, source_output, repo_urls)
        delete_callback = lambda fw, fp, _embedder=embedder: _embedder.delete_file(fw, fp)

        logger.info(f"Embedding changes for {fw_name} in namespace '{namespace or '__default__'}'")
        apply_changes(result, new_manifest, embed_callback, delete_callback)

    # Save updated manifest (now with chunk_ids populated)
    new_manifest.save()
    logger.info(f"Saved updated manifest to {args.new_manifest}")

    print(f"\nDone: +{total_new} new, ~{total_mod} modified, -{total_del} deleted")
    return 0


if __name__ == "__main__":
    exit(main())
