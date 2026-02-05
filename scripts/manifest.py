#!/usr/bin/env python3
"""
Manifest management for tracking file states and chunk IDs.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FileState:
    """State of a single file in the manifest."""
    content_hash: str
    chunk_ids: list[str] = field(default_factory=list)
    last_synced: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "content_hash": self.content_hash,
            "chunk_ids": self.chunk_ids,
            "last_synced": self.last_synced
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileState":
        return cls(
            content_hash=data["content_hash"],
            chunk_ids=data.get("chunk_ids", []),
            last_synced=data.get("last_synced")
        )


@dataclass
class FrameworkState:
    """State of a framework in the manifest."""
    commit: str
    files: dict[str, FileState] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "commit": self.commit,
            "files": {path: state.to_dict() for path, state in self.files.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FrameworkState":
        return cls(
            commit=data.get("commit", ""),
            files={
                path: FileState.from_dict(state)
                for path, state in data.get("files", {}).items()
            }
        )


class Manifest:
    """Manifest for tracking scraped documentation state."""

    VERSION = "1.0"

    def __init__(self, path: Path):
        self.path = path
        self.version = self.VERSION
        self.last_sync: Optional[str] = None
        self.frameworks: dict[str, FrameworkState] = {}

    def load(self) -> bool:
        """Load manifest from file. Returns True if loaded, False if new."""
        if not self.path.exists():
            logger.info("No existing manifest found, starting fresh")
            return False

        try:
            with open(self.path) as f:
                data = json.load(f)

            self.version = data.get("version", self.VERSION)
            self.last_sync = data.get("last_sync")
            self.frameworks = {
                name: FrameworkState.from_dict(state)
                for name, state in data.get("frameworks", {}).items()
            }
            logger.info(f"Loaded manifest with {len(self.frameworks)} frameworks")
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load manifest: {e}, starting fresh")
            return False

    def save(self) -> None:
        """Save manifest to file."""
        self.last_sync = datetime.now(timezone.utc).isoformat()

        data = {
            "version": self.version,
            "last_sync": self.last_sync,
            "frameworks": {
                name: state.to_dict()
                for name, state in self.frameworks.items()
            }
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved manifest to {self.path}")

    def get_framework(self, name: str) -> Optional[FrameworkState]:
        """Get framework state by name."""
        return self.frameworks.get(name)

    def set_framework(self, name: str, state: FrameworkState) -> None:
        """Set or update framework state."""
        self.frameworks[name] = state

    def get_file(self, framework: str, file_path: str) -> Optional[FileState]:
        """Get file state by framework and path."""
        fw = self.frameworks.get(framework)
        if fw:
            return fw.files.get(file_path)
        return None

    def set_file(
        self,
        framework: str,
        file_path: str,
        content_hash: str,
        chunk_ids: list[str]
    ) -> None:
        """Set or update file state."""
        if framework not in self.frameworks:
            self.frameworks[framework] = FrameworkState(commit="")

        self.frameworks[framework].files[file_path] = FileState(
            content_hash=content_hash,
            chunk_ids=chunk_ids,
            last_synced=datetime.now(timezone.utc).isoformat()
        )

    def remove_file(self, framework: str, file_path: str) -> Optional[list[str]]:
        """Remove file from manifest. Returns chunk_ids for deletion."""
        fw = self.frameworks.get(framework)
        if fw and file_path in fw.files:
            chunk_ids = fw.files[file_path].chunk_ids
            del fw.files[file_path]
            return chunk_ids
        return None

    def get_all_chunk_ids(self, framework: str) -> list[str]:
        """Get all chunk IDs for a framework."""
        fw = self.frameworks.get(framework)
        if not fw:
            return []
        return [
            chunk_id
            for file_state in fw.files.values()
            for chunk_id in file_state.chunk_ids
        ]
