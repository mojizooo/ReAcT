"""File-backed storage for one recoverable experiment task."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import TaskState


class TaskStore:
    """Own the single state file, snapshots, and run-local artifact paths."""

    def __init__(self, run_dir: Path) -> None:
        """Point a store at an existing task directory without loading it yet."""
        self.run_dir = run_dir
        self.state_path = run_dir / "state.json"

    @classmethod
    def create(cls, run_dir: Path, state: TaskState) -> "TaskStore":
        """Create a new task directory and persist its initial untouched state."""
        store = cls(run_dir)
        if store.state_path.exists():
            raise FileExistsError(f"task state already exists: {store.state_path}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "snapshots").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        store._write_state(state)
        return store

    @classmethod
    def open(cls, run_dir: Path) -> "TaskStore":
        """Open a persisted task and fail clearly when its state file is absent."""
        store = cls(run_dir)
        if not store.state_path.is_file():
            raise FileNotFoundError(f"task state does not exist: {store.state_path}")
        return store

    def load(self) -> TaskState:
        """Load the current complete task state from its canonical JSON file."""
        return TaskState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def update(self, mutate: Callable[[TaskState], TaskState]) -> TaskState:
        """Persist a validated state transition with before and after snapshots."""
        current = self.load()
        index = self._next_snapshot_index()
        self._write_snapshot(index, "before", current)
        updated = mutate(current)
        self._write_state(updated)
        self._write_snapshot(index, "after", updated)
        return updated

    def artifact_path(self, relative_path: str) -> Path:
        """Return a run-local artifact path while forbidding directory traversal."""
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("artifact paths must be relative to the task directory")
        return self.run_dir / candidate

    def write_artifact_json(self, relative_path: str, payload: dict[str, Any]) -> str:
        """Write a JSON artifact and return its normalized run-relative reference."""
        path = self.artifact_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return path.relative_to(self.run_dir).as_posix()

    def write_artifact_text(self, relative_path: str, content: str) -> str:
        """Write one derived human-readable artifact under the same path safety rules."""
        path = self.artifact_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, content)
        return path.relative_to(self.run_dir).as_posix()

    def _next_snapshot_index(self) -> int:
        """Allocate a monotonically increasing snapshot number without extra state."""
        snapshot_dir = self.run_dir / "snapshots"
        return len(list(snapshot_dir.glob("*-before.json")))

    def _write_snapshot(self, index: int, moment: str, state: TaskState) -> None:
        """Write one full state backup around a transition."""
        name = f"{index:03d}-{moment}.json"
        self._atomic_write(self.run_dir / "snapshots" / name, state.model_dump_json(indent=2) + "\n")

    def _write_state(self, state: TaskState) -> None:
        """Write the canonical state atomically to avoid partial JSON after interruption."""
        self._atomic_write(self.state_path, state.model_dump_json(indent=2) + "\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Replace a file only after its full new content has been written locally."""
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
