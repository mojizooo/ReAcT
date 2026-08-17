"""Freeze the reproducibility protocol used by one Director runtime session."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.spectrum_tools import (
    ABSORPTION_WAVELENGTHS,
    DEFAULT_EMISSION_STEP_NM,
    EMISSION_END_NM,
    EMISSION_START_NM,
)

from ..models import TaskState
from .trace_recorder import _safe_value


def build_protocol_fingerprint(
    *,
    state: TaskState,
    tools: dict[str, Any],
    client: Any,
    max_steps: int,
    max_read_only_steps: int,
    max_index_queries: int,
) -> dict[str, Any]:
    """Build a credential-free canonical manifest and its aggregate SHA-256."""
    # Import lazily so the evaluation package also remains independently importable.
    from ..director.contracts import DIRECTOR_INSTRUCTION

    repository_root = Path(__file__).resolve().parents[3]
    skill_hashes = _registered_skill_hashes(client, repository_root)
    active_skill_hashes = {
        name: skill_hashes[name]
        for name in state.active_skills
        if name in skill_hashes
    }
    check_goal = tools.get("check_goal")
    constraint_path = repository_root / "src" / "constraints" / "constraint.md"
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "source_hashes": {
            "implementation_sha256": _tree_sha256(
                repository_root,
                sorted(repository_root.glob("src/**/*.py")),
            ),
            "prompt_sha256": _text_sha256(DIRECTOR_INSTRUCTION),
            "registered_skill_sha256": skill_hashes,
            "active_skill_sha256": active_skill_hashes,
        },
        "director": {
            "model": getattr(client, "model", None),
            "base_url_host": getattr(client, "base_url_host", None),
            "registered_tools": sorted(tools),
        },
        "task_contract": {
            "target_cie": list(state.target),
            "max_rounds": state.max_rounds,
            "goal_tolerance": getattr(check_goal, "tolerance", None),
        },
        "spectrum_contract": {
            "constraint_sha256": _file_sha256(constraint_path),
            "emission_start_nm": EMISSION_START_NM,
            "emission_end_nm": EMISSION_END_NM,
            "emission_step_nm": DEFAULT_EMISSION_STEP_NM,
            "emission_point_count": int(
                round((EMISSION_END_NM - EMISSION_START_NM) / DEFAULT_EMISSION_STEP_NM)
            )
            + 1,
            "absorption_start_nm": float(ABSORPTION_WAVELENGTHS[0]),
            "absorption_end_nm": float(ABSORPTION_WAVELENGTHS[-1]),
            "absorption_step_nm": -1.0,
            "absorption_point_count": len(ABSORPTION_WAVELENGTHS),
            "absorption_required": False,
        },
        "runtime_limits": {
            "max_steps": max_steps,
            "max_read_only_steps": max_read_only_steps,
            "max_index_queries": max_index_queries,
        },
    }
    # Reuse trace sanitization before hashing so the recorded object and digest agree.
    protocol = _safe_value(protocol)
    protocol["protocol_sha256"] = _canonical_sha256(protocol)
    return protocol


def _registered_skill_hashes(client: Any, repository_root: Path) -> dict[str, str]:
    """Hash the exact Skill files exposed by the current decision client."""
    registry = getattr(client, "skill_registry", None)
    skills = getattr(registry, "skills", None)
    if isinstance(skills, dict):
        result = {}
        for name, skill in sorted(skills.items()):
            digest = _file_sha256(Path(skill.path))
            if digest is not None:
                result[str(name)] = digest
        return result

    # Scripted or third-party clients may omit a registry; hash packaged Skills as fallback.
    result = {}
    for path in sorted((repository_root / "src" / "skills").glob("**/SKILL.md")):
        digest = _file_sha256(path)
        if digest is not None:
            result[path.parent.name.replace("_", "-")] = digest
    return result


def _tree_sha256(root: Path, files: list[Path]) -> str | None:
    """Hash relative paths and bytes so renames are visible in the implementation digest."""
    digest = hashlib.sha256()
    included = 0
    for path in files:
        try:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
            included += 1
        except OSError:
            continue
    return digest.hexdigest() if included else None


def _file_sha256(path: Path) -> str | None:
    """Hash one optional protocol file without failing the research runtime."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _text_sha256(value: str) -> str:
    """Hash the exact system instruction sent to the Director."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    """Hash one JSON value with stable Unicode-preserving serialization."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
