"""Best-effort structured runtime trace recording for later evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceRecorder:
    """Append structured observations without participating in task execution.

    Recording is intentionally best effort.  A full disk, malformed value, or
    permission issue must never turn an experiment decision into a failure.
    """

    def __init__(self, run_dir: Path) -> None:
        """Use a sidecar path that is not referenced by ``TaskState``."""
        self.path = run_dir / "evaluation" / "runtime_trace.jsonl"

    def record(self, event: str, **fields: Any) -> None:
        """Append one JSON object, silently disabling collection on write errors."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **_safe_fields(fields),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except (OSError, TypeError, ValueError):
            # Evaluation is a side effect; it must not change scientific execution.
            return

    def read(self) -> list[dict[str, Any]]:
        """Read valid sidecar events, ignoring a truncated final line."""
        if not self.path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return events
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("event"), str):
                events.append(value)
        return events


_SECRET_TERMS = ("api_key", "apikey", "token", "secret", "password", "authorization", "cookie")
_USAGE_KEYS = {"token_usage", "prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens"}


def _safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Remove obvious secret-bearing fields and bound large trace values."""
    return {key: _safe_value(value, key=key.lower()) for key, value in fields.items()}


def _safe_value(value: Any, *, key: str = "") -> Any:
    """Convert common runtime values into JSON-safe, bounded structures."""
    if key not in _USAGE_KEYS and any(term in key for term in _SECRET_TERMS):
        return "[redacted]"
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe_value(v, key=str(k).lower()) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        values = [_safe_value(item) for item in value]
        return values if len(values) <= 64 else [*values[:64], f"...[{len(values) - 64} more items]"]
    return str(value)
