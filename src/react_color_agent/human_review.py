"""Explicit human state transitions for a submitted unreachable application."""

from __future__ import annotations

import json
from pathlib import Path

from .models import LastAction, Observation, Phase
from .storage import TaskStore


def review_unreachable_application(
    run_dir: Path,
    decision: str,
    reason: str,
) -> object:
    """Approve or reject the current application without invoking the Agent."""
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("human review reason must be non-empty")
    if decision not in {"approve", "reject"}:
        raise ValueError("human unreachable decision must be approve or reject")
    store = TaskStore.open(run_dir)
    state = store.load()
    if state.phase is not Phase.AWAITING_HUMAN_REVIEW:
        raise ValueError("task is not awaiting human review")
    if not state.artifacts.unreachable_decision:
        raise ValueError("task has no submitted unreachable application")
    source_decision = json.loads(
        store.artifact_path(state.artifacts.unreachable_decision).read_text(encoding="utf-8")
    )
    if source_decision.get("decision_kind") != "unreachable_application":
        raise ValueError("the active decision is not an unreachable application")
    review_payload = {
        "schema_version": 1,
        "kind": "human_unreachable_review",
        "task_id": state.task_id,
        "round": state.round,
        "decision": decision,
        "reason": normalized_reason,
        "source_decision": state.artifacts.unreachable_decision,
        "source_decision_sha256": _sha256(source_decision),
    }
    artifact = store.write_artifact_json(
        f"artifacts/round-{state.round}/unreachable/human_review.json", review_payload
    )

    def mutate(current):
        artifacts = current.artifacts.model_copy(
            update={
                "human_unreachable_review": artifact,
                # The submitted decision remains in the review artifact and history;
                # clear active pointers so a rejection can resume ordinary design.
                "unreachable_draft": None,
                "unreachable_evaluation": None,
                "unreachable_critic_review": None,
            }
        )
        if decision == "approve":
            phase = Phase.STOPPED
            result = "human approved the bounded unreachable application"
            summary = f"Human approved unreachable application: {normalized_reason}"
        else:
            phase = Phase.DESIGNING
            result = None
            summary = f"Human rejected unreachable application; continue designing: {normalized_reason}"
        return current.model_copy(
            update={
                "phase": phase,
                "artifacts": artifacts,
                "injected_unreachable_critic_review": None,
                "history": [
                    *current.history,
                    Observation(
                        action="human_review_unreachable",
                        status="success",
                        summary=summary,
                        artifacts=[artifact],
                    ),
                ],
                "last_action": LastAction(name="human_review_unreachable", status="success"),
                "result": result,
            }
        )

    return store.update(mutate)


def _sha256(payload: object) -> str:
    """Hash one canonical JSON payload for review provenance."""
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
