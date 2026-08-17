"""Pure contracts for evidence-based unreachable applications and reviews."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import TaskState


def validate_unreachable_draft(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one bounded Director claim without deciding scientific sufficiency."""
    evidence_ids = _sample_ids(payload.get("evidence_sample_ids"), "evidence_sample_ids")
    attempted_routes = _evidence_items(
        payload.get("attempted_routes"),
        field="attempted_routes",
        text_fields=("route", "observed_outcome", "reason"),
        allowed_evidence_ids=set(evidence_ids),
    )
    remaining_options = _evidence_items(
        payload.get("remaining_options"),
        field="remaining_options",
        text_fields=("option", "reason"),
        allowed_evidence_ids=set(evidence_ids),
    )
    uncertainties = _text_list(payload.get("uncertainties"), "uncertainties")
    return {
        "claim": _text(payload.get("claim"), "claim"),
        "evidence_sample_ids": evidence_ids,
        "attempted_routes": attempted_routes,
        "remaining_options": remaining_options,
        "reasoning": _text(payload.get("reasoning"), "reasoning"),
        "uncertainties": uncertainties,
    }


def validate_unreachable_responses(
    review: Mapping[str, Any], responses: Any
) -> list[dict[str, str]]:
    """Require one explicit Director disposition for every Critic finding."""
    if not isinstance(responses, list) or not all(isinstance(item, Mapping) for item in responses):
        raise ValueError("critic_responses must be a list of objects")
    normalized = [
        {
            "finding_id": _text(item.get("finding_id"), "critic response finding_id"),
            "disposition": _text(item.get("disposition"), "critic response disposition"),
            "response": _text(item.get("response"), "critic response"),
        }
        for item in responses
    ]
    allowed = {"accepted", "partially_accepted", "rejected"}
    if any(item["disposition"] not in allowed for item in normalized):
        raise ValueError(
            "critic response disposition must be accepted, partially_accepted, or rejected"
        )
    finding_ids = [str(item.get("finding_id", "")).strip() for item in review.get("findings", [])]
    response_ids = [item["finding_id"] for item in normalized]
    if len(response_ids) != len(set(response_ids)) or set(response_ids) != set(finding_ids):
        raise ValueError("Director must respond exactly once to every unreachable Critic finding")
    return normalized


def validate_unreachable_review(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the dedicated one-shot Critic response before it becomes durable."""
    recommendation = str(payload.get("recommendation", "")).strip()
    allowed_recommendations = {"support_application", "revise", "continue_experiments"}
    if recommendation not in allowed_recommendations:
        raise ValueError(
            "unreachable Critic recommendation must be support_application, revise, or continue_experiments"
        )
    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list) or not all(
        isinstance(item, Mapping) for item in raw_findings
    ):
        raise ValueError("unreachable Critic findings must be a list")
    findings: list[dict[str, Any]] = []
    for item in raw_findings:
        severity = _text(item.get("severity"), "critic finding severity")
        if severity not in {"critical", "warning"}:
            raise ValueError("unreachable Critic finding severity must be critical or warning")
        findings.append(
            {
                "finding_id": _text(item.get("finding_id"), "critic finding_id"),
                "severity": severity,
                "category": _text(item.get("category"), "critic finding category"),
                "claim": _text(item.get("claim"), "critic finding claim"),
                "evidence_sample_ids": _optional_sample_ids(item.get("evidence_sample_ids")),
                "reason": _text(item.get("reason"), "critic finding reason"),
                "suggestion": _text(item.get("suggestion"), "critic finding suggestion"),
            }
        )
    finding_ids = [item["finding_id"] for item in findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ValueError("unreachable Critic finding IDs must be unique")
    return {
        "recommendation": recommendation,
        "findings": findings,
        "acceptable_aspects": _optional_text_list(payload.get("acceptable_aspects")),
        "unresolved_uncertainties": _optional_text_list(
            payload.get("unresolved_uncertainties")
        ),
    }


def unreachable_critic_packet(
    state: TaskState,
    draft: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    measured_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact facts-first packet without raw spectrum arrays."""
    material_slots = [
        {"slot": chr(ord("A") + index), "material": material}
        for index, material in enumerate(state.materials)
    ]
    # Copy through JSON so no later caller mutation can change the review packet.
    return json.loads(
        json.dumps(
            {
                "review_kind": "unreachable_application",
                "task": {
                    "task_id": state.task_id,
                    "target_cie": list(state.target),
                    "completed_round": state.round,
                    "max_rounds": state.max_rounds,
                    "remaining_round_budget": max(0, state.max_rounds - state.round),
                },
                "material_slots": material_slots,
                "measured_history_summary": state.injected_research_briefing,
                "director_cited_measured_records": list(measured_records),
                "director_draft": dict(draft),
                "deterministic_evidence_evaluation": dict(evaluation),
                "authority_boundary": (
                    "The Critic is advisory. The Director may continue or submit, and only a human "
                    "approval can stop the task early."
                ),
            },
            ensure_ascii=False,
        )
    )


def _evidence_items(
    raw_items: Any,
    *,
    field: str,
    text_fields: tuple[str, ...],
    allowed_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    """Validate route-like objects that cite only top-level measured evidence."""
    if not isinstance(raw_items, list) or not raw_items or not all(
        isinstance(item, Mapping) for item in raw_items
    ):
        raise ValueError(f"{field} must be a non-empty list of objects")
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        evidence_ids = _sample_ids(item.get("evidence_sample_ids"), f"{field} evidence_sample_ids")
        unknown = [sample_id for sample_id in evidence_ids if sample_id not in allowed_evidence_ids]
        if unknown:
            raise ValueError(
                f"{field} evidence_sample_ids must also appear in top-level evidence_sample_ids: "
                + ", ".join(unknown)
            )
        normalized.append(
            {
                **{name: _text(item.get(name), f"{field} {name}") for name in text_fields},
                "evidence_sample_ids": evidence_ids,
            }
        )
    return normalized


def _sample_ids(value: Any, field: str) -> list[str]:
    """Return a non-empty, duplicate-free list of sample IDs."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = [_text(item, field) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _optional_sample_ids(value: Any) -> list[str]:
    """Normalize a Critic evidence list that may legitimately be empty."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("critic evidence_sample_ids must be a list")
    result = [_text(item, "critic evidence_sample_id") for item in value]
    if len(result) != len(set(result)):
        raise ValueError("critic evidence_sample_ids must not contain duplicates")
    return result


def _text_list(value: Any, field: str) -> list[str]:
    """Return a required list of non-empty strings."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return [_text(item, field) for item in value]


def _optional_text_list(value: Any) -> list[str]:
    """Return an optional list of non-empty strings."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Critic summary fields must be lists")
    return [_text(item, "Critic summary item") for item in value]


def _text(value: Any, field: str) -> str:
    """Normalize one required human-readable field."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text
