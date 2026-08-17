"""Deterministic evidence contracts for the human-reviewed stop branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from react_color_agent.director.unreachable import (
    validate_unreachable_draft,
    validate_unreachable_responses,
    validate_unreachable_review,
)
from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore
from tools.research_tools import _canonical_sha256, _load_json


class ProposeUnreachableRequestTool:
    """Validate one Director-authored, measured-only unreachable application draft."""

    name = "propose_unreachable_request"

    def __init__(self, store: TaskStore) -> None:
        """Bind evidence reads and artifacts to one recoverable task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Save the draft and a factual coverage report without making a stop decision."""
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_json(self.store, dataset_artifact)
        draft = validate_unreachable_draft(arguments)
        round_number = int(arguments["round"])
        task = dataset.get("task", {})
        if str(task.get("task_id")) != str(arguments.get("task_id")):
            raise ValueError("research dataset belongs to another task")
        measured = measured_records(dataset)
        by_id = {record["sample_id"]: record for record in measured}
        unknown = [sample_id for sample_id in draft["evidence_sample_ids"] if sample_id not in by_id]
        if unknown:
            raise ValueError(
                "unreachable evidence_sample_ids must identify measured observations: "
                + ", ".join(unknown)
            )
        evaluation = _evaluate_evidence(draft, measured, arguments)
        draft_payload = {
            "schema_version": 1,
            "kind": "unreachable_request_draft",
            "task_id": str(arguments["task_id"]),
            "round": round_number,
            "target_cie": list(arguments["target"]),
            "source_dataset": dataset_artifact,
            "source_dataset_sha256": _canonical_sha256(dataset),
            **draft,
        }
        draft_artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/unreachable/draft.json", draft_payload
        )
        evaluation_payload = {
            "schema_version": 1,
            "kind": "unreachable_evidence_evaluation",
            "round": round_number,
            "source_dataset": dataset_artifact,
            "source_dataset_sha256": _canonical_sha256(dataset),
            "source_draft": draft_artifact,
            "source_draft_sha256": _canonical_sha256(draft_payload),
            "origin_policy": "measured_only",
            "result": evaluation,
        }
        evaluation_artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/unreachable/evidence_evaluation.json",
            evaluation_payload,
        )
        return ToolResult(
            status="success",
            summary=(
                f"Saved an evidence-bounded unreachable draft citing "
                f"{len(draft['evidence_sample_ids'])} measured samples; one Critic review is required."
            ),
            data={"draft": draft_payload, "evaluation": evaluation_payload},
            artifacts=[draft_artifact, evaluation_artifact],
        )


class ContinueAfterUnreachableReviewTool:
    """Record a Director decision to continue experiments after Critic review."""

    name = "continue_after_unreachable_review"

    def __init__(self, store: TaskStore) -> None:
        """Bind the decision to one run-local application packet."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate Critic responses and save a continuation decision."""
        source = _load_unreachable_sources(self.store, arguments)
        responses = validate_unreachable_responses(source["critic"], arguments.get("critic_responses"))
        next_question = _text(arguments.get("next_research_question"), "next_research_question")
        next_purpose = _text(arguments.get("next_measurement_purpose"), "next_measurement_purpose")
        decision = _decision_payload(
            source,
            arguments,
            decision_kind="unreachable_review_continued",
            final_reason=_text(arguments.get("final_reason"), "final_reason"),
            critic_responses=responses,
            continuation={
                "next_research_question": next_question,
                "next_measurement_purpose": next_purpose,
            },
        )
        decision_artifact = self.store.write_artifact_json(
            f"artifacts/round-{source['round']}/unreachable/director_decision.json", decision
        )
        return ToolResult(
            status="success",
            summary="Director withdrew the unreachable application and chose to continue experiment design.",
            data={"decision_kind": decision["decision_kind"], "decision": decision},
            artifacts=[decision_artifact],
        )


class SubmitUnreachableApplicationTool:
    """Submit a Critic-reviewed unreachable application for explicit human review."""

    name = "submit_unreachable_application"

    def __init__(self, store: TaskStore) -> None:
        """Bind the final application to the current run and measured dataset."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate the final Director object and enter human review without stopping."""
        source = _load_unreachable_sources(self.store, arguments)
        responses = validate_unreachable_responses(source["critic"], arguments.get("critic_responses"))
        final_draft = validate_unreachable_draft(arguments)
        measured_ids = {record["sample_id"] for record in measured_records(source["dataset"])}
        unknown = [
            sample_id
            for sample_id in final_draft["evidence_sample_ids"]
            if sample_id not in measured_ids
        ]
        if unknown:
            raise ValueError(
                "final unreachable evidence must identify measured observations: "
                + ", ".join(unknown)
            )
        decision = _decision_payload(
            source,
            arguments,
            decision_kind="unreachable_application",
            final_reason=_text(arguments.get("final_reason"), "final_reason"),
            critic_responses=responses,
            final_application=final_draft,
        )
        decision_artifact = self.store.write_artifact_json(
            f"artifacts/round-{source['round']}/unreachable/director_decision.json", decision
        )
        return ToolResult(
            status="success",
            summary="Submitted the Critic-reviewed unreachable application for human approval.",
            data={
                "decision_kind": decision["decision_kind"],
                "awaiting_human_review": True,
                "decision": decision,
            },
            artifacts=[decision_artifact],
        )


def _load_unreachable_sources(store: TaskStore, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Load and hash the exact draft, evidence evaluation, and Critic review sources."""
    dataset = _load_json(store, str(arguments["research_dataset_artifact"]))
    draft = _load_json(store, str(arguments["unreachable_draft_artifact"]))
    evaluation = _load_json(store, str(arguments["unreachable_evaluation_artifact"]))
    critic = _load_json(store, str(arguments["unreachable_critic_review_artifact"]))
    dataset_sha = _canonical_sha256(dataset)
    draft_sha = _canonical_sha256(draft)
    task_id = str(arguments["task_id"])
    round_number = int(arguments["round"])
    if str(dataset.get("task", {}).get("task_id")) != task_id:
        raise ValueError("research dataset belongs to another task")
    if str(draft.get("task_id")) != task_id or int(draft.get("round", -1)) != round_number:
        raise ValueError("unreachable draft belongs to another task or round")
    if int(evaluation.get("round", -1)) != round_number or int(critic.get("round", -1)) != round_number:
        raise ValueError("unreachable evaluation or Critic review belongs to another round")
    if draft.get("source_dataset_sha256") != dataset_sha or evaluation.get("source_dataset_sha256") != dataset_sha:
        raise ValueError("unreachable artifacts do not belong to the current research dataset")
    if evaluation.get("source_draft") != str(arguments["unreachable_draft_artifact"]):
        raise ValueError("unreachable evidence evaluation does not belong to the current draft")
    if evaluation.get("source_draft_sha256") != draft_sha:
        raise ValueError("unreachable evidence evaluation draft hash does not match the current draft")
    if critic.get("source_draft") != str(arguments["unreachable_draft_artifact"]):
        raise ValueError("unreachable Critic review does not belong to the current draft")
    if critic.get("source_dataset_sha256") != dataset_sha:
        raise ValueError("unreachable Critic review does not belong to the current research dataset")
    if critic.get("source_draft_sha256") != draft_sha:
        raise ValueError("unreachable Critic review draft hash does not match the current draft")
    normalized_critic = validate_unreachable_review(critic)
    measured_ids = {record["sample_id"] for record in measured_records(dataset)}
    cited_ids = set(draft.get("evidence_sample_ids", []))
    unknown_critic_ids = [
        sample_id
        for finding in normalized_critic["findings"]
        for sample_id in finding["evidence_sample_ids"]
        if sample_id not in measured_ids or sample_id not in cited_ids
    ]
    if unknown_critic_ids:
        raise ValueError(
            "unreachable Critic findings must cite measured samples from the draft: "
            + ", ".join(dict.fromkeys(unknown_critic_ids))
        )
    return {
        "dataset": dataset,
        "dataset_sha256": dataset_sha,
        "draft": draft,
        "draft_artifact": str(arguments["unreachable_draft_artifact"]),
        "evaluation": evaluation,
        "evaluation_artifact": str(arguments["unreachable_evaluation_artifact"]),
        "critic": critic,
        "critic_artifact": str(arguments["unreachable_critic_review_artifact"]),
        "round": round_number,
    }


def _decision_payload(
    source: Mapping[str, Any],
    arguments: Mapping[str, Any],
    *,
    decision_kind: str,
    final_reason: str,
    critic_responses: list[dict[str, str]],
    continuation: dict[str, str] | None = None,
    final_application: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one auditable Director decision with all source hashes."""
    decision: dict[str, Any] = {
        "schema_version": 1,
        "decision_kind": decision_kind,
        "task_id": str(arguments["task_id"]),
        "round": int(source["round"]),
        "source_dataset": str(arguments["research_dataset_artifact"]),
        "source_dataset_sha256": source["dataset_sha256"],
        "source_draft": source["draft_artifact"],
        "source_draft_sha256": _canonical_sha256(source["draft"]),
        "source_evaluation": source["evaluation_artifact"],
        "source_critic_review": source["critic_artifact"],
        "source_critic_review_sha256": _canonical_sha256(source["critic"]),
        "critic_recommendation": source["critic"].get("recommendation"),
        "critic_responses": critic_responses,
        "final_reason": final_reason,
    }
    if continuation is not None:
        decision["continuation"] = continuation
    if final_application is not None:
        decision["final_application"] = final_application
    return decision


def _evaluate_evidence(
    draft: Mapping[str, Any],
    measured: list[dict[str, Any]],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate factual coverage and residual summaries without a scientific verdict."""
    target = [float(value) for value in arguments["target"]]
    by_id = {record["sample_id"]: record for record in measured}
    cited = [by_id[sample_id] for sample_id in draft["evidence_sample_ids"]]
    nearest = min(measured, key=lambda record: float(record["distance"]))
    per_batch: dict[str, dict[str, Any]] = {}
    for record in measured:
        current = per_batch.setdefault(
            record["batch_id"], {"batch_id": record["batch_id"], "round": record["round"], "best": None}
        )
        if current["best"] is None or record["distance"] < current["best"]["distance"]:
            current["best"] = record
    cited_ids = set(draft["evidence_sample_ids"])
    undisclosed = [
        {
            "sample_id": record["sample_id"],
            "batch_id": record["batch_id"],
            "cie": record["cie"],
            "distance": record["distance"],
        }
        for record in sorted(measured, key=lambda item: item["distance"])[:10]
        if record["sample_id"] not in cited_ids
    ]
    return {
        "verdict": "facts_only_no_scientific_verdict",
        "measured_sample_count": len(measured),
        "measured_batch_count": len(per_batch),
        "cited_sample_count": len(cited),
        "cited_batch_count": len({record["batch_id"] for record in cited}),
        "represented_active_component_counts": sorted(
            {
                sum(float(value) > 0 for value in record["recipe"]["concentrations_mmol_ml"])
                for record in cited
            }
        ),
        "target_cie": target,
        "nearest_measured": nearest,
        "nearest_coordinate_residual": [
            float(value) - target[index] for index, value in enumerate(nearest["cie"])
        ],
        "per_batch_best": list(per_batch.values()),
        "undiscussed_target_near_samples": undisclosed,
        "attempted_route_names": [
            route.get("route") for route in draft["attempted_routes"] if route.get("route")
        ],
        "remaining_budget": max(0, int(arguments["max_rounds"]) - int(arguments["round"])),
        "limitations": [
            "This evaluation validates provenance and coverage only; it does not determine whether the target is scientifically unreachable.",
            "CIE coordinates are measured facts for ranking and residual reporting, not proof of a linear composition law.",
        ],
    }


def measured_records(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten measured observations into compact records suitable for the Critic packet."""
    records: list[dict[str, Any]] = []
    for batch in dataset.get("batches", []):
        if not bool(batch.get("origin", {}).get("scientific_eligible")):
            continue
        for observation in batch.get("observations", []):
            identity = observation["identity"]
            records.append(
                {
                    "sample_id": str(identity["sample_id"]),
                    "recipe_id": str(identity["recipe_id"]),
                    "batch_id": str(batch["batch_id"]),
                    "round": int(batch["round"]),
                    "recipe": observation["recipe"],
                    "cie": observation["measurement"]["cie"],
                    "distance": float(observation["evaluation"]["distance"]),
                    "within_tolerance": bool(observation["evaluation"].get("within_tolerance", False)),
                }
            )
    if not records:
        raise ValueError("research dataset contains no scientifically eligible measured observations")
    return records


def _text(value: Any, field: str) -> str:
    """Normalize one required Director field."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text
