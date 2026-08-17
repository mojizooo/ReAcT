"""Controlled historical research context for the single Experiment Director."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore

ANALYSIS_KINDS = {
    "research_analysis": "research_analysis.json",
    "dataset_diagnosis": "dataset_diagnosis.json",
    "composition_effects": "composition_effects.json",
    "model_comparison": None,
    "design_decision": "design_decision.json",
    "predicted_candidates": "predicted_candidates.json",
    "candidate_selection": "candidate_selection.json",
    "exploratory_selection": "exploratory_selection.json",
    "design_outcome_review": "design_outcome_review.json",
    "models": None,
}

MODEL_SELECTION_DECISION = "model_selection"
REVIEWED_BATCH_DECISION = "reviewed_batch"


def _load_json(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Read one artifact through TaskStore so models never choose a filesystem path."""
    try:
        return json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load JSON artifact {artifact}") from error


def _dataset_sha256(dataset: dict[str, Any]) -> str:
    """Match the canonical stable hashing convention used by research artifacts."""
    encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class GetResearchBriefingTool:
    """Build the small cross-round history that Runtime injects before a new design decision."""

    name = "get_research_briefing"

    def __init__(self, store: TaskStore) -> None:
        """Bind briefing generation to the active recoverable task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Render measured facts and explicitly labelled prior Agent interpretations without raw spectra."""
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_json(self.store, dataset_artifact)
        target = [float(item) for item in arguments["target"]]
        measured_batches = [batch for batch in dataset["batches"] if batch["origin"]["scientific_eligible"]]
        synthetic_batches = [batch for batch in dataset["batches"] if not batch["origin"]["scientific_eligible"]]
        measured_observations = [
            {"batch": batch, "observation": observation}
            for batch in measured_batches
            for observation in batch["observations"]
        ]
        nearest = min(
            (
                {
                    "sample_id": item["observation"]["identity"]["sample_id"],
                    "batch_id": item["batch"]["batch_id"],
                    "cie": item["observation"]["measurement"]["cie"],
                    "distance": math.dist(item["observation"]["measurement"]["cie"], target),
                }
                for item in measured_observations
            ),
            key=lambda item: item["distance"],
            default=None,
        )
        lineage = [self._decision_lineage(batch) for batch in dataset["batches"]]
        decisions = [item for item in lineage if item["decision"] is not None]
        latest_decision = decisions[-1]["decision"] if decisions else None
        selections = [item for item in lineage if item["candidate_selection"] is not None]
        latest_selection = selections[-1]["candidate_selection"] if selections else None
        exploratory_selections = [item for item in lineage if item["exploratory_selection"] is not None]
        latest_exploratory_selection = exploratory_selections[-1]["exploratory_selection"] if exploratory_selections else None
        briefing = {
            "schema_version": 1,
            "source_dataset": dataset_artifact,
            "source_dataset_sha256": _dataset_sha256(dataset),
            "task": {
                "task_id": dataset["task"]["task_id"],
                "target_cie": target,
                "max_rounds": dataset["task"]["max_rounds"],
            },
            "measured_evidence_summary": {
                "measured_batch_count": len(measured_batches),
                "measured_sample_count": len(measured_observations),
                "nearest_target_sample": nearest,
                "excluded_non_scientific_batches": [
                    {"batch_id": batch["batch_id"], "origin": batch["origin"]["kind"]}
                    for batch in synthetic_batches
                ],
            },
            "decision_lineage": lineage,
            "prior_agent_interpretation": {
                "latest_selected_method": latest_decision["selected_method"] if latest_decision else None,
                "latest_selection_reasons": latest_decision["selection_reasons"] if latest_decision else [],
                "working_hypotheses": latest_decision["working_hypotheses"] if latest_decision else [],
                "next_measurement_purpose": latest_decision["next_measurement_purpose"] if latest_decision else None,
                "latest_candidate_selection": latest_selection,
                "latest_exploratory_selection": latest_exploratory_selection,
                "notice": "These are prior Agent interpretations, not measured scientific facts. Read the cited records before relying on them.",
            },
            "access_guidance": [
                "Use query_research_index for target-near measured samples and compositions.",
                "Use review_design_outcomes for factual review of the just-completed measured batch.",
                "Use get_experiment_record or get_spectrum_data only for specific needed evidence.",
                "Use get_analysis_record for a full prior review, diagnosis, model comparison, decision, or model result.",
            ],
        }
        artifact = self.store.write_artifact_json("artifacts/research_briefing.json", briefing)
        return ToolResult(
            status="success",
            summary=f"Built briefing with {len(measured_observations)} measured observations and {len(decisions)} prior decisions.",
            data=briefing,
            artifacts=[artifact],
        )

    def _decision_lineage(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Read only the fixed per-round decision location, preserving missing decisions explicitly."""
        round_number = int(batch["round"])
        decision_path = f"artifacts/round-{round_number}/design_decision.json"
        decision = None
        if self.store.artifact_path(decision_path).is_file():
            record = _load_json(self.store, decision_path)
            decision = self._normalize_decision(record, decision_path)
        return {
            "round": round_number,
            "batch_id": batch["batch_id"],
            "origin": batch["origin"]["kind"],
            "scientific_eligible": batch["origin"]["scientific_eligible"],
            "best_measured_distance": min(
                (item["evaluation"]["distance"] for item in batch["observations"]), default=None
            ) if batch["origin"]["scientific_eligible"] else None,
            "decision": decision,
            "candidate_selection": self._candidate_selection_lineage(round_number),
            "exploratory_selection": self._exploratory_selection_lineage(round_number),
            "design_outcome_review": self._design_outcome_review_lineage(round_number),
        }

    @staticmethod
    def _normalize_decision(record: dict[str, Any], decision_path: str) -> dict[str, Any]:
        """Normalize model and critic-reviewed decisions without conflating their meanings."""
        decision_kind = record.get("decision_kind")
        if decision_kind is None:
            # Reviewed-batch artifacts existed before the discriminator was introduced.
            decision_kind = (
                REVIEWED_BATCH_DECISION
                if {"primary_option", "allocation_reason", "critic_verdict", "final_reason"}.issubset(record)
                else MODEL_SELECTION_DECISION
                if {"selected_method", "selection_reasons", "next_measurement_purpose"}.issubset(record)
                else None
            )
        if decision_kind == MODEL_SELECTION_DECISION:
            return {
                "decision_kind": MODEL_SELECTION_DECISION,
                "strategy": record["strategy"],
                "selected_method": record["selected_method"],
                "selection_reasons": record["selection_reasons"],
                "facts_used_sample_ids": record["facts_used_sample_ids"],
                "working_hypotheses": record["working_hypotheses"],
                "next_measurement_purpose": record["next_measurement_purpose"],
                "source_model_comparison_artifact": record["source_model_comparison_artifact"],
            }
        if decision_kind == REVIEWED_BATCH_DECISION:
            return {
                "decision_kind": REVIEWED_BATCH_DECISION,
                "strategy": record["strategy"],
                "facts_used_sample_ids": record["facts_used_sample_ids"],
                "next_batch": record["next_batch"],
                "research_question": record["research_question"],
                "primary_option": record["primary_option"],
                "competing_option": record["competing_option"],
                "allocation_reason": record["allocation_reason"],
                "critic_verdict": record["critic_verdict"],
                "critic_responses": record["critic_responses"],
                "final_reason": record["final_reason"],
                # Common fields keep the briefing API stable while explicitly
                # indicating that this was not a response-model selection.
                "selected_method": None,
                "selection_reasons": [record["allocation_reason"]],
                "working_hypotheses": [],
                "next_measurement_purpose": record["primary_option"],
                "source_model_comparison_artifact": None,
            }
        raise ValueError(
            f"unrecognized design decision schema at {decision_path}: "
            "expected decision_kind=model_selection or reviewed_batch"
        )

    def _candidate_selection_lineage(self, round_number: int) -> dict[str, Any] | None:
        """Expose compact prior prediction-selection intent without replacing measured evidence."""
        selection_path = f"artifacts/round-{round_number}/candidate_selection.json"
        if not self.store.artifact_path(selection_path).is_file():
            return None
        selection = _load_json(self.store, selection_path)
        return {
            "next_batch": selection["next_batch"],
            "selected_method": selection["selected_method"],
            "selected_candidate_ids": [item["candidate_id"] for item in selection["selected_candidates"]],
            "selection_reason": selection["selection_reason"],
            "scientific_status": selection["scientific_status"],
        }

    def _exploratory_selection_lineage(self, round_number: int) -> dict[str, Any] | None:
        """Expose compact non-model selection intent while keeping measured evidence separate."""
        selection_path = f"artifacts/round-{round_number}/exploratory_selection.json"
        if not self.store.artifact_path(selection_path).is_file():
            return None
        selection = _load_json(self.store, selection_path)
        return {
            "next_batch": selection["next_batch"],
            "strategy": selection["strategy"],
            "reference_sample_id": selection["reference_sample_id"],
            "active_slots": selection["active_slots"],
            "selection_reason": selection["selection_reason"],
            "scientific_status": selection["scientific_status"],
        }

    def _design_outcome_review_lineage(self, round_number: int) -> dict[str, Any] | None:
        """Expose review counts while retaining per-recipe facts for on-demand access."""
        artifact = f"artifacts/round-{round_number}/design_outcome_review.json"
        if not self.store.artifact_path(artifact).is_file():
            return None
        summary = _load_json(self.store, artifact)["summary"]
        return {
            "artifact": artifact,
            "recipe_count": summary["recipe_count"],
            "with_reference_comparison_count": summary["with_reference_comparison_count"],
            "closer_to_target_count": summary["closer_to_target_count"],
            "farther_from_target_count": summary["farther_from_target_count"],
            "no_reference_comparison_count": summary["no_reference_comparison_count"],
        }


class GetCompleteBatchHistoryTool:
    """Build the complete non-spectral history required before a new batch design."""

    name = "get_complete_batch_history"

    def __init__(self, store: TaskStore) -> None:
        """Bind complete history construction to one recoverable task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Join fixed prior design, observation, and decision records without raw spectra arrays."""
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_json(self.store, dataset_artifact)
        history = {
            "schema_version": 1,
            "source_dataset": dataset_artifact,
            "source_dataset_sha256": _dataset_sha256(dataset),
            "task": dataset["task"],
            "batches": [self._batch_history(batch) for batch in dataset["batches"]],
            "interpretation_notice": (
                "Measurements, CIE values, QC, and spectrum references are task facts. "
                "Design contexts, analyses, models, and decisions are Agent interpretations. "
                "synthetic_dry_run records are visible for operational lineage but are not scientific evidence."
            ),
            "spectrum_access": (
                "Raw emission and absorption arrays are intentionally excluded. "
                "Use get_spectrum_data with a listed sample_id when a specific spectrum is needed."
            ),
        }
        artifact = self.store.write_artifact_json("artifacts/complete_batch_history.json", history)
        return ToolResult(
            status="success",
            summary=f"Built complete history for {len(history['batches'])} prior batches.",
            data=history,
            artifacts=[artifact],
        )

    def _batch_history(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Read one batch using only artifact locations anchored in the canonical dataset."""
        design_artifact = str(batch["plan"]["design_artifact"])
        design = _load_json(self.store, design_artifact)
        round_number = int(batch["round"])
        return {
            "batch_id": batch["batch_id"],
            "round": round_number,
            "origin": batch["origin"],
            "design": {
                "artifact": design_artifact,
                "plan_id": design["plan_id"],
                "design_context": design["design_context"],
                "applied_skills": design["applied_skills"],
                "recipes": design["recipes"],
            },
            "observations": batch["observations"],
            "analysis_records": self._analysis_records(round_number),
        }

    def _analysis_records(self, round_number: int) -> dict[str, Any]:
        """Preserve absent artifacts explicitly instead of inferring missing scientific conclusions."""
        records = {
            kind: self._read_round_artifact(round_number, filename)
            for kind, filename in ANALYSIS_KINDS.items()
            if filename is not None
        }
        model_dir = self.store.artifact_path(f"artifacts/round-{round_number}/models")
        model_paths = sorted(model_dir.glob("*.json")) if model_dir.is_dir() else []
        records["models"] = [
            _load_json(self.store, path.relative_to(self.store.run_dir).as_posix())
            for path in model_paths
            if not path.name.startswith("model_comparison_")
        ]
        comparison_paths = [path for path in model_paths if path.name.startswith("model_comparison_")]
        records["model_comparison"] = (
            _load_json(self.store, comparison_paths[-1].relative_to(self.store.run_dir).as_posix())
            if comparison_paths
            else None
        )
        return records

    def _read_round_artifact(self, round_number: int, filename: str) -> dict[str, Any] | None:
        """Load one known round artifact if it was created during the historical workflow."""
        artifact = f"artifacts/round-{round_number}/{filename}"
        return _load_json(self.store, artifact) if self.store.artifact_path(artifact).is_file() else None


class GetAnalysisRecordTool:
    """Read a complete historical analysis record using round and kind, never an Agent path."""

    name = "get_analysis_record"

    def __init__(self, store: TaskStore) -> None:
        """Bind controlled historical artifact lookup to one task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate requested round eligibility and return the selected fixed analysis artifact."""
        round_number = int(arguments["round"])
        kind = str(arguments["kind"])
        if round_number < 1 or kind not in ANALYSIS_KINDS:
            raise ValueError("round must be positive and kind must be a supported analysis record")
        dataset = _load_json(self.store, str(arguments["research_dataset_artifact"]))
        batch = next((item for item in dataset["batches"] if int(item["round"]) == round_number), None)
        if batch is None:
            return ToolResult(status="not_found", summary="No experiment batch exists for the requested round.")
        if not bool(arguments.get("include_synthetic", False)) and not batch["origin"]["scientific_eligible"]:
            return ToolResult(status="not_found", summary="The requested round is not a permitted measured-data record.")
        payload = self._read_kind(round_number, kind)
        if payload is None:
            return ToolResult(status="not_found", summary=f"No {kind} artifact exists for round {round_number}.")
        return ToolResult(
            status="success",
            summary=f"Loaded {kind} record for round {round_number}.",
            data={"round": round_number, "kind": kind, "record": payload},
        )

    def _read_kind(self, round_number: int, kind: str) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Resolve only fixed artifact names or the fixed run-local models directory."""
        if kind == "model_comparison":
            model_dir = self.store.artifact_path(f"artifacts/round-{round_number}/models")
            paths = sorted(model_dir.glob("model_comparison_*.json")) if model_dir.is_dir() else []
            return _load_json(self.store, paths[-1].relative_to(self.store.run_dir).as_posix()) if paths else None
        if kind == "models":
            model_dir = self.store.artifact_path(f"artifacts/round-{round_number}/models")
            paths = sorted(model_dir.glob("*.json")) if model_dir.is_dir() else []
            records = [
                _load_json(self.store, path.relative_to(self.store.run_dir).as_posix())
                for path in paths
                if not path.name.startswith("model_comparison_")
            ]
            return records or None
        filename = ANALYSIS_KINDS[kind]
        if filename is None:
            return None
        artifact = f"artifacts/round-{round_number}/{filename}"
        return _load_json(self.store, artifact) if self.store.artifact_path(artifact).is_file() else None
