"""Factual post-measurement reviews of CPWL recipe design intent."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore


class ReviewDesignOutcomesTool:
    """Compare one completed measured batch with its declared reference samples."""

    name = "review_design_outcomes"

    def __init__(self, store: TaskStore) -> None:
        """Bind review artifacts to the active recoverable research task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Write deterministic CIE comparisons without interpreting scientific mechanisms."""
        dataset_artifact = str(arguments["research_dataset_artifact"])
        round_number = int(arguments["round"])
        dataset = _load_json(self.store, dataset_artifact)
        batch = next((item for item in dataset.get("batches", []) if int(item["round"]) == round_number), None)
        if batch is None:
            raise ValueError(f"no experiment batch exists for round {round_number}")
        if not bool(batch.get("origin", {}).get("scientific_eligible")):
            raise ValueError("design outcome review requires a scientifically eligible measured batch")

        # References are deliberately limited to earlier measured observations in this task.
        observations = _measured_observations(dataset)
        recipe_reviews = [
            self._review_observation(observation, observations, round_number)
            for observation in batch.get("observations", [])
        ]
        summary = _summary(recipe_reviews)
        review = {
            "schema_version": 1,
            "round": round_number,
            "batch_id": batch["batch_id"],
            "source_dataset": dataset_artifact,
            "source_dataset_sha256": _canonical_sha256(dataset),
            "origin_policy": "measured_only",
            "recipes": recipe_reviews,
            "summary": summary,
            "limitations": [
                "This review reports deterministic measured facts and does not establish mechanism or causality.",
                "Natural-language discrimination hypotheses are not automatically classified as supported or not supported.",
            ],
        }
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/design_outcome_review.json", review
        )
        return ToolResult(
            status="success",
            summary=(
                f"Reviewed {summary['recipe_count']} measured recipes for {batch['batch_id']}; "
                f"{summary['with_reference_comparison_count']} have declared reference comparisons."
            ),
            data={
                "round": round_number,
                "batch_id": batch["batch_id"],
                "recipes": recipe_reviews,
                "summary": summary,
                "limitations": review["limitations"],
            },
            artifacts=[artifact],
        )

    def _review_observation(
        self,
        observation: dict[str, Any],
        observations: dict[str, dict[str, Any]],
        round_number: int,
    ) -> dict[str, Any]:
        """Build one recipe-level review from its persisted plan intent and measured result."""
        identity = observation["identity"]
        intent = observation.get("design_intent", {})
        discrimination = intent.get("discrimination")
        current_cie, current_distance = _measurement_facts(observation, str(identity["sample_id"]))
        review = {
            "recipe_id": identity["recipe_id"],
            "sample_id": identity["sample_id"],
            "purpose": intent.get("purpose", identity.get("design_role")),
            "recipe": observation["recipe"],
            "measurement": {
                "cie": current_cie,
                "target_cie": observation["evaluation"]["target_cie"],
                "target_distance": current_distance,
                "within_tolerance": observation["evaluation"]["within_tolerance"],
            },
            "design_intent_status": "available" if discrimination is not None else "missing_design_intent",
            "discrimination": discrimination,
            "reference_comparisons": [],
        }
        if discrimination is None:
            review["comparison_status"] = "no_reference_comparison"
            review["limitation"] = "No recipe-level discrimination was recorded for this plan."
            return review

        reference_ids = [str(item) for item in discrimination.get("reference_sample_ids", [])]
        if not reference_ids:
            review["comparison_status"] = "no_reference_comparison"
        else:
            review["comparison_status"] = "reference_comparisons_available"
            review["reference_comparisons"] = [
                _reference_comparison(
                    current_cie=current_cie,
                    current_distance=current_distance,
                    reference=_required_reference(observations, reference_id, round_number),
                    reference_id=reference_id,
                )
                for reference_id in reference_ids
            ]
        review["hypothesis_assessment"] = "not_automatically_determined"
        review["limitation"] = (
            "Natural-language experimental hypotheses require Agent interpretation of cited facts."
        )
        return review


def _load_json(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Load one validated task-local JSON artifact."""
    try:
        return json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load JSON artifact {artifact}") from error


def _canonical_sha256(payload: Any) -> str:
    """Hash JSON content with the canonical artifact convention."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measured_observations(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index scientifically eligible observations with their producing round."""
    return {
        str(observation["identity"]["sample_id"]): {"round": int(batch["round"]), "observation": observation}
        for batch in dataset.get("batches", [])
        if batch.get("origin", {}).get("kind") == "measured"
        and bool(batch.get("origin", {}).get("scientific_eligible"))
        for observation in batch.get("observations", [])
    }


def _measurement_facts(observation: dict[str, Any], sample_id: str) -> tuple[list[float], float]:
    """Validate and return the CIE coordinates and target distance needed for comparison."""
    cie = observation.get("measurement", {}).get("cie")
    distance = observation.get("evaluation", {}).get("distance")
    if not isinstance(cie, list) or len(cie) != 2 or distance is None:
        raise ValueError(f"measured observation {sample_id} has incomplete CIE evaluation facts")
    return [float(value) for value in cie], float(distance)


def _required_reference(
    observations: dict[str, dict[str, Any]], reference_id: str, current_round: int
) -> dict[str, Any]:
    """Require one earlier measured reference instead of accepting arbitrary sample IDs."""
    reference = observations.get(reference_id)
    if reference is None:
        raise ValueError(f"reference sample is not a scientifically eligible measured observation: {reference_id}")
    if int(reference["round"]) >= current_round:
        raise ValueError(f"reference sample must come from an earlier measured batch: {reference_id}")
    return reference["observation"]


def _reference_comparison(
    *,
    current_cie: list[float],
    current_distance: float,
    reference: dict[str, Any],
    reference_id: str,
) -> dict[str, Any]:
    """Calculate only direct CIE and target-distance differences for one reference."""
    reference_cie, reference_distance = _measurement_facts(reference, reference_id)
    distance_change = current_distance - reference_distance
    return {
        "reference_sample_id": reference_id,
        "reference_cie": reference_cie,
        "reference_target_distance": reference_distance,
        "delta_x": current_cie[0] - reference_cie[0],
        "delta_y": current_cie[1] - reference_cie[1],
        "cie_distance": math.dist(current_cie, reference_cie),
        "target_distance_change": distance_change,
        "target_distance_assessment": _distance_assessment(distance_change),
    }


def _distance_assessment(change: float, *, epsilon: float = 1e-12) -> str:
    """Label only the deterministic change in target distance."""
    if change < -epsilon:
        return "closer_to_target"
    if change > epsilon:
        return "farther_from_target"
    return "same_target_distance"


def _summary(recipe_reviews: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize comparisons without replacing the complete per-recipe evidence."""
    comparisons = [
        comparison
        for review in recipe_reviews
        for comparison in review["reference_comparisons"]
    ]
    assessments = [comparison["target_distance_assessment"] for comparison in comparisons]
    return {
        "recipe_count": len(recipe_reviews),
        "with_reference_comparison_count": sum(bool(review["reference_comparisons"]) for review in recipe_reviews),
        "closer_to_target_count": assessments.count("closer_to_target"),
        "farther_from_target_count": assessments.count("farther_from_target"),
        "same_target_distance_count": assessments.count("same_target_distance"),
        "no_reference_comparison_count": sum(not review["reference_comparisons"] for review in recipe_reviews),
    }
