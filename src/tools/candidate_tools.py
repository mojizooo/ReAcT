"""Constrained model-based CPWL candidate generation for one measured local scope."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import product
from typing import Any

import numpy as np

from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore


def _load_json(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Read one task-local JSON artifact through the store path guard."""
    try:
        return json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load JSON artifact {artifact}") from error


def _dataset_sha256(dataset: dict[str, Any]) -> str:
    """Match the stable research-dataset hash used by prior analysis layers."""
    encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measured_observations(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index real observations only, so simulations cannot supply predictive candidates."""
    return {
        observation["identity"]["sample_id"]: observation
        for batch in dataset["batches"]
        if batch["origin"]["scientific_eligible"]
        for observation in batch["observations"]
    }


class GeneratePredictedCandidatesTool:
    """Generate inspectable interpolation-only candidates from one Agent-selected local model."""

    name = "generate_predicted_candidates"

    def __init__(self, store: TaskStore) -> None:
        """Bind all candidate artifacts to the current recoverable research task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Reconstruct the selected model and enumerate only its measured local range."""
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_json(self.store, dataset_artifact)
        dataset_hash = _dataset_sha256(dataset)
        decision_artifact = str(arguments["design_decision_artifact"])
        comparison_artifact = str(arguments["model_comparison_artifact"])
        decision = _load_json(self.store, decision_artifact)
        comparison = _load_json(self.store, comparison_artifact)
        if decision["source_dataset_sha256"] != dataset_hash or comparison["source_dataset_sha256"] != dataset_hash:
            raise ValueError("candidate generation requires decision and model comparison from the current dataset")
        selected_method = str(decision["selected_method"])
        if selected_method == "none":
            raise ValueError("the Agent selected no local model, so predictive candidates are unavailable")
        selected_entries = [
            entry
            for entry in comparison["ranked_models"]
            if entry["method"] == selected_method and entry["fit_status"] == "supported"
        ]
        if len(selected_entries) != 1:
            raise ValueError("the selected method must resolve to exactly one supported compared model scope")
        model_artifact = str(selected_entries[0]["artifact"])
        model = _load_json(self.store, model_artifact)
        if model["source_dataset_sha256"] != dataset_hash or model["support"]["fit_status"] != "supported":
            raise ValueError("selected model artifact is not a supported fit for the current dataset")
        if model["method"] != selected_method:
            raise ValueError("selected model artifact does not match the Agent decision")
        grid_size = _validate_grid_size(arguments.get("grid_size", 7))
        observations = _measured_observations(dataset)
        training_ids = [str(item) for item in model["input_sample_ids"]]
        if not training_ids or any(sample_id not in observations for sample_id in training_ids):
            raise ValueError("selected model references missing or non-measured training samples")
        material_slots = [material["slot"] for material in dataset["task"]["materials"]]
        active_slots = [str(slot) for slot in model["active_slots"]]
        active_indices = [material_slots.index(slot) for slot in active_slots]
        training = [observations[sample_id] for sample_id in training_ids]
        features = np.asarray(
            [[float(item["recipe"]["concentrations_mmol_ml"][index]) for index in active_indices] for item in training],
            dtype=float,
        )
        training_cie = np.asarray([item["measurement"]["cie"] for item in training], dtype=float)
        ranges = [[float(features[:, index].min()), float(features[:, index].max())] for index in range(features.shape[1])]
        candidate_rows = self._candidate_rows(
            dataset=dataset,
            model=model,
            active_slots=active_slots,
            active_indices=active_indices,
            training_ids=training_ids,
            training=training,
            features=features,
            training_cie=training_cie,
            ranges=ranges,
            grid_size=grid_size,
        )
        candidate_rows.sort(
            key=lambda row: (row["predicted_target_distance"], tuple(row["concentrations_mmol_ml"]))
        )
        for index, row in enumerate(candidate_rows, start=1):
            row["candidate_id"] = f"P{int(arguments['round'])}-C{index}"
        selectable = [row for row in candidate_rows if row["selection_eligible"]]
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{int(arguments['round'])}/predicted_candidates.json",
            {
                "schema_version": 1,
                "round": int(arguments["round"]),
                "status": "PREDICTED_CANDIDATES_NOT_MEASURED",
                "source_dataset": dataset_artifact,
                "source_dataset_sha256": dataset_hash,
                "origin_policy": "measured_only",
                "target_cie": dataset["task"]["target_cie"],
                "source_design_decision": decision_artifact,
                "source_model_comparison": comparison_artifact,
                "selected_model_artifact": model_artifact,
                "selected_method": selected_method,
                "training_sample_ids": training_ids,
                "active_slots": active_slots,
                "active_ranges_mmol_ml": {slot: ranges[index] for index, slot in enumerate(active_slots)},
                "grid_size_per_dimension": grid_size,
                "candidates": candidate_rows,
                "selection_policy": "Only interpolation candidates that do not duplicate a measured recipe may enter a new CPWL batch.",
            },
        )
        briefing = [_candidate_brief(row) for row in selectable[:24]]
        return ToolResult(
            status="success",
            summary=(
                f"Generated {len(candidate_rows)} model predictions; {len(selectable)} unmeasured interpolation "
                "candidates are eligible for Agent selection."
            ),
            data={
                "round": int(arguments["round"]),
                "selected_method": selected_method,
                "candidate_count": len(candidate_rows),
                "selectable_candidate_count": len(selectable),
                "selectable_candidates": briefing,
            },
            artifacts=[artifact],
        )

    def _candidate_rows(
        self,
        *,
        dataset: dict[str, Any],
        model: dict[str, Any],
        active_slots: list[str],
        active_indices: list[int],
        training_ids: list[str],
        training: list[dict[str, Any]],
        features: np.ndarray,
        training_cie: np.ndarray,
        ranges: list[list[float]],
        grid_size: int,
    ) -> list[dict[str, Any]]:
        """Construct full recipes and keep each candidate's numerical support evidence."""
        target = np.asarray(dataset["task"]["target_cie"], dtype=float)
        base_recipe = list(training[0]["recipe"]["concentrations_mmol_ml"])
        grid = [np.linspace(low, high, grid_size) for low, high in ranges]
        rows = []
        for values in product(*grid):
            query = np.asarray(values, dtype=float)
            predicted, details = _predict_model(model, features, training_cie, training_ids, query)
            concentrations = list(base_recipe)
            for index, value in zip(active_indices, query):
                concentrations[index] = float(round(float(value), 12))
            nearest_index, nearest_distance = _nearest_training(features, query)
            duplicate = next(
                (
                    sample_id
                    for sample_id, observation in zip(training_ids, training)
                    if np.allclose(
                        np.asarray(observation["recipe"]["concentrations_mmol_ml"], dtype=float),
                        np.asarray(concentrations, dtype=float),
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
                None,
            )
            classification = _coverage_classification(query, features)
            exclusions = []
            if classification != "interpolation":
                exclusions.append("candidate lies outside measured local geometric coverage")
            if duplicate is not None:
                exclusions.append("candidate duplicates an existing measured recipe")
            rows.append(
                {
                    "candidate_id": None,
                    "status": "PREDICTED_CANDIDATE",
                    "concentrations_mmol_ml": concentrations,
                    "active_concentrations_mmol_ml": {slot: float(value) for slot, value in zip(active_slots, query)},
                    "predicted_cie": [float(predicted[0]), float(predicted[1])],
                    "predicted_target_distance": float(np.linalg.norm(predicted - target)),
                    "coverage": {
                        "classification": classification,
                        "nearest_training_sample_id": training_ids[nearest_index],
                        "nearest_training_standardized_distance": nearest_distance,
                    },
                    "duplicate_of_measured_sample_id": duplicate,
                    "selection_eligible": not exclusions,
                    "exclusion_reasons": exclusions,
                    "prediction_details": details,
                }
            )
        return rows


def _validate_grid_size(value: Any) -> int:
    """Keep candidate enumeration deliberately small for a local research iteration."""
    try:
        grid_size = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("grid_size must be an integer between 3 and 11") from error
    if not 3 <= grid_size <= 11:
        raise ValueError("grid_size must be between 3 and 11")
    return grid_size


def _predict_model(
    model: dict[str, Any], features: np.ndarray, cie: np.ndarray, sample_ids: list[str], query: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply only parameters stored in the fitted artifact; no model is silently refit."""
    if model["method"] == "local_ridge":
        saved = model["result"]["model"]
        means = np.asarray(saved["feature_means_mmol_ml"], dtype=float)
        scales = np.asarray(saved["feature_scales_mmol_ml"], dtype=float)
        design = np.concatenate([[1.0], (query - means) / scales])
        x_coefficients = np.asarray(saved["coefficients_per_standardized_feature"]["x"], dtype=float)
        y_coefficients = np.asarray(saved["coefficients_per_standardized_feature"]["y"], dtype=float)
        return np.asarray([design @ x_coefficients, design @ y_coefficients], dtype=float), {
            "method": "local_ridge",
            "feature_standardization": {"means_mmol_ml": means.tolist(), "scales_mmol_ml": scales.tolist()},
        }
    if model["method"] == "weighted_neighbors":
        neighbor_count = int(model["result"]["neighbor_count"])
        scales = features.std(axis=0)
        scales[scales == 0] = 1.0
        distances = np.linalg.norm((features - query) / scales, axis=1)
        indices = np.argsort(distances)[: min(neighbor_count, len(features))]
        weights = 1.0 / np.maximum(distances[indices], 1e-12)
        prediction = np.average(cie[indices], axis=0, weights=weights)
        return np.asarray(prediction, dtype=float), {
            "method": "weighted_neighbors",
            "neighbors": [
                {
                    "sample_id": sample_ids[index],
                    "standardized_distance": float(distances[index]),
                    "weight": float(weight / weights.sum()),
                }
                for index, weight in zip(indices, weights)
            ],
        }
    raise ValueError("candidate generation supports only local_ridge or weighted_neighbors artifacts")


def _nearest_training(features: np.ndarray, query: np.ndarray) -> tuple[int, float]:
    """Report distance in the same standardised local composition coordinates used for neighbors."""
    scales = features.std(axis=0)
    scales[scales == 0] = 1.0
    distances = np.linalg.norm((features - query) / scales, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _coverage_classification(query: np.ndarray, features: np.ndarray) -> str:
    """Distinguish local interpolation from coordinate-box extrapolation in one or two dimensions."""
    if np.any(query < features.min(axis=0) - 1e-12) or np.any(query > features.max(axis=0) + 1e-12):
        return "extrapolation"
    if features.shape[1] == 1:
        return "interpolation"
    return "interpolation" if _point_in_convex_hull(query, features) else "extrapolation"


def _point_in_convex_hull(point: np.ndarray, points: np.ndarray) -> bool:
    """Use a small monotonic-chain hull to avoid adding a scientific-computing dependency."""
    unique = sorted({(float(row[0]), float(row[1])) for row in points})
    if len(unique) < 3:
        return False

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for item in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], item) <= 0:
            lower.pop()
        lower.append(item)
    upper: list[tuple[float, float]] = []
    for item in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], item) <= 0:
            upper.pop()
        upper.append(item)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return False
    signs = [cross(hull[index], hull[(index + 1) % len(hull)], (float(point[0]), float(point[1]))) for index in range(len(hull))]
    return all(value >= -1e-12 for value in signs) or all(value <= 1e-12 for value in signs)


def _candidate_brief(row: dict[str, Any]) -> dict[str, Any]:
    """Expose a bounded Agent-facing selection view while retaining full evidence in the artifact."""
    return {
        "candidate_id": row["candidate_id"],
        "concentrations_mmol_ml": row["concentrations_mmol_ml"],
        "predicted_cie": row["predicted_cie"],
        "predicted_target_distance": row["predicted_target_distance"],
        "nearest_training_sample_id": row["coverage"]["nearest_training_sample_id"],
        "classification": row["coverage"]["classification"],
        "status": row["status"],
    }
