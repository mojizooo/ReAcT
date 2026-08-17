"""Bounded local CIE response models and Agent-authored decision records."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore

MODEL_NAMES = {"local_ridge", "weighted_neighbors"}
DECISION_STRATEGIES = {"coverage", "diagnostic", "local_optimization", "repeat_validation"}


def _load_json(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Read one controlled run-local JSON artifact without accepting external paths."""
    try:
        return json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load JSON artifact {artifact}") from error


def _dataset_sha256(dataset: dict[str, Any]) -> str:
    """Produce the same stable hash convention used by the research data index."""
    encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _eligible_observations(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index only real observations so a local response model cannot learn from dry runs."""
    observations: dict[str, dict[str, Any]] = {}
    for batch in dataset["batches"]:
        if not batch["origin"]["scientific_eligible"]:
            continue
        for observation in batch["observations"]:
            observations[observation["identity"]["sample_id"]] = {
                "batch_id": batch["batch_id"],
                "round": batch["round"],
                "observation": observation,
            }
    return observations


class FitLocalResponseModelTool:
    """Fit one explicitly selected low-dimensional local CIE response model with LOOCV."""

    name = "fit_local_response_model"

    def __init__(self, store: TaskStore) -> None:
        """Bind model artifacts to the active task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate a measured local scope, then fit or explain why fitting is unsupported."""
        method = str(arguments["method"])
        if method not in MODEL_NAMES:
            raise ValueError("method must be local_ridge or weighted_neighbors")
        active_slots = _validate_slots(arguments["active_slots"])
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_json(self.store, dataset_artifact)
        dataset_hash = _dataset_sha256(dataset)
        diagnosis = _load_json(self.store, str(arguments["diagnosis_artifact"]))
        if diagnosis["source_dataset_sha256"] != dataset_hash or diagnosis["origin_policy"] != "measured_only":
            raise ValueError("local models require the current measured-only diagnosis artifact")
        selected = self._select_observations(dataset, arguments.get("sample_ids"))
        material_slots = [material["slot"] for material in dataset["task"]["materials"]]
        missing_slots = sorted(set(active_slots) - set(material_slots))
        if missing_slots:
            raise ValueError(f"active slots are absent from this task: {', '.join(missing_slots)}")
        active_indices = [material_slots.index(slot) for slot in active_slots]
        support = _assess_local_scope(selected, active_indices, len(material_slots))
        round_number = max((int(item["round"]) for item in selected), default=int(arguments.get("round", 0)))
        model_key = _model_key(method, active_slots, [item["observation"]["identity"]["sample_id"] for item in selected])
        payload: dict[str, Any] = {
            "schema_version": 1,
            "round": round_number,
            "source_dataset": dataset_artifact,
            "source_dataset_sha256": dataset_hash,
            "origin_policy": "measured_only",
            "method": method,
            "active_slots": active_slots,
            "input_sample_ids": [item["observation"]["identity"]["sample_id"] for item in selected],
            "support": support,
        }
        if support["fit_status"] == "supported":
            features = np.array(
                [
                    [float(item["observation"]["recipe"]["concentrations_mmol_ml"][index]) for index in active_indices]
                    for item in selected
                ],
                dtype=float,
            )
            cie = np.array([item["observation"]["measurement"]["cie"] for item in selected], dtype=float)
            if method == "local_ridge":
                result = _fit_local_ridge(features, cie, float(arguments.get("ridge_alpha", 1.0)))
            else:
                result = _fit_weighted_neighbors(features, cie, int(arguments.get("neighbor_count", 4)))
            payload["result"] = result
            summary = f"Fitted {method} with {len(selected)} measured local samples; mean LOOCV error is {result['loocv']['mean_cie_error']:.6f}."
        else:
            payload["result"] = None
            summary = f"Did not fit {method}: {'; '.join(support['limitations'])}"
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/models/{method}_{model_key}.json", payload
        )
        return ToolResult(
            status="success",
            summary=summary,
            data={"method": method, "fit_status": support["fit_status"], "support": support},
            artifacts=[artifact],
        )

    @staticmethod
    def _select_observations(dataset: dict[str, Any], requested_ids: Any) -> list[dict[str, Any]]:
        """Resolve a model scope by stable IDs and reject unknown or ineligible observations."""
        eligible = _eligible_observations(dataset)
        if requested_ids is None:
            return list(eligible.values())
        if not isinstance(requested_ids, list) or not requested_ids or not all(isinstance(item, str) for item in requested_ids):
            raise ValueError("sample_ids must be a non-empty list of measured sample IDs when provided")
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("sample_ids must not contain duplicates")
        unknown = [sample_id for sample_id in requested_ids if sample_id not in eligible]
        if unknown:
            raise ValueError("sample_ids contain unknown or ineligible observations: " + ", ".join(unknown))
        return [eligible[sample_id] for sample_id in requested_ids]


class CompareModelsTool:
    """Compare reproducible local model artifacts without choosing a scientific strategy."""

    name = "compare_models"

    def __init__(self, store: TaskStore) -> None:
        """Bind comparison artifacts to one task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Rank successfully fitted models by LOOCV while retaining scopes and limitations."""
        model_artifacts = arguments["model_artifacts"]
        if not isinstance(model_artifacts, list) or len(model_artifacts) < 2:
            raise ValueError("model comparison requires at least two model artifacts")
        models = [_load_json(self.store, str(artifact)) for artifact in model_artifacts]
        dataset = _load_json(self.store, str(arguments["research_dataset_artifact"]))
        dataset_hash = _dataset_sha256(dataset)
        if any(model["source_dataset_sha256"] != dataset_hash for model in models):
            raise ValueError("all compared models must use the current research dataset")
        entries = [_comparison_entry(model, str(artifact)) for model, artifact in zip(models, model_artifacts)]
        entries.sort(key=lambda item: (item["mean_loocv_cie_error"] is None, item["mean_loocv_cie_error"] or float("inf")))
        round_number = max(int(model["round"]) for model in models)
        comparison_key = hashlib.sha256("|".join(sorted(str(item) for item in model_artifacts)).encode("utf-8")).hexdigest()[:12]
        limitations = []
        scopes = {(tuple(model["active_slots"]), tuple(model["input_sample_ids"])) for model in models}
        if len(scopes) > 1:
            limitations.append("Compared models use different active slots or sample scopes; compare their LOOCV values cautiously.")
        if any(entry["fit_status"] != "supported" for entry in entries):
            limitations.append("At least one requested model was unsupported and is retained only as a negative result.")
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/models/model_comparison_{comparison_key}.json",
            {
                "schema_version": 1,
                "round": round_number,
                "source_dataset": str(arguments["research_dataset_artifact"]),
                "source_dataset_sha256": dataset_hash,
                "origin_policy": "measured_only",
                "model_artifacts": model_artifacts,
                "ranked_models": entries,
                "limitations": limitations,
                "selection_boundary": "The Experiment Director must select a method or reject all methods; this tool does not choose a scientific strategy.",
            },
        )
        supported_count = sum(entry["fit_status"] == "supported" for entry in entries)
        return ToolResult(
            status="success",
            summary=f"Compared {len(entries)} models; {supported_count} completed LOOCV and were ranked by mean error.",
            data={"ranked_models": entries, "limitations": limitations},
            artifacts=[artifact],
        )


class WriteDesignDecisionTool:
    """Persist the Director's explainable method choice without generating a new experiment plan."""

    name = "write_design_decision"

    def __init__(self, store: TaskStore) -> None:
        """Bind structured Agent decision records to one research task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate artifact lineage and write an auditable model-selection decision."""
        strategy = str(arguments["strategy"])
        if strategy not in DECISION_STRATEGIES:
            raise ValueError("strategy must be coverage, diagnostic, local_optimization, or repeat_validation")
        selected_method = str(arguments["selected_method"])
        if selected_method not in MODEL_NAMES | {"none"}:
            raise ValueError("selected_method must be local_ridge, weighted_neighbors, or none")
        dataset = _load_json(self.store, str(arguments["research_dataset_artifact"]))
        dataset_hash = _dataset_sha256(dataset)
        analysis_artifact = str(arguments.get("research_analysis_artifact") or "").strip() or None
        analysis = _load_json(self.store, analysis_artifact) if analysis_artifact else None
        comparison_artifact = arguments.get("model_comparison_artifact")
        comparison = _load_json(self.store, str(comparison_artifact)) if comparison_artifact else None
        if analysis is not None and analysis["source_dataset_sha256"] != dataset_hash:
            raise ValueError("research analysis does not match the current research dataset")
        if comparison is not None and comparison["source_dataset_sha256"] != dataset_hash:
            raise ValueError("model comparison does not match the current research dataset")
        if selected_method != "none":
            if analysis is None:
                raise ValueError("a selected response model requires a research analysis artifact")
            if comparison is None:
                raise ValueError("a selected response model requires a model comparison artifact")
            compared_methods = {entry["method"] for entry in comparison["ranked_models"]}
            if selected_method not in compared_methods:
                raise ValueError("selected_method was not included in the provided model comparison")
        facts_used = _validate_fact_sample_ids(dataset, arguments.get("facts_used_sample_ids", []))
        reasons = _validate_text_list(arguments["selection_reasons"], "selection_reasons")
        hypotheses = _validate_text_list(arguments.get("working_hypotheses", []), "working_hypotheses")
        rejected_options = arguments.get("rejected_options", [])
        if not isinstance(rejected_options, list) or not all(isinstance(item, dict) for item in rejected_options):
            raise ValueError("rejected_options must be a list of option/reason objects")
        normalized_rejected_options = []
        for rejected in rejected_options:
            # Accept the previous method field for existing callers, but methods and
            # strategies share one neutral option field in the durable record.
            option = str(rejected.get("option", rejected.get("method", ""))).strip()
            reason = str(rejected.get("reason", "")).strip()
            if not option or not reason:
                raise ValueError("each rejected option requires non-empty option and reason")
            normalized_rejected_options.append({"option": option, "reason": reason})
        if arguments.get("candidate_region") is not None:
            raise ValueError("candidate_region must remain null until constrained candidate generation is migrated")
        measured_round = max(
            (int(item["round"]) for item in _eligible_observations(dataset).values()),
            default=0,
        )
        round_number = max(
            measured_round,
            int(analysis["round"]) if analysis is not None else 0,
            int(comparison["round"]) if comparison is not None else 0,
        )
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/design_decision.json",
            {
                "schema_version": 1,
                "decision_kind": "model_selection",
                "round": round_number,
                "source_dataset_sha256": dataset_hash,
                "source_analysis_artifact": analysis_artifact,
                "source_model_comparison_artifact": str(comparison_artifact) if comparison_artifact else None,
                "evidence_mode": "analysis_artifact" if analysis is not None else "direct_measured_records",
                "strategy": strategy,
                "research_question": _validate_text(arguments["research_question"], "research_question"),
                "facts_used_sample_ids": facts_used,
                "working_hypotheses": hypotheses,
                "selected_method": selected_method,
                "selection_reasons": reasons,
                "rejected_options": normalized_rejected_options,
                "candidate_region": None,
                "next_measurement_purpose": _validate_text(
                    arguments["next_measurement_purpose"], "next_measurement_purpose"
                ),
                "status": (
                    "ANALYSIS_SELECTED_NO_CANDIDATE_GENERATED"
                    if selected_method != "none"
                    else (
                        "ANALYSIS_SELECTED_EXPLORATORY_FOLLOWUP"
                        if analysis is not None
                        else "DIRECT_EVIDENCE_SELECTED_EXPLORATORY_FOLLOWUP"
                    )
                ),
            },
        )
        return ToolResult(
            status="success",
            summary=f"Recorded the Agent's {strategy} decision with selected_method={selected_method}.",
            data={"round": round_number, "selected_method": selected_method, "strategy": strategy},
            artifacts=[artifact],
        )


def _validate_slots(value: Any) -> list[str]:
    """Require a compact, ordered one- or two-material active subspace."""
    if not isinstance(value, list) or not 1 <= len(value) <= 2 or not all(isinstance(item, str) for item in value):
        raise ValueError("active_slots must contain one or two material slots")
    slots = [item.strip().upper() for item in value]
    if len(set(slots)) != len(slots) or any(slot not in {"A", "B", "C", "D", "E"} for slot in slots):
        raise ValueError("active_slots must be unique A-E material slots")
    return slots


def _assess_local_scope(
    selected: list[dict[str, Any]], active_indices: list[int], material_count: int
) -> dict[str, Any]:
    """Reject confounded or underspecified local scopes before numerical fitting begins."""
    limitations = []
    required_count = max(6, len(active_indices) + 4)
    if len(selected) < required_count:
        limitations.append(f"At least {required_count} measured samples are required for this {len(active_indices)}D local model.")
    concentrations = np.array(
        [item["observation"]["recipe"]["concentrations_mmol_ml"][:material_count] for item in selected], dtype=float
    )
    inactive_indices = [index for index in range(material_count) if index not in active_indices]
    for index in inactive_indices:
        if len(np.unique(concentrations[:, index])) > 1:
            limitations.append("Inactive material concentrations vary within the selected scope, so the active effects are confounded.")
            break
    active = concentrations[:, active_indices]
    if len(np.unique(active, axis=0)) < len(active_indices) + 2:
        limitations.append("The selected active concentrations do not contain enough unique local recipes.")
    if np.linalg.matrix_rank(active - active.mean(axis=0)) < len(active_indices):
        limitations.append("The selected active concentration matrix lacks independent variation.")
    return {
        "fit_status": "supported" if not limitations else "unsupported",
        "sample_count": len(selected),
        "active_dimension": len(active_indices),
        "limitations": limitations,
    }


def _fit_local_ridge(features: np.ndarray, cie: np.ndarray, alpha: float) -> dict[str, Any]:
    """Fit separate standardized ridge regressions for x and y and evaluate each holdout."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge_alpha must be a positive finite number")
    predictions = []
    for holdout in range(len(features)):
        mask = np.arange(len(features)) != holdout
        predicted, _model = _ridge_predict(features[mask], cie[mask], features[holdout], alpha)
        predictions.append(_prediction_record(holdout, cie[holdout], predicted))
    fitted, model = _ridge_predict(features, cie, features[0], alpha)
    del fitted
    return {
        "ridge_alpha": alpha,
        "model": model,
        "loocv": _loocv_summary(predictions),
    }


def _ridge_predict(
    train_features: np.ndarray, train_cie: np.ndarray, query: np.ndarray, alpha: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a ridge model with an unpenalized intercept in standardized feature coordinates."""
    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    if np.any(scales == 0):
        raise ValueError("local ridge requires variation in every active feature")
    standardized = (train_features - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.diag([0.0, *([alpha] * standardized.shape[1])])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_cie)
    query_design = np.concatenate([[1.0], (query - means) / scales])
    return query_design @ coefficients, {
        "feature_means_mmol_ml": means.tolist(),
        "feature_scales_mmol_ml": scales.tolist(),
        "coefficients_per_standardized_feature": {
            "x": coefficients[:, 0].tolist(),
            "y": coefficients[:, 1].tolist(),
        },
    }


def _fit_weighted_neighbors(features: np.ndarray, cie: np.ndarray, neighbor_count: int) -> dict[str, Any]:
    """Use distance-weighted local interpolation and leave-one-out predictions without a response formula."""
    if not 1 <= neighbor_count <= 5:
        raise ValueError("neighbor_count must be between 1 and 5")
    predictions = []
    for holdout in range(len(features)):
        mask = np.arange(len(features)) != holdout
        predicted, neighbor_details = _neighbor_predict(
            features[mask], cie[mask], features[holdout], min(neighbor_count, int(mask.sum()))
        )
        record = _prediction_record(holdout, cie[holdout], predicted)
        record["neighbors"] = neighbor_details
        predictions.append(record)
    return {"neighbor_count": neighbor_count, "loocv": _loocv_summary(predictions)}


def _neighbor_predict(
    train_features: np.ndarray, train_cie: np.ndarray, query: np.ndarray, neighbor_count: int
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Estimate CIE from nearest standardized recipes, giving exact duplicates finite dominant weight."""
    scales = train_features.std(axis=0)
    scales[scales == 0] = 1.0
    distances = np.linalg.norm((train_features - query) / scales, axis=1)
    nearest_indices = np.argsort(distances)[:neighbor_count]
    nearest_distances = distances[nearest_indices]
    weights = 1.0 / np.maximum(nearest_distances, 1e-12)
    predicted = np.average(train_cie[nearest_indices], axis=0, weights=weights)
    return predicted, [
        {"distance": float(distance), "weight": float(weight / weights.sum())}
        for distance, weight in zip(nearest_distances, weights)
    ]


def _prediction_record(index: int, observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Keep each held-out CIE prediction inspectable instead of returning only an aggregate score."""
    return {
        "holdout_index": index,
        "observed_cie": observed.tolist(),
        "predicted_cie": predicted.tolist(),
        "cie_error": float(np.linalg.norm(observed - predicted)),
    }


def _loocv_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize LOOCV error while retaining all individual estimates for scientific inspection."""
    errors = [item["cie_error"] for item in predictions]
    return {
        "mean_cie_error": float(np.mean(errors)),
        "median_cie_error": float(np.median(errors)),
        "max_cie_error": float(np.max(errors)),
        "predictions": predictions,
    }


def _model_key(method: str, slots: list[str], sample_ids: list[str]) -> str:
    """Allocate deterministic distinct model artifacts for different scopes in one round."""
    text = json.dumps({"method": method, "slots": slots, "sample_ids": sample_ids}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _comparison_entry(model: dict[str, Any], artifact: str) -> dict[str, Any]:
    """Project comparison-relevant fields without discarding a failed fit's diagnostic value."""
    result = model["result"]
    loocv = result["loocv"] if result is not None else None
    return {
        "method": model["method"],
        "artifact": artifact,
        "fit_status": model["support"]["fit_status"],
        "active_slots": model["active_slots"],
        "input_sample_ids": model["input_sample_ids"],
        "mean_loocv_cie_error": loocv["mean_cie_error"] if loocv else None,
        "max_loocv_cie_error": loocv["max_cie_error"] if loocv else None,
        "limitations": model["support"]["limitations"],
    }


def _validate_fact_sample_ids(dataset: dict[str, Any], value: Any) -> list[str]:
    """Ensure Agent-cited facts refer to measured sample IDs in the current dataset."""
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("facts_used_sample_ids must be a non-empty list of measured sample IDs")
    eligible_ids = set(_eligible_observations(dataset))
    unknown = [item for item in value if item not in eligible_ids]
    if unknown:
        raise ValueError("facts_used_sample_ids contain unknown or ineligible samples: " + ", ".join(unknown))
    return list(dict.fromkeys(value))


def _validate_text_list(value: Any, name: str) -> list[str]:
    """Require concise non-empty Agent-authored rationale while retaining it as a hypothesis, not a fact."""
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _validate_text(value: Any, name: str) -> str:
    """Validate a required Agent-authored research statement without treating it as measured evidence."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text
