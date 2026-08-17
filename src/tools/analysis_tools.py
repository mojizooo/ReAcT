"""Deterministic first-layer diagnostics for task-local colour research data."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from typing import Any

from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore


def _load_dataset(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Load the canonical dataset through the run-local path safety boundary."""
    path = store.artifact_path(artifact)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load research dataset {artifact}") from error


def _dataset_sha256(dataset: dict[str, Any]) -> str:
    """Match the stable dataset hashing convention used by the derived research index."""
    encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _eligible_samples(dataset: dict[str, Any], batch_id: str | None = None) -> list[dict[str, Any]]:
    """Flatten only real observations so analysis tools cannot silently use dry-run data."""
    samples: list[dict[str, Any]] = []
    for batch in dataset["batches"]:
        if batch_id and batch["batch_id"] != batch_id:
            continue
        if not batch["origin"]["scientific_eligible"]:
            continue
        for observation in batch["observations"]:
            samples.append({"batch_id": batch["batch_id"], "round": batch["round"], "observation": observation})
    return samples


def _round_number(samples: list[dict[str, Any]], fallback: int) -> int:
    """Use the latest included round for a compact per-round derived artifact location."""
    return max((int(sample["round"]) for sample in samples), default=fallback)


class DiagnoseDatasetTool:
    """Describe coverage and limitations before an Agent attempts any colour-response model."""

    name = "diagnose_dataset"

    def __init__(self, store: TaskStore) -> None:
        """Bind diagnostic artifacts to one recoverable research task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Write a measured-only coverage diagnostic without proposing a model or candidate."""
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_dataset(self.store, dataset_artifact)
        target = _validate_target(arguments["target"])
        batch_id = _optional_text(arguments.get("batch_id"))
        samples = _eligible_samples(dataset, batch_id)
        materials = dataset["task"]["materials"]
        result = self._diagnose(samples, materials, target)
        round_number = _round_number(samples, int(arguments.get("round", 0)))
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/dataset_diagnosis.json",
            {
                "schema_version": 1,
                "round": round_number,
                "source_dataset": dataset_artifact,
                "source_dataset_sha256": _dataset_sha256(dataset),
                "origin_policy": "measured_only",
                "target_cie": target,
                "batch_filter": batch_id,
                "result": result,
            },
        )
        return ToolResult(
            status="success",
            summary=(
                f"Diagnosed {result['sample_count']} eligible samples; "
                f"target_in_cie_box={result['cie_geometry']['target_in_bounding_box']}."
            ),
            data=result,
            artifacts=[artifact],
        )

    @staticmethod
    def _diagnose(
        samples: list[dict[str, Any]], materials: list[dict[str, Any]], target: list[float]
    ) -> dict[str, Any]:
        """Compute transparent coverage facts using only stored recipe and CIE values."""
        material_coverage = []
        pair_coverage = []
        for index, material in enumerate(materials):
            values = sorted(
                {
                    float(sample["observation"]["recipe"]["concentrations_mmol_ml"][index])
                    for sample in samples
                }
            )
            positive = [value for value in values if value > 0]
            material_coverage.append(
                {
                    "slot": material["slot"],
                    "name": material["name"],
                    "min_concentration_mmol_ml": min(values) if values else None,
                    "max_concentration_mmol_ml": max(values) if values else None,
                    "distinct_positive_concentration_count": len(positive),
                    "nonzero_sample_count": sum(
                        float(sample["observation"]["recipe"]["concentrations_mmol_ml"][index]) > 0
                        for sample in samples
                    ),
                }
            )
        for first, second in combinations(range(len(materials)), 2):
            count = sum(
                float(sample["observation"]["recipe"]["concentrations_mmol_ml"][first]) > 0
                and float(sample["observation"]["recipe"]["concentrations_mmol_ml"][second]) > 0
                for sample in samples
            )
            pair_coverage.append(
                {"slots": [materials[first]["slot"], materials[second]["slot"]], "joint_nonzero_sample_count": count}
            )
        cies = [sample["observation"]["measurement"]["cie"] for sample in samples]
        nearest = min(
            (
                {
                    "sample_id": sample["observation"]["identity"]["sample_id"],
                    "cie": sample["observation"]["measurement"]["cie"],
                    "distance": math.dist(sample["observation"]["measurement"]["cie"], target),
                }
                for sample in samples
            ),
            key=lambda item: item["distance"],
            default=None,
        )
        cie_box = {
            "x": [min((cie[0] for cie in cies), default=None), max((cie[0] for cie in cies), default=None)],
            "y": [min((cie[1] for cie in cies), default=None), max((cie[1] for cie in cies), default=None)],
        }
        target_in_box = bool(
            cies
            and cie_box["x"][0] <= target[0] <= cie_box["x"][1]
            and cie_box["y"][0] <= target[1] <= cie_box["y"][1]
        )
        repeats = _repeat_summary(samples)
        flags = [
            {
                "sample_id": sample["observation"]["identity"]["sample_id"],
                "flags": sample["observation"]["qc"]["flags"],
            }
            for sample in samples
            if sample["observation"]["qc"]["flags"]
        ]
        active_dimensions = sum(item["nonzero_sample_count"] > 0 for item in material_coverage)
        limitations = _diagnostic_limitations(len(samples), active_dimensions, target_in_box, nearest)
        return {
            "sample_count": len(samples),
            "sample_ids": [sample["observation"]["identity"]["sample_id"] for sample in samples],
            "unique_recipe_count": len(
                {tuple(sample["observation"]["recipe"]["concentrations_mmol_ml"]) for sample in samples}
            ),
            "material_coverage": material_coverage,
            "pair_coverage": pair_coverage,
            "active_material_dimension": active_dimensions,
            "cie_geometry": {
                "bounding_box": cie_box,
                "target_in_bounding_box": target_in_box,
                "nearest_sample": nearest,
                "note": "The CIE bounding box is a coverage diagnostic, not evidence of physical reachability.",
            },
            "data_quality": {
                "all_contracts_passed": all(
                    sample["observation"]["qc"]["data_contract"] == "passed" for sample in samples
                ),
                "flagged_samples": flags,
                "repeated_recipes": repeats,
            },
            "limitations": limitations,
        }


class ScreenCompositionEffectsTool:
    """Summarize anchor and sparse-probe evidence without fitting a global response model."""

    name = "screen_composition_effects"

    def __init__(self, store: TaskStore) -> None:
        """Bind composition-screen artifacts to the current task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Write measured-only directional evidence and explicit data limitations for the Agent."""
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_dataset(self.store, dataset_artifact)
        target = _validate_target(arguments["target"])
        samples = _eligible_samples(dataset, _optional_text(arguments.get("batch_id")))
        materials = dataset["task"]["materials"]
        result = self._screen(samples, materials, target)
        round_number = _round_number(samples, int(arguments.get("round", 0)))
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/composition_effects.json",
            {
                "schema_version": 1,
                "round": round_number,
                "source_dataset": dataset_artifact,
                "source_dataset_sha256": _dataset_sha256(dataset),
                "origin_policy": "measured_only",
                "target_cie": target,
                "result": result,
            },
        )
        return ToolResult(
            status="success",
            summary=f"Screened {len(result['single_component_anchors'])} anchors and {len(result['binary_probes'])} binary probes.",
            data=result,
            artifacts=[artifact],
        )

    @staticmethod
    def _screen(
        samples: list[dict[str, Any]], materials: list[dict[str, Any]], target: list[float]
    ) -> dict[str, Any]:
        """Extract observed endpoints and probes while avoiding unsupported non-additivity claims."""
        nearest = min(
            samples,
            key=lambda sample: math.dist(sample["observation"]["measurement"]["cie"], target),
            default=None,
        )
        target_vector = None
        if nearest is not None:
            nearest_cie = nearest["observation"]["measurement"]["cie"]
            target_vector = {
                "from_sample_id": nearest["observation"]["identity"]["sample_id"],
                "from_cie": nearest_cie,
                "to_target_delta": [target[0] - nearest_cie[0], target[1] - nearest_cie[1]],
            }
        anchors = []
        probes = []
        for index, material in enumerate(materials):
            material_samples = [sample for sample in samples if _active_indices(sample["observation"])[0] == [index]]
            if len(material_samples) < 2:
                continue
            ordered = sorted(
                material_samples,
                key=lambda sample: float(sample["observation"]["recipe"]["concentrations_mmol_ml"][index]),
            )
            low, high = ordered[0], ordered[-1]
            low_cie = low["observation"]["measurement"]["cie"]
            high_cie = high["observation"]["measurement"]["cie"]
            anchors.append(
                {
                    "slot": material["slot"],
                    "name": material["name"],
                    "sample_ids": [sample["observation"]["identity"]["sample_id"] for sample in ordered],
                    "concentration_range_mmol_ml": [
                        low["observation"]["recipe"]["concentrations_mmol_ml"][index],
                        high["observation"]["recipe"]["concentrations_mmol_ml"][index],
                    ],
                    "endpoint_cie_delta": [high_cie[0] - low_cie[0], high_cie[1] - low_cie[1]],
                    "interpretation_limit": "Observed endpoint direction only; it does not establish linearity or transfer to mixtures.",
                }
            )
        for first, second in combinations(range(len(materials)), 2):
            pair_samples = [
                sample
                for sample in samples
                if _active_indices(sample["observation"])[0] == [first, second]
            ]
            if not pair_samples:
                continue
            probes.append(
                {
                    "slots": [materials[first]["slot"], materials[second]["slot"]],
                    "names": [materials[first]["name"], materials[second]["name"]],
                    "samples": [
                        {
                            "sample_id": sample["observation"]["identity"]["sample_id"],
                            "concentrations_mmol_ml": sample["observation"]["recipe"]["concentrations_mmol_ml"],
                            "cie": sample["observation"]["measurement"]["cie"],
                        }
                        for sample in pair_samples
                    ],
                    "non_additivity_status": "not_evaluated",
                    "interpretation_limit": "No concentration-matched single-component reference is assumed by this first-layer screen.",
                }
            )
        limitations = []
        if not samples:
            limitations.append("No measured observations are available; synthetic data is excluded by policy.")
        if not anchors:
            limitations.append("No material has at least two single-component anchors for endpoint screening.")
        if probes:
            limitations.append("Binary probes are descriptive only; non-additivity requires concentration-matched controls.")
        return {
            "sample_ids": [sample["observation"]["identity"]["sample_id"] for sample in samples],
            "target_vector_from_nearest": target_vector,
            "single_component_anchors": anchors,
            "binary_probes": probes,
            "limitations": limitations,
        }


class CompileResearchAnalysisTool:
    """Package diagnostics and screening into one reproducible analysis artifact for Agent review."""

    name = "compile_research_analysis"

    def __init__(self, store: TaskStore) -> None:
        """Bind the generated analysis record to one task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Verify matching inputs and write a fact-only first-layer analysis summary."""
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_dataset(self.store, dataset_artifact)
        diagnosis_artifact = str(arguments["diagnosis_artifact"])
        effects_artifact = str(arguments["composition_effects_artifact"])
        diagnosis = _load_artifact(self.store, diagnosis_artifact)
        effects = _load_artifact(self.store, effects_artifact)
        dataset_hash = _dataset_sha256(dataset)
        if diagnosis["source_dataset_sha256"] != dataset_hash or effects["source_dataset_sha256"] != dataset_hash:
            raise ValueError("analysis inputs do not match the current research dataset")
        if diagnosis["origin_policy"] != "measured_only" or effects["origin_policy"] != "measured_only":
            raise ValueError("research analysis requires measured-only source artifacts")
        round_number = max(int(diagnosis["round"]), int(effects["round"]))
        nearest = diagnosis["result"]["cie_geometry"]["nearest_sample"]
        facts = []
        if nearest is not None:
            facts.append(
                {
                    "kind": "nearest_target_sample",
                    "sample_ids": [nearest["sample_id"]],
                    "cie": nearest["cie"],
                    "distance": nearest["distance"],
                }
            )
        facts.append(
            {
                "kind": "dataset_coverage",
                "sample_ids": diagnosis["result"]["sample_ids"],
                "active_material_dimension": diagnosis["result"]["active_material_dimension"],
                "target_in_cie_bounding_box": diagnosis["result"]["cie_geometry"]["target_in_bounding_box"],
            }
        )
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/research_analysis.json",
            {
                "schema_version": 1,
                "round": round_number,
                "source_dataset": dataset_artifact,
                "source_dataset_sha256": dataset_hash,
                "origin_policy": "measured_only",
                "research_question": str(arguments["research_question"]).strip(),
                "methods": [
                    {
                        "name": "diagnose_dataset",
                        "artifact": diagnosis_artifact,
                        "input_sample_ids": diagnosis["result"]["sample_ids"],
                        "parameters": {"target_cie": diagnosis["target_cie"]},
                        "limitations": diagnosis["result"]["limitations"],
                    },
                    {
                        "name": "screen_composition_effects",
                        "artifact": effects_artifact,
                        "input_sample_ids": effects["result"]["sample_ids"],
                        "parameters": {"target_cie": effects["target_cie"]},
                        "limitations": effects["result"]["limitations"],
                    },
                ],
                "conclusions": {
                    "facts": facts,
                    "hypotheses": [],
                    "model_status": "not_attempted",
                    "limitations": list(
                        dict.fromkeys(diagnosis["result"]["limitations"] + effects["result"]["limitations"])
                    ),
                },
            },
        )
        return ToolResult(
            status="success",
            summary="Compiled measured-only diagnostic and composition-screen evidence; no predictive model was attempted.",
            data={"round": round_number, "model_status": "not_attempted", "fact_count": len(facts)},
            artifacts=[artifact],
        )


def _validate_target(value: Any) -> list[float]:
    """Validate a two-coordinate CIE target before using it in geometry calculations."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("target must contain exactly two CIE coordinates")
    target = [float(component) for component in value]
    if any(component < 0 or component > 1 for component in target):
        raise ValueError("target CIE coordinates must be between 0 and 1")
    return target


def _optional_text(value: Any) -> str | None:
    """Normalize optional model filter input without treating an empty string as a filter."""
    text = str(value).strip() if value is not None else ""
    return text or None


def _active_indices(observation: dict[str, Any]) -> tuple[list[int], tuple[float, ...]]:
    """Return active recipe component positions and a stable concentration tuple."""
    concentrations = tuple(float(value) for value in observation["recipe"]["concentrations_mmol_ml"])
    return [index for index, value in enumerate(concentrations) if value > 0], concentrations


def _repeat_summary(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose repeated recipe CIE spread without inventing a repeatability threshold."""
    groups: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        _indices, concentration_key = _active_indices(sample["observation"])
        groups[concentration_key].append(sample)
    summaries = []
    for concentrations, group in groups.items():
        if len(group) < 2:
            continue
        cies = [item["observation"]["measurement"]["cie"] for item in group]
        maximum_spread = max(math.dist(first, second) for first, second in combinations(cies, 2))
        summaries.append(
            {
                "concentrations_mmol_ml": list(concentrations),
                "sample_ids": [item["observation"]["identity"]["sample_id"] for item in group],
                "max_cie_spread": maximum_spread,
            }
        )
    return summaries


def _diagnostic_limitations(
    sample_count: int, active_dimensions: int, target_in_box: bool, nearest: dict[str, Any] | None
) -> list[str]:
    """State only deterministic limits that prevent over-interpreting sparse observations."""
    limitations = []
    if sample_count == 0:
        limitations.append("No measured observations are available; no scientific model can be assessed.")
        return limitations
    if active_dimensions > 2:
        limitations.append("More than two active materials are present; first-version local modelling requires prior screening.")
    if not target_in_box:
        limitations.append("Target lies outside the observed CIE bounding box; treat future predictions as directional exploration, not interpolation.")
    if nearest is not None and nearest["distance"] > 0.05:
        limitations.append("The nearest measured CIE is more than 0.05 from the target; target-local modelling is not yet supported.")
    if sample_count < 6:
        limitations.append("Fewer than six measured samples are available; cross-validated local response modelling is not supported.")
    return limitations


def _load_artifact(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Read a JSON analysis artifact through the same task-local path boundary."""
    try:
        return json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load analysis artifact {artifact}") from error
