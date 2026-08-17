"""Deterministic current-task research dataset, index, and controlled evidence reads."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from react_color_agent.cie import CIE_TOLERANCE, cie_within_coordinate_tolerance
from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore
from tools.spectrum_tools import _parse_absorption, _parse_emission

# Keep the legacy name for persisted protocol compatibility while sharing the canonical value.
GOAL_TOLERANCE = CIE_TOLERANCE


def _canonical_sha256(payload: Any) -> str:
    """Hash JSON content stably so the derived index can identify its exact source."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Read one run-local JSON artifact through TaskStore path validation."""
    path = store.artifact_path(artifact)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON artifact {artifact}") from error


class UpdateResearchDatasetTool:
    """Join one qualified round into the canonical task-local scientific dataset."""

    name = "update_research_dataset"

    def __init__(self, store: TaskStore) -> None:
        """Bind the generated data artifacts to the active recoverable task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Create or extend the dataset, then deterministically rebuild its query index."""
        design = _load_json(self.store, str(arguments["experiment_plan"]).replace(".xlsx", "_design.json"))
        manifest = _load_json(self.store, str(arguments["spectra_artifact"]))
        measurement = _load_json(self.store, str(arguments["measurement_artifact"]))
        analysis = _load_json(self.store, str(arguments["analysis_artifact"]))
        round_number = int(manifest["round"])
        batch_id = self._batch_id(design, round_number)
        origin = self._origin(manifest)
        dataset_path = "artifacts/research_dataset.json"
        dataset = self._load_or_create_dataset(dataset_path, arguments, design)
        new_batch = self._build_batch(
            batch_id=batch_id,
            round_number=round_number,
            origin=origin,
            design=design,
            manifest=manifest,
            measurement=measurement,
            analysis=analysis,
            arguments=arguments,
        )
        self._merge_batch(dataset, new_batch)
        dataset_ref = self.store.write_artifact_json(dataset_path, dataset)
        index = _build_research_index(dataset, dataset_ref)
        index_ref = self.store.write_artifact_json("artifacts/research_index.json", index)
        notebook_ref = self.store.write_artifact_text(
            "artifacts/research_notebook.md", _render_notebook(dataset, index)
        )
        return ToolResult(
            status="success",
            summary=(
                f"Recorded {len(new_batch['observations'])} {origin['kind']} observations for {batch_id}; "
                f"scientific eligibility is {origin['scientific_eligible']}."
            ),
            data={
                "batch_id": batch_id,
                "round": round_number,
                "observation_count": len(new_batch["observations"]),
                "scientific_eligible": origin["scientific_eligible"],
            },
            artifacts=[dataset_ref, index_ref, notebook_ref],
        )

    @staticmethod
    def _batch_id(design: dict[str, Any], round_number: int) -> str:
        """Resolve the shared B-number prefix from the approved plan recipe IDs."""
        recipes = design.get("recipes", [])
        if not recipes or "-N" not in str(recipes[0].get("recipe_id", "")):
            raise ValueError("experiment design has no CPWL recipe IDs")
        batch_id = str(recipes[0]["recipe_id"]).split("-N", maxsplit=1)[0]
        expected_batch_id = f"B{round_number}"
        if batch_id != expected_batch_id:
            raise ValueError(f"plan batch {batch_id} does not match measurement round {round_number}")
        return batch_id

    @staticmethod
    def _origin(manifest: dict[str, Any]) -> dict[str, Any]:
        """Validate explicit source eligibility before it can affect the scientific dataset."""
        origin = manifest.get("origin")
        if not isinstance(origin, dict) or origin.get("kind") not in {"measured", "synthetic_dry_run"}:
            raise ValueError("spectra manifest has no valid explicit data origin")
        kind = str(origin["kind"])
        eligible = kind == "measured"
        if bool(origin.get("scientific_eligible")) != eligible:
            raise ValueError("spectra manifest origin eligibility is inconsistent")
        return {
            "kind": kind,
            "scientific_eligible": eligible,
            "submitted_path": str(manifest.get("source_path", "")),
            "notice_artifact": origin.get("notice_artifact"),
        }

    def _load_or_create_dataset(
        self, dataset_path: str, arguments: dict[str, Any], design: dict[str, Any]
    ) -> dict[str, Any]:
        """Keep one small JSON data source for exactly one task, without a database."""
        path = self.store.artifact_path(dataset_path)
        if path.is_file():
            dataset = _load_json(self.store, dataset_path)
            if dataset.get("task", {}).get("task_id") != arguments["task_id"]:
                raise ValueError("research dataset belongs to another task")
            return dataset
        conditions = {
            "emission_response_corrected": True,
            "excitation_wavelength_nm": 350,
            "solvent_or_host": None,
            "temperature_c": 25,
        }
        return {
            "schema_version": 1,
            "task": {
                "task_id": arguments["task_id"],
                "target_cie": list(arguments["target"]),
                "max_rounds": int(arguments["max_rounds"]),
                "materials": [
                    {
                        "slot": material["slot"],
                        "name": material["name"],
                        "pubchem_cid": int(material["material_key"]),
                        "molecular_weight_g_mol": material["molecular_weight_g_mol"],
                    }
                    for material in design["materials"]
                ],
                "conditions": conditions,
            },
            "data_policy": {
                "scope": "current_task_only",
                "scientific_eligible_origins": ["measured"],
                "raw_spectra_embedded": False,
            },
            "batches": [],
        }

    def _build_batch(
        self,
        *,
        batch_id: str,
        round_number: int,
        origin: dict[str, Any],
        design: dict[str, Any],
        manifest: dict[str, Any],
        measurement: dict[str, Any],
        analysis: dict[str, Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Join stable per-sample facts from the plan, spectra, CIE, and analysis artifacts."""
        recipes = {str(recipe["recipe_id"]): recipe for recipe in design["recipes"]}
        spectra = {str(sample["sample_id"]): sample for sample in manifest["samples"]}
        measurements = {str(sample["sample_id"]): sample for sample in measurement["measurements"]}
        analyses = {str(sample["sample_id"]): sample for sample in analysis["candidates"]}
        expected_ids = {f"{recipe_id}-D" for recipe_id in recipes}
        if set(spectra) != expected_ids or set(measurements) != expected_ids or set(analyses) != expected_ids:
            raise ValueError("round artifacts do not contain exactly the plan detection samples")
        observations = [
            self._build_observation(
                sample_id=sample_id,
                recipe=recipes[sample_id.removesuffix("-D")],
                spectrum=spectra[sample_id],
                measurement=measurements[sample_id],
                analysis=analyses[sample_id],
                target=list(arguments["target"]),
            )
            for sample_id in sorted(expected_ids, key=_sample_sort_key)
        ]
        return {
            "batch_id": batch_id,
            "round": round_number,
            "origin": origin,
            "plan": {
                "plan_id": design["plan_id"],
                "xlsx_artifact": str(arguments["experiment_plan"]),
                "design_artifact": str(arguments["experiment_plan"]).replace(".xlsx", "_design.json"),
            },
            "observation_artifacts": {
                "spectra_manifest": str(arguments["spectra_artifact"]),
                "measurement_result": str(arguments["measurement_artifact"]),
                "analysis_result": str(arguments["analysis_artifact"]),
                "goal_check": None,
            },
            "observations": observations,
        }

    @staticmethod
    def _build_observation(
        *,
        sample_id: str,
        recipe: dict[str, Any],
        spectrum: dict[str, Any],
        measurement: dict[str, Any],
        analysis: dict[str, Any],
        target: list[float],
    ) -> dict[str, Any]:
        """Produce one self-contained fact record without duplicating full spectrum arrays."""
        emission_intensities = [float(value) for value in spectrum["emission_intensities"]]
        absorption_qc = dict(
            spectrum.get(
                "absorption_qc",
                {
                    "status": "usable",
                    "reasons": [],
                    "scientific_use": "qualified for absorption-derived analysis",
                },
            )
        )
        absorption_status = str(absorption_qc.get("status", "not_provided"))
        absorption_usable = absorption_status == "usable"
        # Partial files may contribute the finite maximum only; they never enter full spectral analysis.
        absorption_summary_eligible = absorption_status in {"usable", "partial"}
        absorption_values = [
            float(value)
            for value in (spectrum.get("absorbance") or [])
            if value is not None
        ]
        absorption_grid = (
            _grid(spectrum["absorption_wavelengths_nm"])
            if spectrum.get("absorption_wavelengths_nm")
            else None
        )
        # Preserve sub-nanometre peak positions from the qualified raw emission spectrum.
        emission_wavelengths = [float(value) for value in spectrum["emission_wavelengths_nm"]]
        peak_index = max(range(len(emission_intensities)), key=emission_intensities.__getitem__)
        distance = float(analysis["distance"])
        return {
            "identity": {
                "recipe_id": recipe["recipe_id"],
                "sample_id": sample_id,
                "design_role": recipe["purpose"],
            },
            "recipe": {
                "concentrations_mmol_ml": recipe["concentrations_mmol_ml"],
                "molar_fractions": recipe["molar_fractions"],
                "total_concentration_mmol_ml": recipe["total_concentration_mmol_ml"],
                "stock_volumes_ml": recipe["stock_volumes_ml"],
                "solvent_volume_ml": recipe["solvent_volume_ml"],
            },
            "design_intent": {
                "purpose": recipe["purpose"],
                # Pre-upgrade plans remain ingestible without inventing missing Agent reasoning.
                "discrimination": recipe.get("discrimination"),
            },
            "evidence": {
                "emission_raw_path": spectrum["raw_emission_path"],
                "absorption_raw_path": spectrum["raw_absorption_path"],
                "emission_sha256": spectrum["emission_sha256"],
                "absorption_sha256": spectrum["absorption_sha256"],
                "emission_grid_nm": _grid(spectrum["emission_wavelengths_nm"]),
                "absorption_grid_nm": absorption_grid,
                "absorption_qc": absorption_qc,
            },
            "measurement": {
                "observer": measurement["observer"],
                "xyz_relative": measurement["xyz_relative"],
                "cie": measurement["cie"],
            },
            "evaluation": {
                "target_cie": target,
                "distance": distance,
                "within_tolerance": cie_within_coordinate_tolerance(measurement["cie"], target),
            },
            "qc": {
                "data_contract": (
                    "passed" if absorption_usable else "passed_with_optional_absorption_unavailable"
                ),
                "emission_peak_nm": emission_wavelengths[peak_index],
                "emission_max_intensity": emission_intensities[peak_index],
                "absorption_max_absorbance": (
                    max(absorption_values)
                    if absorption_summary_eligible and absorption_values
                    else None
                ),
                "absorption": absorption_qc,
                "flags": (
                    [] if absorption_usable else [
                        "optional absorption unavailable for full analysis: "
                        + "; ".join(str(reason) for reason in absorption_qc.get("reasons", []))
                    ]
                ),
            },
        }

    @staticmethod
    def _merge_batch(dataset: dict[str, Any], incoming: dict[str, Any]) -> None:
        """Allow idempotent recovery while rejecting any attempt to rewrite scientific facts."""
        existing = next((batch for batch in dataset["batches"] if batch["batch_id"] == incoming["batch_id"]), None)
        if existing is None:
            dataset["batches"].append(incoming)
            dataset["batches"].sort(key=lambda batch: int(batch["round"]))
            return
        if _canonical_sha256(existing) != _canonical_sha256(incoming):
            raise ValueError("existing batch has different source facts and cannot be overwritten")


class QueryResearchIndexTool:
    """Return compact current-task CIE matches without exposing arbitrary artifact paths."""

    name = "query_research_index"

    def __init__(self, store: TaskStore) -> None:
        """Bind controlled index queries to a single task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Filter and rank compact CIE entries, excluding simulations by default."""
        index = _load_json(self.store, str(arguments["research_index_artifact"]))
        include_synthetic = bool(arguments.get("include_synthetic", False))
        target = arguments.get("target_cie")
        if target is not None and (not isinstance(target, list) or len(target) != 2):
            raise ValueError("target_cie must contain exactly two coordinates")
        limit = int(arguments.get("limit", 12))
        if limit < 1 or limit > 100:
            raise ValueError("limit must be within 1..100")
        active_component_count = arguments.get("active_component_count")
        if active_component_count is not None:
            active_component_count = int(active_component_count)
            if active_component_count < 1 or active_component_count > 5:
                raise ValueError("active_component_count must be within 1..5")
        matches: list[dict[str, Any]] = []
        for entry in index["cie_index"]:
            if not include_synthetic and not entry["scientific_eligible"]:
                continue
            if arguments.get("batch_id") and entry["batch_id"] != arguments["batch_id"]:
                continue
            if arguments.get("design_role") and entry["design_role"] != arguments["design_role"]:
                continue
            # Derive the count for pre-upgrade run indexes so an active run can resume
            # without mutating its already-recorded scientific dataset.
            entry_component_count = int(
                entry.get(
                    "active_component_count",
                    sum(float(value) > 0 for value in entry["recipe"]["concentrations_mmol_ml"]),
                )
            )
            if active_component_count is not None and entry_component_count != active_component_count:
                continue
            item = dict(entry)
            item["active_component_count"] = entry_component_count
            if target is not None:
                item["query_distance"] = math.dist(item["cie"], [float(target[0]), float(target[1])])
                if arguments.get("max_distance") is not None and item["query_distance"] > float(arguments["max_distance"]):
                    continue
            matches.append(item)
        matches.sort(
            key=lambda item: (
                item.get("query_distance", float("inf")),
                int(item["round"]),
                _sample_sort_key(str(item["sample_id"])),
            )
        )
        return ToolResult(
            status="success",
            summary=f"Found {len(matches[:limit])} indexed observations.",
            data={"matches": matches[:limit], "source_dataset_sha256": index["source_dataset_sha256"]},
        )


class GetExperimentRecordTool:
    """Return one complete observation selected by a stable sample or recipe identifier."""

    name = "get_experiment_record"

    def __init__(self, store: TaskStore) -> None:
        """Bind record lookups to the current task dataset only."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Look up an exact record while enforcing simulation visibility policy."""
        sample_id = str(arguments.get("sample_id", "")).strip()
        recipe_id = str(arguments.get("recipe_id", "")).strip()
        if bool(sample_id) == bool(recipe_id):
            raise ValueError("provide exactly one of sample_id or recipe_id")
        include_synthetic = bool(arguments.get("include_synthetic", False))
        dataset = _load_json(self.store, str(arguments["research_dataset_artifact"]))
        for batch in dataset["batches"]:
            if not include_synthetic and not batch["origin"]["scientific_eligible"]:
                continue
            for observation in batch["observations"]:
                identity = observation["identity"]
                if identity["sample_id"] == sample_id or identity["recipe_id"] == recipe_id:
                    return ToolResult(
                        status="success",
                        summary=f"Loaded experiment record {identity['sample_id']}.",
                        data={
                            "task": dataset["task"],
                            # Keep the external submission location internal to the dataset artifact.
                            "origin": {
                                "kind": batch["origin"]["kind"],
                                "scientific_eligible": batch["origin"]["scientific_eligible"],
                                "notice_artifact": batch["origin"]["notice_artifact"],
                            },
                            "observation_artifacts": batch["observation_artifacts"],
                            "observation": observation,
                        },
                    )
        return ToolResult(status="not_found", summary="No permitted experiment record matched the identifier.")


class GetSpectrumDataTool:
    """Read archived raw spectra only through a sample record already present in the dataset."""

    name = "get_spectrum_data"

    def __init__(self, store: TaskStore) -> None:
        """Bind controlled raw-spectrum retrieval to one task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Return requested wavelength points without accepting a model-supplied filesystem path."""
        kind = str(arguments.get("kind", ""))
        if kind not in {"emission", "absorption"}:
            raise ValueError("kind must be emission or absorption")
        record = GetExperimentRecordTool(self.store).run(
            {
                "research_dataset_artifact": arguments["research_dataset_artifact"],
                "sample_id": arguments["sample_id"],
                "include_synthetic": bool(arguments.get("include_synthetic", False)),
            }
        )
        if record.status != "success":
            return record
        evidence = record.data["observation"]["evidence"]
        raw_reference = evidence.get(f"{kind}_raw_path")
        if not raw_reference:
            return ToolResult(
                status="not_found",
                summary=f"No {kind} spectrum was provided for {arguments['sample_id']}.",
                data={"sample_id": arguments["sample_id"], "kind": kind},
            )
        raw_path = self.store.artifact_path(str(raw_reference))
        if kind == "emission":
            wavelengths, values = _parse_emission(raw_path)
            points = [{"wavelength_nm": wavelength, "intensity": value} for wavelength, value in zip(wavelengths, values)]
            digest = evidence["emission_sha256"]
        else:
            dataset_qc = evidence.get("absorption_qc", {"status": "usable", "reasons": []})
            wavelengths, _transmittance, absorbance, parsed_qc = _parse_absorption(raw_path)
            effective_qc = parsed_qc if parsed_qc.get("status") == "excluded" else dataset_qc
            if effective_qc.get("status") != "usable":
                return ToolResult(
                    status="failed",
                    summary=(
                        f"Absorption spectrum for {arguments['sample_id']} is not usable for full analysis: "
                        + "; ".join(str(reason) for reason in effective_qc.get("reasons", []))
                    ),
                    data={
                        "sample_id": arguments["sample_id"],
                        "kind": kind,
                        "absorption_qc": effective_qc,
                    },
                )
            points = [{"wavelength_nm": wavelength, "absorbance": value} for wavelength, value in zip(wavelengths, absorbance)]
            digest = evidence["absorption_sha256"]
        start = float(arguments.get("start_nm", min(wavelengths)))
        end = float(arguments.get("end_nm", max(wavelengths)))
        if start > end:
            raise ValueError("start_nm must not exceed end_nm")
        selected = [point for point in points if start <= point["wavelength_nm"] <= end]
        return ToolResult(
            status="success",
            summary=f"Loaded {len(selected)} {kind} spectrum points for {arguments['sample_id']}.",
            data={"sample_id": arguments["sample_id"], "kind": kind, "sha256": digest, "points": selected},
        )


def _grid(wavelengths: list[int] | list[float]) -> list[float]:
    """Represent a regular wavelength grid without losing sub-nanometre resolution."""
    if len(wavelengths) < 2:
        raise ValueError("spectrum must contain at least two wavelength points")
    return [
        float(wavelengths[0]),
        float(wavelengths[-1]),
        round(float(wavelengths[1] - wavelengths[0]), 10),
    ]


def _sample_sort_key(sample_id: str) -> tuple[int, int]:
    """Order CPWL sample IDs numerically rather than lexicographically."""
    batch_text, recipe_text, _detection = sample_id.split("-")
    return int(batch_text.removeprefix("B")), int(recipe_text.removeprefix("N"))


def _build_research_index(dataset: dict[str, Any], dataset_ref: str) -> dict[str, Any]:
    """Build the small, transparent projection used by CIE and recipe queries."""
    entries: list[dict[str, Any]] = []
    sample_index: dict[str, dict[str, Any]] = {}
    for batch in dataset["batches"]:
        for observation in batch["observations"]:
            identity = observation["identity"]
            entry = {
                "sample_id": identity["sample_id"],
                "batch_id": batch["batch_id"],
                "round": batch["round"],
                "origin": batch["origin"]["kind"],
                "scientific_eligible": batch["origin"]["scientific_eligible"],
                "recipe_id": identity["recipe_id"],
                "design_role": identity["design_role"],
                # Composition cardinality is stable query metadata, unlike free-text design purpose.
                "active_component_count": sum(
                    float(value) > 0 for value in observation["recipe"]["concentrations_mmol_ml"]
                ),
                "cie": observation["measurement"]["cie"],
                "recipe": {
                    key: observation["recipe"][key]
                    for key in ("concentrations_mmol_ml", "molar_fractions", "total_concentration_mmol_ml")
                },
            }
            entries.append(entry)
            sample_index[identity["sample_id"]] = {
                key: entry[key]
                for key in ("batch_id", "round", "origin", "scientific_eligible", "recipe_id")
            }
    return {
        "schema_version": 1,
        "source_dataset": dataset_ref,
        "source_dataset_sha256": _canonical_sha256(dataset),
        "sample_index": sample_index,
        "cie_index": entries,
    }


def _render_notebook(dataset: dict[str, Any], index: dict[str, Any]) -> str:
    """Render a compact human-readable projection without creating another fact source."""
    lines = [
        "# Research Notebook",
        "",
        f"- Task: `{dataset['task']['task_id']}`",
        f"- Target CIE: `{dataset['task']['target_cie']}`",
        f"- Dataset SHA-256: `{index['source_dataset_sha256']}`",
        "",
        "| Batch | Origin | Observations | Best CIE distance |",
        "| --- | --- | ---: | ---: |",
    ]
    for batch in dataset["batches"]:
        best_distance = min(item["evaluation"]["distance"] for item in batch["observations"])
        lines.append(
            f"| {batch['batch_id']} | {batch['origin']['kind']} | {len(batch['observations'])} | {best_distance:.6f} |"
        )
    lines.extend(["", "## 配方辨别意图", ""])
    for batch in dataset["batches"]:
        lines.extend([
            f"### {batch['batch_id']}",
            "",
            "| Sample | Purpose | Hypothesis | Reference samples | If supported | If not supported |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for observation in batch["observations"]:
            intent = observation["design_intent"]
            discrimination = intent["discrimination"]
            if discrimination is None:
                lines.append(
                    f"| {observation['identity']['sample_id']} | {intent['purpose']} | not recorded | not recorded | not recorded | not recorded |"
                )
                continue
            references = ", ".join(discrimination["reference_sample_ids"]) or "initial batch; none"
            lines.append(
                "| {sample_id} | {purpose} | {hypothesis} | {references} | {supported} | {not_supported} |".format(
                    sample_id=observation["identity"]["sample_id"],
                    purpose=intent["purpose"],
                    hypothesis=discrimination["hypothesis"],
                    references=references,
                    supported=discrimination["outcome_if_supported"],
                    not_supported=discrimination["outcome_if_not_supported"],
                )
            )
        lines.append("")
    lines.extend(["", "This file is derived from `research_dataset.json` and is not a scientific fact source.", ""])
    return "\n".join(lines)
