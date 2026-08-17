"""Bounded online evidence tools and deterministic local experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from react_color_agent.cie import CIE_TOLERANCE, cie_euclidean_distance, cie_within_coordinate_tolerance
from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore
from skills.cpwl.experiment_plan_xlsx.cpwl_xlsx import (
    CONCENTRATION_TOLERANCE_MMOL_ML,
    MAX_COMPONENT_CONCENTRATION_MMOL_ML,
    CpwlDiscrimination,
    CpwlMaterial,
    build_followup_cpwl_batch,
    build_initial_cpwl_batch,
    write_cpwl_xlsx_artifacts,
)

JsonTransport = Callable[[str], dict[str, Any]]


def _parse_discrimination(value: Any) -> CpwlDiscrimination:
    """Validate one Agent-authored recipe decision boundary before it becomes an artifact."""
    if not isinstance(value, dict):
        raise ValueError("each recipe requires a discrimination object")
    try:
        return CpwlDiscrimination.model_validate(value)
    except ValueError as error:
        raise ValueError(f"invalid recipe discrimination: {error}") from error


def _measurement_return_directory(artifacts: list[str]) -> str:
    """Find the empty, run-local data-return directory created with a CPWL plan."""
    return next(
        artifact for artifact in artifacts if artifact.startswith("artifacts/measurement_returns/")
    )


def _get_json(url: str) -> dict[str, Any]:
    """Fetch JSON from a fixed tool endpoint using only the standard library."""
    with urlopen(url, timeout=15) as response:  # noqa: S310 - callers construct only fixed hosts.
        return json.loads(response.read().decode("utf-8"))


class PubChemTool:
    """Resolve a chemical name through the PubChem PUG REST name endpoint."""

    name = "query_pubchem"

    def __init__(self, transport: JsonTransport = _get_json) -> None:
        """Accept an injectable transport so tests do not issue network requests."""
        self.transport = transport

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Return an identity record or a structured public-source failure."""
        name = str(arguments["name"]).strip()
        if not name:
            raise ValueError("PubChem name must not be empty")
        properties = "IUPACName,MolecularFormula,MolecularWeight,ConnectivitySMILES,InChIKey"
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{quote(name, safe='')}/property/{properties}/JSON"
        )
        try:
            payload = self.transport(url)
        except HTTPError as error:
            if error.code == 404:
                return ToolResult(status="not_found", summary=f"PubChem found no identity for {name}.")
            return ToolResult(status="network_error", summary=f"PubChem HTTP error {error.code} for {name}.")
        except (URLError, TimeoutError, OSError) as error:
            return ToolResult(status="network_error", summary=f"PubChem network error for {name}: {error}")

        properties_list = payload.get("PropertyTable", {}).get("Properties", [])
        if len(properties_list) != 1:
            return ToolResult(status="ambiguous", summary=f"PubChem did not return one identity for {name}.")
        property_record = properties_list[0]
        cid = property_record.get("CID")
        if cid is None:
            return ToolResult(status="failed", summary=f"PubChem response for {name} lacks a CID.")

        data = {
            "material": str(arguments.get("material", name)),
            "query_name": name,
            "cid": cid,
            "pubchem_name": property_record.get("IUPACName"),
            "molecular_formula": property_record.get("MolecularFormula"),
            "molecular_weight": property_record.get("MolecularWeight"),
            "connectivity_smiles": property_record.get("ConnectivitySMILES"),
            "inchikey": property_record.get("InChIKey"),
            "source_url": url,
        }
        return ToolResult(status="success", summary=f"PubChem confirmed CID {cid} for {name}.", data=data)


class CrossrefTool:
    """Search a small Crossref result set only after PubChem identity is confirmed."""

    name = "search_crossref"

    def __init__(self, transport: JsonTransport = _get_json) -> None:
        """Accept an injectable transport so tests keep the online boundary closed."""
        self.transport = transport

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Return up to three literature references from the fixed Crossref API."""
        if not arguments.get("identity_confirmed"):
            raise ValueError("Crossref requires confirmed PubChem identity")
        if not arguments.get("information_insufficient"):
            raise ValueError("Crossref requires insufficient PubChem information")
        query = str(arguments["query"]).strip()
        parameters = urlencode(
            {
                "query.bibliographic": query,
                "rows": 3,
                "select": "DOI,title,published,URL",
            }
        )
        url = f"https://api.crossref.org/works?{parameters}"
        try:
            payload = self.transport(url)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            return ToolResult(status="network_error", summary=f"Crossref network error for {query}: {error}")

        references = [
            {
                "doi": item.get("DOI"),
                "title": (item.get("title") or [None])[0],
                "published": item.get("published"),
                "url": item.get("URL"),
            }
            for item in payload.get("message", {}).get("items", [])[:3]
        ]
        return ToolResult(
            status="success",
            summary=f"Crossref returned {len(references)} references for {query}.",
            data={"query": query, "references": references, "source_url": url},
        )


class SaveMaterialEvidenceTool:
    """Persist confirmed public identities and explicitly list missing optical evidence."""

    name = "save_material_evidence"

    def __init__(self, store: TaskStore) -> None:
        """Bind evidence artifacts to a single task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Write only supplied source data and stated measurement gaps to an artifact."""
        supplied_materials = arguments.get("materials") or self._stored_pubchem_materials()
        entries: list[dict[str, Any]] = []
        for supplied in supplied_materials:
            pubchem = supplied.get("pubchem", {})
            if pubchem.get("cid") is None:
                raise ValueError("every evidence entry requires a confirmed PubChem CID")
            entries.append(
                {
                    "user_name": supplied["user_name"],
                    "pubchem": pubchem,
                    "crossref": supplied.get("crossref", []),
                    # These are required measurements, not inferred material values.
                    "missing_measurements": [
                        "absorption spectrum",
                        "emission spectrum",
                        "quantum yield",
                        "solubility in experiment solvent",
                    ],
                }
            )
        artifact = self.store.write_artifact_json(
            "artifacts/evidence/materials.json", {"materials": entries}
        )
        return ToolResult(
            status="success",
            summary=f"Saved PubChem evidence for {len(entries)} materials.",
            data={"confirmed_materials": [entry["user_name"] for entry in entries]},
            artifacts=[artifact],
        )

    def _stored_pubchem_materials(self) -> list[dict[str, Any]]:
        """Recover PubChem observations in the stable user task material order."""
        query_dir = self.store.artifact_path("artifacts/evidence/queries")
        if not query_dir.is_dir():
            return []
        by_material: dict[str, dict[str, Any]] = {}
        for query_path in sorted(query_dir.glob("*.json")):
            pubchem = json.loads(query_path.read_text(encoding="utf-8"))
            material = pubchem.get("material")
            if material:
                by_material[material] = {"user_name": material, "pubchem": pubchem}
        task_order = self.store.load().materials
        ordered = [by_material[name] for name in task_order if name in by_material]
        # Retain unexpected legacy query records after the declared task materials;
        # normal runs never rely on filesystem hash order for workbook slots.
        extras = [entry for name, entry in by_material.items() if name not in task_order]
        return [*ordered, *extras]


class DesignInitialBatchTool:
    """Validate and write an Agent-designed first CPWL screening batch."""

    name = "design_initial_batch"

    def __init__(self, store: TaskStore) -> None:
        """Bind generated plan artifacts to one task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Write the Agent-proposed B1 recipes after evidence and physical-contract validation."""
        evidence_path = self.store.artifact_path(str(arguments["evidence_artifact"]))
        if not evidence_path.is_file():
            raise ValueError("material evidence artifact does not exist")
        round_number = int(arguments.get("round", 0))
        if round_number != 0:
            raise ValueError("design_initial_batch only creates B1; use design_followup_batch for later rounds")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        materials = [
            CpwlMaterial(
                slot="ABCDE"[index],
                material_key=str(entry["pubchem"]["cid"]),
                name=str(entry["user_name"]),
                molecular_weight_g_mol=float(entry["pubchem"]["molecular_weight"]),
            )
            for index, entry in enumerate(evidence["materials"])
        ]
        scientific_rationale = str(arguments.get("scientific_rationale", "")).strip()
        if not scientific_rationale:
            raise ValueError("scientific_rationale must explain the initial screening design")
        agent_recipes = self._validate_agent_recipes(arguments.get("recipes"))
        plan = build_initial_cpwl_batch(
            run_id=self.store.load().task_id,
            target_cie=tuple(arguments["target"]),
            materials=materials,
            agent_recipes=agent_recipes,
            scientific_rationale=scientific_rationale,
            applied_skills=list(arguments.get("applied_skills", [])),
        )
        artifacts = write_cpwl_xlsx_artifacts(self.store, plan)
        measurement_return_directory = _measurement_return_directory(artifacts)
        return ToolResult(
            status="success",
            summary=(
                f"Created and validated Agent-designed CPWL XLSX plan for experiment round {round_number + 1}; "
                f"empty return directories are ready at {measurement_return_directory}."
            ),
            data={
                "round": round_number + 1,
                "recipe_count": len(plan.recipes),
                "recipe_source": "agent",
                "measurement_return_directory": measurement_return_directory,
            },
            artifacts=artifacts,
        )

    @staticmethod
    def _validate_agent_recipes(
        value: Any,
    ) -> list[tuple[tuple[float, float, float, float, float], str, CpwlDiscrimination]]:
        """Accept Agent-authored scientific choices while keeping the numerical workbook boundary deterministic."""
        if not isinstance(value, list) or not 1 <= len(value) <= 24 or not all(isinstance(item, dict) for item in value):
            raise ValueError("initial recipes must contain one to twenty-four recipe objects")
        recipes = []
        for item in value:
            raw_concentrations = item.get("concentrations_mmol_ml")
            purpose = str(item.get("purpose", "")).strip()
            if not isinstance(raw_concentrations, list) or len(raw_concentrations) != 5:
                raise ValueError("each initial recipe must provide exactly five concentrations")
            if not purpose:
                raise ValueError("each initial recipe requires a non-empty scientific purpose")
            discrimination = _parse_discrimination(item.get("discrimination"))
            if discrimination.reference_sample_ids:
                raise ValueError("initial recipe discrimination must not cite historical samples")
            try:
                concentrations = tuple(float(component) for component in raw_concentrations)
            except (TypeError, ValueError) as error:
                raise ValueError("initial recipe concentrations must be numeric") from error
            recipes.append((concentrations, purpose, discrimination))
        return recipes  # type: ignore[return-value]


class DesignFollowupBatchTool:
    """Turn Agent-selected interpolation candidates into one bounded CPWL follow-up batch."""

    name = "design_followup_batch"

    def __init__(self, store: TaskStore) -> None:
        """Bind candidate selection and generated CPWL artifacts to the current task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate candidate lineage, then write B{round+1} without treating predictions as measurements."""
        candidate_artifact = str(arguments["predicted_candidates_artifact"])
        dataset_artifact = str(arguments["research_dataset_artifact"])
        candidates = self._load_json(candidate_artifact)
        dataset = self._load_json(dataset_artifact)
        round_number = int(arguments["round"])
        if int(candidates["round"]) != round_number:
            raise ValueError("candidate artifact does not belong to the current completed experiment round")
        if candidates["source_dataset_sha256"] != self._dataset_sha256(dataset):
            raise ValueError("candidate artifact does not match the current research dataset")
        if candidates["status"] != "PREDICTED_CANDIDATES_NOT_MEASURED" or candidates["origin_policy"] != "measured_only":
            raise ValueError("follow-up plans require measured-only predicted candidate artifacts")
        candidate_ids = arguments.get("candidate_ids")
        if not isinstance(candidate_ids, list) or not 1 <= len(candidate_ids) <= 12 or not all(
            isinstance(item, str) and item.strip() for item in candidate_ids
        ):
            raise ValueError("candidate_ids must contain one to twelve non-empty candidate IDs")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must not contain duplicates")
        available = {str(item["candidate_id"]): item for item in candidates["candidates"]}
        missing = [item for item in candidate_ids if item not in available]
        if missing:
            raise ValueError("candidate_ids are absent from the controlled candidate artifact: " + ", ".join(missing))
        selected = [available[item] for item in candidate_ids]
        ineligible = [
            item["candidate_id"]
            for item in selected
            if item["status"] != "PREDICTED_CANDIDATE"
            or item["coverage"]["classification"] != "interpolation"
            or not item["selection_eligible"]
        ]
        if ineligible:
            raise ValueError("only unmeasured interpolation candidates can enter a follow-up batch: " + ", ".join(ineligible))
        discriminations = self._validate_candidate_discriminations(
            arguments.get("candidate_discriminations"),
            candidate_ids,
            self._measured_sample_ids(dataset),
        )
        selection_reason = str(arguments["selection_reason"]).strip()
        if not selection_reason:
            raise ValueError("selection_reason must explain the Agent's chosen candidate subset")
        batch_number = round_number + 1
        materials = [
            CpwlMaterial(
                slot=str(material["slot"]),
                material_key=str(material["pubchem_cid"]),
                name=str(material["name"]),
                molecular_weight_g_mol=float(material["molecular_weight_g_mol"]),
            )
            for material in dataset["task"]["materials"]
        ]
        selection_artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/candidate_selection.json",
            {
                "schema_version": 1,
                "round": round_number,
                "next_batch": f"B{batch_number}",
                "source_predicted_candidates": candidate_artifact,
                "source_dataset_sha256": candidates["source_dataset_sha256"],
                "selected_method": candidates["selected_method"],
                "selection_reason": selection_reason,
                "selected_candidates": selected,
                "candidate_discriminations": {
                    candidate_id: discrimination.model_dump()
                    for candidate_id, discrimination in discriminations.items()
                },
                "scientific_status": "PREDICTED_CANDIDATES_PENDING_MEASUREMENT",
            },
        )
        plan = build_followup_cpwl_batch(
            run_id=self.store.load().task_id,
            target_cie=tuple(float(value) for value in arguments["target"]),
            materials=materials,
            batch_number=batch_number,
            candidate_recipes=[
                (
                    tuple(float(value) for value in item["concentrations_mmol_ml"]),
                    "model_predicted_candidate",
                    discriminations[str(item["candidate_id"])],
                )
                for item in selected
            ],
            applied_skills=list(arguments.get("applied_skills", [])),
            design_context={
                "strategy": "Agent-selected local-model interpolation candidates pending measurement",
                "source_predicted_candidates": candidate_artifact,
                "selection_artifact": selection_artifact,
                "selected_candidate_ids": candidate_ids,
            },
        )
        plan_artifacts = write_cpwl_xlsx_artifacts(self.store, plan)
        measurement_return_directory = _measurement_return_directory(plan_artifacts)
        return ToolResult(
            status="success",
            summary=(
                f"Created and validated CPWL B{batch_number} XLSX plan with {len(selected)} predicted "
                f"interpolation candidates pending measurement; empty return directories are ready at "
                f"{measurement_return_directory}."
            ),
            data={
                "round": batch_number,
                "recipe_count": len(selected),
                "selected_candidate_ids": candidate_ids,
                "measurement_return_directory": measurement_return_directory,
            },
            artifacts=[*plan_artifacts, selection_artifact],
        )

    def _load_json(self, artifact: str) -> dict[str, Any]:
        """Read a run-local JSON artifact without accepting arbitrary external paths."""
        try:
            return json.loads(self.store.artifact_path(artifact).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load JSON artifact {artifact}") from error

    @staticmethod
    def _dataset_sha256(dataset: dict[str, Any]) -> str:
        """Produce the shared stable hash used by candidate artifacts."""
        encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _measured_sample_ids(dataset: dict[str, Any]) -> set[str]:
        """Expose only real measurements as valid references for a future formulation."""
        return {
            str(observation["identity"]["sample_id"])
            for batch in dataset["batches"]
            if batch["origin"]["scientific_eligible"]
            for observation in batch["observations"]
        }

    @staticmethod
    def _validate_candidate_discriminations(
        value: Any,
        candidate_ids: list[str],
        measured_sample_ids: set[str],
    ) -> dict[str, CpwlDiscrimination]:
        """Match every selected prediction to one falsifiable, measured-data-aware intent."""
        if not isinstance(value, list) or len(value) != len(candidate_ids) or not all(isinstance(item, dict) for item in value):
            raise ValueError("candidate_discriminations must provide one object for every selected candidate")
        parsed: dict[str, CpwlDiscrimination] = {}
        for item in value:
            candidate_id = str(item.get("candidate_id", "")).strip()
            if candidate_id not in candidate_ids or candidate_id in parsed:
                raise ValueError("candidate_discriminations must use each selected candidate ID exactly once")
            discrimination = _parse_discrimination(item.get("discrimination"))
            if not discrimination.reference_sample_ids:
                raise ValueError("follow-up recipe discrimination requires at least one measured reference sample")
            unknown = set(discrimination.reference_sample_ids) - measured_sample_ids
            if unknown:
                raise ValueError("recipe discrimination cites unknown measured samples: " + ", ".join(sorted(unknown)))
            parsed[candidate_id] = discrimination
        if set(parsed) != set(candidate_ids):
            raise ValueError("candidate_discriminations must cover every selected candidate")
        return parsed


class DesignExploratoryFollowupBatchTool:
    """Turn a measured-data coverage, diagnostic, or repeat decision into a bounded CPWL batch."""

    name = "design_exploratory_followup_batch"

    def __init__(self, store: TaskStore) -> None:
        """Bind exploratory recipe validation and plan output to the current research task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate an Agent-designed non-model experiment before writing the next CPWL workbook."""
        dataset = self._load_json(str(arguments["research_dataset_artifact"]))
        decision_artifact = str(arguments["design_decision_artifact"])
        decision = self._load_json(decision_artifact)
        round_number = int(arguments["round"])
        self._validate_lineage(dataset, decision, round_number)
        strategy = str(decision["strategy"])
        reference_sample_id = str(arguments["reference_sample_id"]).strip()
        eligible_samples = self._eligible_samples(dataset)
        reference = eligible_samples.get(reference_sample_id)
        if reference is None:
            raise ValueError("reference_sample_id must identify a measured sample in the current dataset")
        if reference_sample_id not in decision["facts_used_sample_ids"]:
            raise ValueError("reference_sample_id must be cited in the saved design decision")
        provisional = self.store.load().provisional_goal_candidate
        if provisional is not None:
            if strategy != "repeat_validation":
                raise ValueError("a provisional target hit requires repeat_validation")
            if reference_sample_id != provisional.sample_id:
                raise ValueError("repeat_validation must use the provisional target sample as reference")
        active_slots = self._validate_active_slots(arguments.get("active_slots"), len(dataset["task"]["materials"]))
        recipe_entries = self._validate_recipes(arguments.get("recipes"))
        recipe_values = [entry[0] for entry in recipe_entries]
        self._validate_discrimination_references(
            [entry[1] for entry in recipe_entries],
            set(eligible_samples),
            set(decision["facts_used_sample_ids"]),
        )
        reference_values = tuple(float(value) for value in reference["recipe"]["concentrations_mmol_ml"])
        existing = {
            self._recipe_key(observation["recipe"]["concentrations_mmol_ml"])
            for observation in eligible_samples.values()
        }
        self._validate_strategy_recipes(
            strategy=strategy,
            recipes=recipe_values,
            reference=reference_values,
            active_slots=active_slots,
            existing=existing,
        )
        selection_reason = str(arguments["selection_reason"]).strip()
        if not selection_reason:
            raise ValueError("selection_reason must explain the controlled exploratory batch")
        batch_number = round_number + 1
        purpose = {
            "coverage": "coverage_exploration",
            "diagnostic": "diagnostic_control",
            "repeat_validation": "repeat_validation",
        }[strategy]
        selection_artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/exploratory_selection.json",
            {
                "schema_version": 1,
                "round": round_number,
                "next_batch": f"B{batch_number}",
                "strategy": strategy,
                "source_design_decision": decision_artifact,
                "source_dataset_sha256": self._dataset_sha256(dataset),
                "reference_sample_id": reference_sample_id,
                "active_slots": active_slots,
                "selection_reason": selection_reason,
                "selected_recipes": [
                    {
                        "concentrations_mmol_ml": list(recipe),
                        "purpose": purpose,
                        "discrimination": discrimination.model_dump(),
                    }
                    for recipe, discrimination in recipe_entries
                ],
                "scientific_status": "EXPLORATORY_RECIPES_PENDING_MEASUREMENT",
            },
        )
        materials = [
            CpwlMaterial(
                slot=str(material["slot"]),
                material_key=str(material["pubchem_cid"]),
                name=str(material["name"]),
                molecular_weight_g_mol=float(material["molecular_weight_g_mol"]),
            )
            for material in dataset["task"]["materials"]
        ]
        plan = build_followup_cpwl_batch(
            run_id=self.store.load().task_id,
            target_cie=tuple(float(value) for value in arguments["target"]),
            materials=materials,
            batch_number=batch_number,
            candidate_recipes=[
                (recipe, purpose, discrimination)
                for recipe, discrimination in recipe_entries
            ],
            applied_skills=list(arguments.get("applied_skills", [])),
            design_context={
                "strategy": f"Agent-selected {strategy} experiment pending measurement",
                "source_design_decision": decision_artifact,
                "selection_artifact": selection_artifact,
                "reference_sample_id": reference_sample_id,
                "active_slots": active_slots,
            },
        )
        plan_artifacts = write_cpwl_xlsx_artifacts(self.store, plan)
        measurement_return_directory = _measurement_return_directory(plan_artifacts)
        return ToolResult(
            status="success",
            summary=(
                f"Created and validated CPWL B{batch_number} XLSX plan with {len(recipe_values)} "
                f"{strategy} recipes pending measurement; empty return directories are ready at "
                f"{measurement_return_directory}."
            ),
            data={
                "round": batch_number,
                "recipe_count": len(recipe_values),
                "strategy": strategy,
                "reference_sample_id": reference_sample_id,
                "measurement_return_directory": measurement_return_directory,
            },
            artifacts=[*plan_artifacts, selection_artifact],
        )

    def _validate_lineage(self, dataset: dict[str, Any], decision: dict[str, Any], round_number: int) -> None:
        """Require a current measured-only, explicitly non-model decision for exploratory recipes."""
        if int(decision["round"]) != round_number:
            raise ValueError("design decision does not belong to the current completed experiment round")
        if decision["source_dataset_sha256"] != self._dataset_sha256(dataset):
            raise ValueError("design decision does not match the current research dataset")
        if decision["strategy"] not in {"coverage", "diagnostic", "repeat_validation"}:
            raise ValueError("exploratory plans require coverage, diagnostic, or repeat_validation strategy")
        if decision["selected_method"] != "none":
            raise ValueError("exploratory plans require a design decision with selected_method=none")

    @staticmethod
    def _eligible_samples(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Index only measured observations so a non-scientific batch cannot anchor a new plan."""
        return {
            str(observation["identity"]["sample_id"]): observation
            for batch in dataset["batches"]
            if batch["origin"]["scientific_eligible"]
            for observation in batch["observations"]
        }

    @staticmethod
    def _validate_active_slots(value: Any, material_count: int) -> list[str]:
        """Limit exploratory controls to one or two actual material slots in the current task."""
        if not isinstance(value, list) or not 1 <= len(value) <= 2 or not all(isinstance(item, str) for item in value):
            raise ValueError("active_slots must contain one or two material slots")
        slots = [item.strip().upper() for item in value]
        allowed = list("ABCDE"[:material_count])
        if len(set(slots)) != len(slots) or any(slot not in allowed for slot in slots):
            raise ValueError("active_slots must be unique slots used by the current task")
        return slots

    @staticmethod
    def _validate_recipes(
        value: Any,
    ) -> list[tuple[tuple[float, float, float, float, float], CpwlDiscrimination]]:
        """Parse a compact Agent-authored recipe list before physical workbook validation."""
        if not isinstance(value, list) or not 1 <= len(value) <= 12 or not all(isinstance(item, dict) for item in value):
            raise ValueError("follow-up recipes must contain one to twelve recipe objects")
        recipes = []
        for item in value:
            raw = item.get("concentrations_mmol_ml")
            if not isinstance(raw, list) or len(raw) != 5:
                raise ValueError("each exploratory recipe must provide exactly five concentrations")
            concentrations = tuple(float(component) for component in raw)
            if any(
                not math.isfinite(component)
                or component < 0
                or component
                > MAX_COMPONENT_CONCENTRATION_MMOL_ML + CONCENTRATION_TOLERANCE_MMOL_ML
                for component in concentrations
            ):
                raise ValueError(
                    "exploratory recipe concentrations must be finite values within "
                    f"0..{MAX_COMPONENT_CONCENTRATION_MMOL_ML:.6g} mmol/ml"
                )
            if sum(concentrations) <= 0:
                raise ValueError("each exploratory recipe needs at least one positive concentration")
            recipes.append((concentrations, _parse_discrimination(item.get("discrimination"))))
        return recipes  # type: ignore[return-value]

    @staticmethod
    def _validate_discrimination_references(
        discriminations: list[CpwlDiscrimination],
        measured_sample_ids: set[str],
        decision_sample_ids: set[str],
    ) -> None:
        """Require exploratory recipes to cite the measured evidence already used by the decision."""
        for discrimination in discriminations:
            references = set(discrimination.reference_sample_ids)
            if not references:
                raise ValueError("follow-up recipe discrimination requires at least one measured reference sample")
            unknown = references - measured_sample_ids
            if unknown:
                raise ValueError("recipe discrimination cites unknown measured samples: " + ", ".join(sorted(unknown)))
            uncited = references - decision_sample_ids
            if uncited:
                raise ValueError("recipe discrimination must cite samples used by the saved design decision: " + ", ".join(sorted(uncited)))

    def _validate_strategy_recipes(
        self,
        *,
        strategy: str,
        recipes: list[tuple[float, float, float, float, float]],
        reference: tuple[float, float, float, float, float],
        active_slots: list[str],
        existing: set[tuple[float, float, float, float, float]],
    ) -> None:
        """Preserve controlled-variable meaning and prevent accidental duplicate exploratory samples."""
        active_indices = {"ABCDE".index(slot) for slot in active_slots}
        keys = [self._recipe_key(recipe) for recipe in recipes]
        if strategy == "repeat_validation":
            if any(key != self._recipe_key(reference) for key in keys):
                raise ValueError("repeat_validation recipes must exactly repeat the measured reference formulation")
            return
        if len(set(keys)) != len(keys):
            raise ValueError("coverage and diagnostic recipes must not duplicate each other")
        duplicate_existing = [recipe for recipe, key in zip(recipes, keys) if key in existing]
        if duplicate_existing:
            raise ValueError("coverage and diagnostic recipes must be new relative to measured formulations")
        for recipe in recipes:
            if any(
                not math.isclose(recipe[index], reference[index], rel_tol=0, abs_tol=1e-12)
                for index in range(5)
                if index not in active_indices
            ):
                raise ValueError("inactive material concentrations must remain equal to the measured reference")
            if all(
                math.isclose(recipe[index], reference[index], rel_tol=0, abs_tol=1e-12)
                for index in active_indices
            ):
                raise ValueError("each coverage or diagnostic recipe must change at least one active material")

    @staticmethod
    def _recipe_key(values: list[float] | tuple[float, float, float, float, float]) -> tuple[float, float, float, float, float]:
        """Normalize JSON and floating-point recipe values for exact CPWL duplicate comparisons."""
        return tuple(round(float(value), 12) for value in values)  # type: ignore[return-value]

    def _load_json(self, artifact: str) -> dict[str, Any]:
        """Read a run-local JSON artifact without allowing an Agent-controlled file path."""
        try:
            return json.loads(self.store.artifact_path(artifact).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load JSON artifact {artifact}") from error

    @staticmethod
    def _dataset_sha256(dataset: dict[str, Any]) -> str:
        """Match the canonical research dataset hashing convention used by decision artifacts."""
        encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class AnalyzeResultsTool:
    """Select the measured sample closest to the CIE target using Euclidean distance."""

    name = "analyze_results"

    def __init__(self, store: TaskStore) -> None:
        """Bind analysis output to the active task directory."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Produce a deterministic per-round analysis from accepted laboratory data."""
        measurement_artifact = str(arguments["measurement_artifact"])
        measurement_path = self.store.artifact_path(measurement_artifact)
        payload = json.loads(measurement_path.read_text(encoding="utf-8"))
        target = tuple(float(component) for component in arguments["target"])
        if len(target) != 2:
            raise ValueError("target must contain exactly two CIE coordinates")
        candidates = [
            {
                **measurement,
                "distance": cie_euclidean_distance(measurement["cie"], target),
            }
            for measurement in payload["measurements"]
        ]
        best = min(candidates, key=lambda candidate: candidate["distance"])
        round_number = int(payload["round"])
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/analysis_result.json",
            {
                "round": round_number,
                "target_cie": list(target),
                "best_sample": best,
                "candidates": candidates,
            },
        )
        return ToolResult(
            status="success",
            summary=(
                f"Best sample is {best['sample_id']} at CIE distance {best['distance']:.6f}."
            ),
            data={"round": round_number, "best_distance": best["distance"]},
            artifacts=[artifact],
        )


class CheckGoalTool:
    """Require both an initial measured CIE hit and one independent exact-repeat hit."""

    name = "check_goal"

    def __init__(self, store: TaskStore, tolerance: float = CIE_TOLERANCE) -> None:
        """Use independent coordinate tolerances without making acceptance an LLM judgement."""
        if tolerance <= 0:
            raise ValueError("goal tolerance must be positive")
        self.store = store
        self.tolerance = tolerance

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Classify the current result as missed, provisional, confirmed, or failed repeat."""
        analysis_artifact = str(arguments["analysis_artifact"])
        analysis_path = self.store.artifact_path(analysis_artifact)
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        distance = float(analysis["best_sample"]["distance"])
        round_number = int(analysis["round"])
        batch = self._current_round_batch(arguments, round_number)
        scientific_eligible = bool(batch["origin"]["scientific_eligible"])
        provisional = self.store.load().provisional_goal_candidate
        provisional_payload = provisional.model_dump(mode="json") if provisional is not None else None
        confirmation_samples: list[dict[str, Any]] = []

        if provisional is None:
            best_sample_id = str(analysis["best_sample"]["sample_id"])
            observation = self._observation(batch, best_sample_id) if scientific_eligible else None
            if scientific_eligible and observation is not None and cie_within_coordinate_tolerance(
                observation["measurement"]["cie"], analysis["target_cie"], self.tolerance
            ):
                provisional_payload = {
                    "source_round": round_number,
                    "sample_id": best_sample_id,
                    "recipe_id": str(observation["identity"]["recipe_id"]),
                    "cie": [float(value) for value in observation["measurement"]["cie"]],
                    "distance": float(observation["evaluation"]["distance"]),
                    "concentrations_mmol_ml": [
                        float(value) for value in observation["recipe"]["concentrations_mmol_ml"]
                    ],
                }
                verification_status = "provisional_hit"
            else:
                verification_status = "not_met"
        elif not scientific_eligible:
            # Synthetic data can exercise the loop but cannot confirm or invalidate a real candidate.
            verification_status = "confirmation_ineligible"
        else:
            if round_number <= provisional.source_round:
                raise ValueError("goal confirmation must come from a later experiment batch")
            expected_recipe = self._recipe_key(provisional.concentrations_mmol_ml)
            repeated = [
                observation
                for observation in batch["observations"]
                if self._recipe_key(observation["recipe"]["concentrations_mmol_ml"])
                == expected_recipe
            ]
            if not repeated:
                raise ValueError("goal confirmation batch contains no exact repeat of the provisional recipe")
            confirmation_samples = [
                {
                    "sample_id": str(observation["identity"]["sample_id"]),
                    "recipe_id": str(observation["identity"]["recipe_id"]),
                    "cie": [float(value) for value in observation["measurement"]["cie"]],
                    "distance": float(observation["evaluation"]["distance"]),
                    "within_tolerance": cie_within_coordinate_tolerance(
                        observation["measurement"]["cie"], analysis["target_cie"], self.tolerance
                    ),
                }
                for observation in repeated
            ]
            verification_status = (
                "confirmed"
                if any(sample["within_tolerance"] for sample in confirmation_samples)
                else "confirmation_failed"
            )

        met = verification_status == "confirmed"
        artifact = self.store.write_artifact_json(
            "artifacts/reports/goal_check.json",
            {
                "round": round_number,
                "met": met,
                "verification_status": verification_status,
                "distance": distance,
                "tolerance": self.tolerance,
                "scientific_eligible": scientific_eligible,
                "best_sample": analysis["best_sample"],
                "provisional_candidate": provisional_payload,
                "confirmation_samples": confirmation_samples,
            },
        )
        summaries = {
            "provisional_hit": "CIE target provisionally reached; an independent exact repeat is required",
            "confirmed": "CIE target independently confirmed",
            "confirmation_failed": "CIE target confirmation failed",
            "confirmation_ineligible": "CIE target confirmation remains pending because this batch is not scientific evidence",
            "not_met": "CIE target not met",
        }
        return ToolResult(
            status="success",
            summary=f"{summaries[verification_status]}; current best distance is {distance:.6f}.",
            data={
                "met": met,
                "verification_status": verification_status,
                "distance": distance,
                "round": round_number,
                "provisional_candidate": provisional_payload,
                "confirmation_samples": confirmation_samples,
            },
            artifacts=[artifact],
        )

    def _current_round_batch(self, arguments: dict[str, Any], round_number: int) -> dict[str, Any]:
        """Load the canonical current batch so repeat identity and origin remain auditable."""
        dataset_artifact = arguments.get("research_dataset_artifact")
        if not dataset_artifact:
            raise ValueError("goal check requires a research dataset artifact")
        dataset_path = self.store.artifact_path(str(dataset_artifact))
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        batch = next((item for item in dataset["batches"] if int(item["round"]) == round_number), None)
        if batch is None:
            raise ValueError("research dataset has no record for the analyzed round")
        return batch

    @staticmethod
    def _observation(batch: dict[str, Any], sample_id: str) -> dict[str, Any]:
        """Resolve one analysis winner back to its recipe-bearing dataset observation."""
        observation = next(
            (item for item in batch["observations"] if item["identity"]["sample_id"] == sample_id),
            None,
        )
        if observation is None:
            raise ValueError("analysis best sample is missing from the current research batch")
        return observation

    @staticmethod
    def _recipe_key(values: Any) -> tuple[float, ...]:
        """Normalize five floating-point concentrations for an exact recipe comparison."""
        return tuple(round(float(value), 12) for value in values)
