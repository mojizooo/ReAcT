"""Draft, evaluate, and finalize one scientifically reviewed follow-up batch."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from typing import Any

import numpy as np

from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore
from skills.cpwl.experiment_plan_xlsx.cpwl_xlsx import (
    CONCENTRATION_TOLERANCE_MMOL_ML,
    MAX_COMPONENT_CONCENTRATION_MMOL_ML,
    MAX_COMPONENT_VOLUME_ML,
    MAX_TOTAL_CONCENTRATION_MMOL_ML,
    MIN_TOTAL_CONCENTRATION_MMOL_ML,
    PRODUCT_VOLUME_ML,
    STOCK_CONCENTRATION_MMOL_ML,
    VOLUME_TOLERANCE_ML,
    CpwlDiscrimination,
    CpwlMaterial,
    build_followup_cpwl_batch,
    write_cpwl_xlsx_artifacts,
)
from tools.candidate_tools import _coverage_classification, _predict_model

MAX_FOLLOWUP_RECIPES = 12
STRATEGIES = {
    "local_refinement",
    "model_guided",
    "component_adjustment",
    "region_switch",
    "concentration_scan",
    "multi_component_search",
    "diagnostic",
}
STRATEGY_ALIASES = {
    # Keep one narrow legacy alias so an older prompt label does not waste an API turn.
    "coverage": "multi_component_search",
}


def _load_json(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Read one controlled run-local JSON artifact."""
    try:
        return json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load JSON artifact {artifact}") from error


def _canonical_sha256(payload: Any) -> str:
    """Hash a JSON-compatible object using the artifact lineage convention."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measured_observations(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index real observations only, excluding synthetic dry runs from scientific evidence."""
    return {
        str(observation["identity"]["sample_id"]): observation
        for batch in dataset.get("batches", [])
        if bool(batch.get("origin", {}).get("scientific_eligible"))
        for observation in batch.get("observations", [])
    }


def _materials(dataset: dict[str, Any]) -> list[CpwlMaterial]:
    """Convert canonical dataset material slots to the CPWL writer contract."""
    return [
        CpwlMaterial(
            slot=str(material["slot"]),
            material_key=str(material["pubchem_cid"]),
            name=str(material["name"]),
            molecular_weight_g_mol=float(material["molecular_weight_g_mol"]),
        )
        for material in dataset["task"]["materials"]
    ]


def _validate_text(value: Any, field: str) -> str:
    """Require a concise non-empty scientific text field."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _validate_options(value: Any, field: str, *, required: bool = True) -> list[dict[str, str]]:
    """Normalize named alternatives without imposing a particular scientific path taxonomy."""
    if value is None and not required:
        return []
    if not isinstance(value, list) or (required and not value) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be a list of option/reason objects")
    return [
        {
            "option": _validate_text(item.get("option"), f"{field}.option"),
            "reason": _validate_text(item.get("reason"), f"{field}.reason"),
        }
        for item in value
    ]


def _validate_volume_contract(recipe_id: str, concentrations: list[float]) -> dict[str, Any]:
    """Validate one recipe and return calculated volumes for actionable diagnostics."""
    stock_volumes = [
        concentration * PRODUCT_VOLUME_ML / STOCK_CONCENTRATION_MMOL_ML
        for concentration in concentrations
    ]
    total_concentration = sum(concentrations)
    solvent_volume = PRODUCT_VOLUME_ML - sum(stock_volumes)
    violations: list[str] = []
    repairs: list[str] = []

    for index, volume in enumerate(stock_volumes):
        if volume > MAX_COMPONENT_VOLUME_ML + VOLUME_TOLERANCE_ML:
            slot = chr(ord("A") + index)
            concentration = concentrations[index]
            violations.append(
                f"slot {slot} stock volume={volume:.6g} ml > {MAX_COMPONENT_VOLUME_ML:g} ml; "
                f"slot {slot} concentration={concentration:.6g} mmol/ml > maximum "
                f"{MAX_COMPONENT_CONCENTRATION_MMOL_ML:.6g} mmol/ml"
            )
            repairs.append(
                f"reduce slot {slot} concentration to <= "
                f"{MAX_COMPONENT_CONCENTRATION_MMOL_ML:.6g} mmol/ml"
            )

    if solvent_volume > MAX_COMPONENT_VOLUME_ML + VOLUME_TOLERANCE_ML:
        violations.append(
            f"solvent_volume_ml={solvent_volume:.6g} ml > {MAX_COMPONENT_VOLUME_ML:g} ml"
        )
        repairs.append(
            "increase the total final concentration to at least "
            f"{MIN_TOTAL_CONCENTRATION_MMOL_ML:.6g} mmol/ml"
        )
    elif solvent_volume < -VOLUME_TOLERANCE_ML:
        violations.append(f"solvent_volume_ml={solvent_volume:.6g} ml < 0 ml")
        repairs.append(
            "reduce the total final concentration to at most "
            f"{MAX_TOTAL_CONCENTRATION_MMOL_ML:.6g} mmol/ml"
        )

    if violations:
        raise ValueError(
            f"recipe {recipe_id} violates the fixed 8 ml volume contract: "
            f"total_concentration_mmol_ml={total_concentration:.6g}; "
            f"{'; '.join(violations)}. Required correction: {'; '.join(dict.fromkeys(repairs))}."
        )

    return {
        "total_concentration_mmol_ml": total_concentration,
        "stock_volumes_ml": stock_volumes,
        "solvent_volume_ml": solvent_volume,
    }


def _validate_recipes(
    value: Any,
    measured_ids: set[str],
    batch_number: int,
) -> list[dict[str, Any]]:
    """Validate complete Director-authored recipes and their measured evidence references."""
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_FOLLOWUP_RECIPES:
        raise ValueError("ordinary follow-up drafts require two to twelve recipes")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("every follow-up recipe must be an object")
    recipes: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        raw = item.get("concentrations_mmol_ml")
        if not isinstance(raw, list) or len(raw) != 5:
            raise ValueError("each follow-up recipe requires exactly five A-E concentrations")
        try:
            concentrations = [float(component) for component in raw]
        except (TypeError, ValueError) as error:
            raise ValueError("recipe concentrations must be numeric") from error
        if any(
            not math.isfinite(component)
            or component < 0
            or component
            > MAX_COMPONENT_CONCENTRATION_MMOL_ML + CONCENTRATION_TOLERANCE_MMOL_ML
            for component in concentrations
        ):
            raise ValueError(
                "recipe concentrations must be finite values within "
                f"0..{MAX_COMPONENT_CONCENTRATION_MMOL_ML:.6g} mmol/ml"
            )
        if sum(concentrations) <= 0:
            raise ValueError("each recipe requires at least one positive component")
        _validate_volume_contract(f"B{batch_number}-N{index}", concentrations)
        role = str(item.get("design_role", "")).strip()
        if role not in {"primary", "competing"}:
            raise ValueError("each recipe design_role must be primary or competing")
        try:
            discrimination = CpwlDiscrimination.model_validate(item.get("discrimination"))
        except ValueError as error:
            raise ValueError(f"invalid recipe discrimination: {error}") from error
        references = set(discrimination.reference_sample_ids)
        if not references:
            raise ValueError("each follow-up recipe must cite at least one measured reference sample")
        if references - measured_ids:
            raise ValueError("recipe discrimination cites a sample absent from measured history")
        recipes.append(
            {
                "draft_recipe_id": f"R{index}",
                "concentrations_mmol_ml": concentrations,
                "purpose": _validate_text(item.get("purpose"), "recipe purpose"),
                "design_role": role,
                "discrimination": discrimination.model_dump(mode="json"),
                "source_candidate_id": str(item.get("source_candidate_id") or "").strip() or None,
            }
        )
    roles = {recipe["design_role"] for recipe in recipes}
    if roles != {"primary", "competing"}:
        raise ValueError("an ordinary follow-up batch requires both primary and competing recipes")
    return recipes


def _validated_draft(arguments: dict[str, Any], dataset: dict[str, Any], round_number: int) -> dict[str, Any]:
    """Build the single canonical draft object before any files are generated."""
    observations = _measured_observations(dataset)
    facts = arguments.get("facts_used_sample_ids")
    if not isinstance(facts, list) or not facts or not all(isinstance(item, str) for item in facts):
        raise ValueError("facts_used_sample_ids must contain measured sample IDs")
    fact_ids = list(dict.fromkeys(item.strip() for item in facts if item.strip()))
    if not fact_ids or set(fact_ids) - set(observations):
        raise ValueError("facts_used_sample_ids contains missing or non-measured samples")
    original_strategy = str(arguments.get("strategy", "")).strip()
    strategy = STRATEGY_ALIASES.get(original_strategy, original_strategy)
    if strategy not in STRATEGIES:
        raise ValueError("strategy is not a supported follow-up design label")
    recipes = _validate_recipes(arguments.get("recipes"), set(observations), round_number + 1)
    # Recipe-level references are facts used by definition; merge them once instead of
    # forcing the model to duplicate every sample ID in two argument locations.
    recipe_reference_ids = [
        str(sample_id)
        for recipe in recipes
        for sample_id in recipe["discrimination"]["reference_sample_ids"]
    ]
    fact_ids = list(dict.fromkeys([*fact_ids, *recipe_reference_ids]))
    draft = {
        "schema_version": 1,
        "round": round_number,
        "batch_id": f"B{round_number + 1}",
        "target_cie": list(dataset["task"]["target_cie"]),
        "source_dataset_sha256": _canonical_sha256(dataset),
        "strategy": strategy,
        "research_question": _validate_text(arguments.get("research_question"), "research_question"),
        "facts_used_sample_ids": fact_ids,
        "primary_option": _validate_text(arguments.get("primary_option"), "primary_option"),
        "competing_option": _validate_text(arguments.get("competing_option"), "competing_option"),
        "rejected_options": _validate_options(arguments.get("rejected_options", []), "rejected_options", required=False),
        "allocation_reason": _validate_text(arguments.get("allocation_reason"), "allocation_reason"),
        "recipes": recipes,
        "scientific_status": "DIRECTOR_DRAFT_PENDING_CRITIC",
    }
    if original_strategy != strategy:
        draft["original_strategy_label"] = original_strategy
    # Reuse the CPWL physical validator without writing any plan artifact at draft time.
    build_followup_cpwl_batch(
        run_id=str(dataset["task"]["task_id"]),
        target_cie=tuple(float(value) for value in dataset["task"]["target_cie"]),
        materials=_materials(dataset),
        batch_number=round_number + 1,
        candidate_recipes=[
            (
                tuple(recipe["concentrations_mmol_ml"]),
                recipe["purpose"],
                CpwlDiscrimination.model_validate(recipe["discrimination"]),
            )
            for recipe in recipes
        ],
        applied_skills=[],
        design_context={"strategy": strategy},
    )
    return draft


def _composition_distance(first: list[float], second: list[float], scale: float) -> float:
    """Measure recipe separation in concentration space using one transparent global scale."""
    return float(np.linalg.norm((np.asarray(first) - np.asarray(second)) / scale) / math.sqrt(5))


def _evaluate_draft(
    draft: dict[str, Any],
    dataset: dict[str, Any],
    store: TaskStore,
    model_artifacts: list[str],
) -> dict[str, Any]:
    """Evaluate actual proposed formulations without selecting an alternative for the Director."""
    observations = _measured_observations(dataset)
    history = list(observations.items())
    all_values = [
        float(value)
        for _sample_id, observation in history
        for value in observation["recipe"]["concentrations_mmol_ml"]
    ]
    scale = max([MAX_TOTAL_CONCENTRATION_MMOL_ML, *all_values])
    recipe_rows: list[dict[str, Any]] = []
    for recipe in draft["recipes"]:
        values = recipe["concentrations_mmol_ml"]
        neighbors = sorted(
            (
                {
                    "sample_id": sample_id,
                    "composition_distance": _composition_distance(
                        values, observation["recipe"]["concentrations_mmol_ml"], scale
                    ),
                    "cie": observation["measurement"]["cie"],
                    "target_distance": observation["evaluation"]["distance"],
                }
                for sample_id, observation in history
            ),
            key=lambda item: (item["composition_distance"], item["target_distance"]),
        )
        active = [index for index, value in enumerate(values) if value > 0]
        comparable = [
            observation for _sample_id, observation in history
            if [index for index, value in enumerate(observation["recipe"]["concentrations_mmol_ml"]) if value > 0] == active
        ]
        if len(comparable) < 2:
            coverage = "unsupported"
        else:
            matrix = np.asarray(
                [[observation["recipe"]["concentrations_mmol_ml"][index] for index in active] for observation in comparable],
                dtype=float,
            )
            query = np.asarray([values[index] for index in active], dtype=float)
            outside = bool(np.any(query < matrix.min(axis=0) - 1e-12) or np.any(query > matrix.max(axis=0) + 1e-12))
            on_edge = bool(np.any(np.isclose(query, matrix.min(axis=0), atol=1e-12)) or np.any(np.isclose(query, matrix.max(axis=0), atol=1e-12)))
            coverage = "extrapolation" if outside else ("boundary" if on_edge else "axis_aligned_interpolation")
        predictions = _supported_model_predictions(
            values, dataset, observations, store, model_artifacts
        )
        recipe_rows.append(
            {
                "draft_recipe_id": recipe["draft_recipe_id"],
                "design_role": recipe["design_role"],
                "nearest_measured": neighbors[:3],
                "duplicate_of_measured_sample_id": (
                    neighbors[0]["sample_id"] if neighbors and neighbors[0]["composition_distance"] <= 1e-9 else None
                ),
                "coverage": coverage,
                "prediction": (
                    {"status": "supported_local_models", "models": predictions}
                    if predictions
                    else {
                        "status": "unsupported",
                        "reason": "no saved supported local model covers this exact formulation",
                    }
                ),
            }
        )
    pair_rows = []
    for first, second in combinations(draft["recipes"], 2):
        distance = _composition_distance(
            first["concentrations_mmol_ml"], second["concentrations_mmol_ml"], scale
        )
        if distance <= 0.02:
            pair_rows.append(
                {
                    "first": first["draft_recipe_id"],
                    "second": second["draft_recipe_id"],
                    "composition_distance": distance,
                    "classification": "duplicate" if distance <= 1e-9 else "very_close",
                }
            )
    role_counts = {
        role: sum(recipe["design_role"] == role for recipe in draft["recipes"])
        for role in ("primary", "competing")
    }
    unique_active_sets = {
        tuple(index for index, value in enumerate(recipe["concentrations_mmol_ml"]) if value > 0)
        for recipe in draft["recipes"]
    }
    warnings: list[str] = []
    duplicates = [row for row in recipe_rows if row["duplicate_of_measured_sample_id"]]
    if duplicates:
        warnings.append("one or more recipes duplicate an existing measured formulation")
    if pair_rows:
        warnings.append("one or more draft recipes are duplicate or very close in composition space")
    if len(unique_active_sets) > 3:
        warnings.append("the batch spans more than three component sets and may dilute a hit-first budget")
    return {
        "schema_version": 1,
        "round": draft["round"],
        "batch_id": draft["batch_id"],
        "source_dataset_sha256": draft["source_dataset_sha256"],
        "draft_sha256": _canonical_sha256(draft),
        "recipe_count": len(draft["recipes"]),
        "role_counts": role_counts,
        "unique_component_sets": [list(values) for values in sorted(unique_active_sets)],
        "recipe_evaluations": recipe_rows,
        "within_batch_close_pairs": pair_rows,
        "warnings": warnings,
        "interpretation_boundary": (
            "This deterministic report evaluates submitted recipes and measured coverage. "
            "It does not choose the scientific strategy or linearly mix CIE xy values."
        ),
    }


def _supported_model_predictions(
    concentrations: list[float],
    dataset: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    store: TaskStore,
    model_artifacts: list[str],
) -> list[dict[str, Any]]:
    """Apply only already-supported saved models when the full recipe stays in their local scope."""
    material_slots = [str(material["slot"]) for material in dataset["task"]["materials"]]
    dataset_hash = _canonical_sha256(dataset)
    target = np.asarray(dataset["task"]["target_cie"], dtype=float)
    predictions: list[dict[str, Any]] = []
    for artifact in model_artifacts:
        try:
            model = _load_json(store, artifact)
            if (
                model.get("source_dataset_sha256") != dataset_hash
                or model.get("support", {}).get("fit_status") != "supported"
            ):
                continue
            active_slots = [str(slot) for slot in model["active_slots"]]
            if not 1 <= len(active_slots) <= 2:
                continue
            active_indices = [material_slots.index(slot) for slot in active_slots]
            training_ids = [str(value) for value in model["input_sample_ids"]]
            if not training_ids or any(sample_id not in observations for sample_id in training_ids):
                continue
            training = [observations[sample_id] for sample_id in training_ids]
            base = training[0]["recipe"]["concentrations_mmol_ml"]
            inactive_indices = [index for index in range(5) if index not in active_indices]
            if any(
                not math.isclose(concentrations[index], float(base[index]), rel_tol=0, abs_tol=1e-12)
                for index in inactive_indices
            ):
                continue
            features = np.asarray(
                [[item["recipe"]["concentrations_mmol_ml"][index] for index in active_indices] for item in training],
                dtype=float,
            )
            query = np.asarray([concentrations[index] for index in active_indices], dtype=float)
            coverage = _coverage_classification(query, features)
            if coverage != "interpolation":
                continue
            cie = np.asarray([item["measurement"]["cie"] for item in training], dtype=float)
            predicted, details = _predict_model(model, features, cie, training_ids, query)
            predictions.append(
                {
                    "model_artifact": artifact,
                    "method": model["method"],
                    "coverage": coverage,
                    "predicted_cie": [float(predicted[0]), float(predicted[1])],
                    "predicted_target_distance": float(np.linalg.norm(predicted - target)),
                    "details": details,
                }
            )
        except (KeyError, TypeError, ValueError):
            # A stale or incompatible model cannot make the draft evaluation fail.
            continue
    return predictions


class ProposeFollowupBatchTool:
    """Persist one Director draft and its deterministic scientific-budget evaluation."""

    name = "propose_followup_batch"

    def __init__(self, store: TaskStore) -> None:
        """Bind draft artifacts to one recoverable research task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate a complete draft, save it, then evaluate the actual formulations."""
        dataset = _load_json(self.store, str(arguments["research_dataset_artifact"]))
        round_number = int(arguments["round"])
        draft = _validated_draft(arguments, dataset, round_number)
        draft_artifact = self.store.write_artifact_json(
            f"artifacts/experiment_plans/batch_{round_number + 1:03}_draft.json", draft
        )
        evaluation = _evaluate_draft(
            draft,
            dataset,
            self.store,
            [str(value) for value in arguments.get("model_artifacts", [])],
        )
        evaluation["source_draft"] = draft_artifact
        evaluation_artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/batch_draft_evaluation.json", evaluation
        )
        return ToolResult(
            status="success",
            summary=(
                f"Saved and evaluated a {len(draft['recipes'])}-recipe {draft['batch_id']} draft; "
                "the runtime will now request one independent Scientific Critic review."
            ),
            data={"draft": draft, "evaluation": evaluation},
            artifacts=[draft_artifact, evaluation_artifact],
        )


class FinalizeFollowupBatchTool:
    """Generate every final batch artifact from one Director-owned reviewed object."""

    name = "finalize_followup_batch"

    def __init__(self, store: TaskStore) -> None:
        """Bind final decisions and CPWL output to the active task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Require Critic responses, validate final recipes, and atomically derive plan artifacts."""
        dataset = _load_json(self.store, str(arguments["research_dataset_artifact"]))
        draft = _load_json(self.store, str(arguments["batch_draft_artifact"]))
        critic = _load_json(self.store, str(arguments["critic_review_artifact"]))
        round_number = int(arguments["round"])
        if int(draft["round"]) != round_number or draft["source_dataset_sha256"] != _canonical_sha256(dataset):
            raise ValueError("draft does not belong to the current dataset and completed round")
        if critic.get("draft_sha256") != _canonical_sha256(draft):
            raise ValueError("Scientific Critic review does not belong to the current draft")
        responses = arguments.get("critic_responses")
        if not isinstance(responses, list) or not all(isinstance(item, dict) for item in responses):
            raise ValueError("critic_responses must be a list")
        normalized_responses = [
            {
                "finding_id": _validate_text(item.get("finding_id"), "critic response finding_id"),
                "disposition": str(item.get("disposition", "")).strip(),
                "response": _validate_text(item.get("response"), "critic response"),
            }
            for item in responses
        ]
        if any(item["disposition"] not in {"accepted", "partially_accepted", "rejected"} for item in normalized_responses):
            raise ValueError("critic response disposition must be accepted, partially_accepted, or rejected")
        finding_ids = {str(item["finding_id"]) for item in critic.get("findings", [])}
        response_ids = {item["finding_id"] for item in normalized_responses}
        if finding_ids != response_ids or len(response_ids) != len(normalized_responses):
            raise ValueError("Director must respond exactly once to every Scientific Critic finding")

        # The final object repeats every choice so edits are explicit and never mutate the saved draft.
        final_arguments = {
            **arguments,
            "facts_used_sample_ids": arguments.get("facts_used_sample_ids", draft["facts_used_sample_ids"]),
        }
        final = _validated_draft(final_arguments, dataset, round_number)
        final.update(
            {
                "source_draft": str(arguments["batch_draft_artifact"]),
                "source_critic_review": str(arguments["critic_review_artifact"]),
                "critic_verdict": critic["verdict"],
                "critic_responses": normalized_responses,
                "final_reason": _validate_text(arguments.get("final_reason"), "final_reason"),
                "scientific_status": "FINAL_RECIPES_PENDING_MEASUREMENT",
            }
        )
        batch_number = round_number + 1
        plan = build_followup_cpwl_batch(
            run_id=self.store.load().task_id,
            target_cie=tuple(float(value) for value in dataset["task"]["target_cie"]),
            materials=_materials(dataset),
            batch_number=batch_number,
            candidate_recipes=[
                (
                    tuple(recipe["concentrations_mmol_ml"]),
                    f"{recipe['design_role']}: {recipe['purpose']}",
                    CpwlDiscrimination.model_validate(recipe["discrimination"]),
                )
                for recipe in final["recipes"]
            ],
            applied_skills=list(arguments.get("applied_skills", [])),
            design_context={
                "strategy": final["strategy"],
                "research_question": final["research_question"],
                "primary_option": final["primary_option"],
                "competing_option": final["competing_option"],
                "allocation_reason": final["allocation_reason"],
                "source_draft": final["source_draft"],
                "source_critic_review": final["source_critic_review"],
                "critic_responses": normalized_responses,
                "final_reason": final["final_reason"],
            },
        )
        plan_artifacts = write_cpwl_xlsx_artifacts(self.store, plan)
        # Preserve the established CPWL top-level schema for later spectrum ingestion while
        # embedding the full reviewed Director object in the same authoritative file.
        final_design = {
            **plan.model_dump(mode="json"),
            "reviewed_design": final,
        }
        final_design_artifact = self.store.write_artifact_json(
            f"artifacts/experiment_plans/batch_{batch_number:03}_design.json", final_design
        )
        decision = {
            "schema_version": 2,
            "decision_kind": "reviewed_batch",
            "round": round_number,
            "next_batch": f"B{batch_number}",
            "source_dataset_sha256": final["source_dataset_sha256"],
            "source_final_design": final_design_artifact,
            "strategy": final["strategy"],
            "research_question": final["research_question"],
            "facts_used_sample_ids": final["facts_used_sample_ids"],
            "primary_option": final["primary_option"],
            "competing_option": final["competing_option"],
            "allocation_reason": final["allocation_reason"],
            "critic_verdict": final["critic_verdict"],
            "critic_responses": normalized_responses,
            "final_reason": final["final_reason"],
        }
        decision_artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/design_decision.json", decision
        )
        # Replace the original plan JSON reference in the returned list with the enriched final design.
        plan_artifacts = [
            final_design_artifact if item.endswith(f"batch_{batch_number:03}_design.json") else item
            for item in plan_artifacts
        ]
        return ToolResult(
            status="success",
            summary=(
                f"Finalized reviewed {final['batch_id']} with {len(final['recipes'])} recipes and created "
                f"the CPWL workbook plus return directory {plan_artifacts[-1]}."
            ),
            data={
                "round": batch_number,
                "recipe_count": len(final["recipes"]),
                "measurement_return_directory": plan_artifacts[-1],
            },
            artifacts=[*plan_artifacts, decision_artifact],
        )
