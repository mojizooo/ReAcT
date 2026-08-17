"""Task-local argument assembly and compact historical-context injection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..models import Action, Phase, TaskState
from ..storage import TaskStore


def enrich_arguments(action: Action, state: TaskState) -> dict[str, Any]:
    """Supply controlled artifact references instead of accepting model-authored paths."""
    arguments = dict(action.arguments)
    if action.name == "design_initial_batch":
        arguments.setdefault("target", list(state.target))
        arguments.setdefault("materials", state.materials)
        arguments.setdefault("evidence_artifact", state.artifacts.material_evidence)
        arguments.setdefault("round", state.round)
        arguments.setdefault("applied_skills", state.active_skills)
    elif action.name == "ingest_spectra":
        arguments.setdefault("experiment_plan", state.artifacts.experiment_plan)
        arguments.setdefault("round", state.round)
        arguments.setdefault("data_path", state.pending_measurement_path)
        arguments.setdefault("data_origin", state.pending_data_origin or "measured")
    elif action.name == "calculate_cie":
        arguments.setdefault("spectra_artifact", state.artifacts.spectra_manifest)
        arguments.setdefault("target", list(state.target))
    elif action.name == "analyze_results":
        arguments.setdefault("measurement_artifact", state.artifacts.measurement_result)
        arguments.setdefault("target", list(state.target))
    elif action.name == "update_research_dataset":
        arguments.setdefault("task_id", state.task_id)
        arguments.setdefault("target", list(state.target))
        arguments.setdefault("max_rounds", state.max_rounds)
        arguments.setdefault("experiment_plan", state.artifacts.experiment_plan)
        arguments.setdefault("spectra_artifact", state.artifacts.spectra_manifest)
        arguments.setdefault("measurement_artifact", state.artifacts.measurement_result)
        arguments.setdefault("analysis_artifact", state.artifacts.analysis_result)
    elif action.name == "check_goal":
        arguments.setdefault("analysis_artifact", state.artifacts.analysis_result)
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
    elif action.name == "extract_spectral_features":
        # Spectral evidence must always resolve through the current task dataset, never a model path.
        arguments["research_dataset_artifact"] = state.artifacts.research_dataset
        arguments["round"] = state.round
    elif action.name == "review_design_outcomes":
        # Reviews may only inspect the just-completed batch of the active task.
        arguments["research_dataset_artifact"] = state.artifacts.research_dataset
        arguments["round"] = state.round

    if action.name == "query_research_index":
        arguments.setdefault("research_index_artifact", state.artifacts.research_index)
    elif action.name in {
        "get_experiment_record",
        "get_spectrum_data",
        "get_research_briefing",
        "get_complete_batch_history",
        "get_analysis_record",
    }:
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        if action.name == "get_research_briefing":
            arguments.setdefault("target", list(state.target))

    if action.name in {"diagnose_dataset", "screen_composition_effects"}:
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("target", list(state.target))
        arguments.setdefault("round", state.round)
    elif action.name == "compile_research_analysis":
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("diagnosis_artifact", state.artifacts.dataset_diagnosis)
        arguments.setdefault("composition_effects_artifact", state.artifacts.composition_effects)
    elif action.name == "fit_local_response_model":
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("diagnosis_artifact", state.artifacts.dataset_diagnosis)
        arguments.setdefault("round", state.round)
    elif action.name == "compare_models":
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("model_artifacts", state.artifacts.response_models)
    elif action.name == "write_design_decision":
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        if state.artifacts.research_analysis:
            arguments.setdefault("research_analysis_artifact", state.artifacts.research_analysis)
        if state.artifacts.model_comparison:
            arguments.setdefault("model_comparison_artifact", state.artifacts.model_comparison)
    elif action.name == "generate_predicted_candidates":
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("design_decision_artifact", state.artifacts.design_decision)
        arguments.setdefault("model_comparison_artifact", state.artifacts.model_comparison)
        arguments.setdefault("round", state.round)
    elif action.name == "design_followup_batch":
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("predicted_candidates_artifact", state.artifacts.predicted_candidates)
        arguments.setdefault("round", state.round)
        arguments.setdefault("target", list(state.target))
        arguments.setdefault("applied_skills", state.active_skills)
    elif action.name == "design_exploratory_followup_batch":
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("design_decision_artifact", state.artifacts.design_decision)
        arguments.setdefault("round", state.round)
        arguments.setdefault("target", list(state.target))
        arguments.setdefault("applied_skills", state.active_skills)
    elif action.name == "propose_followup_batch":
        # Drafts may only evaluate the active task's canonical measured dataset.
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("round", state.round)
        arguments.setdefault("model_artifacts", state.artifacts.response_models)
    elif action.name == "finalize_followup_batch":
        # Final submission is tied to the exact draft and one runtime-created Critic review.
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("batch_draft_artifact", state.artifacts.batch_draft)
        arguments.setdefault("critic_review_artifact", state.artifacts.critic_review)
        arguments.setdefault("round", state.round)
        arguments.setdefault("applied_skills", state.active_skills)
    elif action.name == "propose_unreachable_request":
        # The model supplies only the scientific claim; runtime owns task-local provenance.
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("task_id", state.task_id)
        arguments.setdefault("target", list(state.target))
        arguments.setdefault("round", state.round)
        arguments.setdefault("max_rounds", state.max_rounds)
    elif action.name in {
        "continue_after_unreachable_review",
        "submit_unreachable_application",
    }:
        # Review decisions always bind to the active draft packet, never model-authored paths.
        arguments.setdefault("research_dataset_artifact", state.artifacts.research_dataset)
        arguments.setdefault("unreachable_draft_artifact", state.artifacts.unreachable_draft)
        arguments.setdefault(
            "unreachable_evaluation_artifact", state.artifacts.unreachable_evaluation
        )
        arguments.setdefault(
            "unreachable_critic_review_artifact",
            state.artifacts.unreachable_critic_review,
        )
        arguments.setdefault("task_id", state.task_id)
        arguments.setdefault("round", state.round)
    return arguments


def refresh_research_briefing(
    store: TaskStore,
    tools: dict[str, Any],
    state: TaskState,
    trace: Callable[[str], None],
) -> TaskState:
    """Inject a compact facts-first briefing once per data-backed design stage."""
    if (
        state.phase is not Phase.DESIGNING
        or not state.artifacts.research_dataset
        or state.injected_research_briefing is not None
    ):
        return state
    tool = tools.get("get_research_briefing")
    if tool is None:
        trace("research briefing skipped: tool is not registered")
        return state
    try:
        result = tool.run(enrich_arguments(Action(name="get_research_briefing"), state))
    except Exception as error:  # Briefing is helpful context, not a task-state dependency.
        trace(f"research briefing failed: {error}")
        return state
    if result.status != "success" or not result.artifacts:
        trace(f"research briefing unavailable: {result.summary}")
        return state
    trace("injected compact research briefing for the current design stage")
    return store.update(
        lambda current: current.model_copy(
            update={
                "artifacts": current.artifacts.model_copy(
                    update={"research_briefing": result.artifacts[0]}
                ),
                "injected_research_briefing": result.data,
            }
        )
    )
