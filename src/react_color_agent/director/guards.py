"""Deterministic precondition checks for Director-selected tool calls."""

from __future__ import annotations

from ..models import Action, Phase, TaskState


def can_execute(action: Action, state: TaskState) -> str | None:
    """Reject invalid tool preconditions while leaving scientific strategy to the Agent."""
    if state.phase in {Phase.FINISHED, Phase.STOPPED}:
        return "the task has already reached a terminal phase"
    if state.phase is Phase.AWAITING_HUMAN_REVIEW:
        return "the task is awaiting an explicit human unreachable-application decision"
    if action.name == "query_pubchem" and not str(action.arguments.get("name", "")).strip():
        return "PubChem requires a non-empty English chemical name"
    if action.name == "read_skill" and not str(action.arguments.get("name", "")).strip():
        return "read_skill requires a registered Skill name"
    if action.name == "save_material_evidence" and not any(
        item.action == "query_pubchem" and item.status == "success" for item in state.history
    ):
        return "material evidence requires at least one successful PubChem query"
    if action.name == "design_initial_batch" and not state.all_materials_confirmed:
        return "all materials require confirmed PubChem identity"
    if action.name == "design_initial_batch" and not {"experiment-plan-xlsx", "measurement-data-return"}.issubset(state.active_skills):
        return "CPWL design requires experiment-plan-xlsx and measurement-data-return Skills to be activated first"
    if action.name == "design_initial_batch" and state.round > 0:
        return "design_initial_batch only creates B1; use the controlled follow-up candidate workflow"
    if action.name == "ingest_spectra":
        if state.phase is not Phase.WAITING_FOR_DATA or not state.artifacts.experiment_plan:
            return "spectrum ingestion requires a waiting experiment plan"
        # The runtime injects the user-returned directory after this Guard runs.
        if not state.pending_measurement_path and not str(action.arguments.get("data_path", "")).strip():
            return "spectrum ingestion requires a pending returned-data directory"
    if action.name == "calculate_cie":
        if state.phase is not Phase.ANALYZING or not state.artifacts.spectra_manifest:
            return "CIE calculation requires a qualified spectra manifest"
    if action.name == "analyze_results":
        if state.phase is not Phase.ANALYZING or not state.artifacts.measurement_result:
            return "analysis requires accepted measurements"
    if action.name == "update_research_dataset":
        if state.phase is not Phase.ANALYZING:
            return "research dataset update requires an analyzing task"
        if not all([
            state.artifacts.experiment_plan,
            state.artifacts.spectra_manifest,
            state.artifacts.measurement_result,
            state.artifacts.analysis_result,
        ]):
            return "research dataset update requires plan, spectra, CIE, and analysis artifacts"
    if action.name in {"query_research_index", "get_experiment_record", "get_spectrum_data"}:
        if not state.artifacts.research_dataset or not state.artifacts.research_index:
            return "research data retrieval requires an updated research dataset and index"
    if action.name == "get_research_briefing" and not state.artifacts.research_dataset:
        return "research briefing requires an updated research dataset"
    if action.name == "get_complete_batch_history" and not state.artifacts.research_dataset:
        return "complete batch history requires an updated research dataset"
    if action.name == "get_analysis_record" and not state.artifacts.research_dataset:
        return "analysis record lookup requires an updated research dataset"
    if action.name == "extract_spectral_features":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset or not state.artifacts.research_index:
            return "spectral feature diagnostics require a designing task with an updated research dataset and index"
    if action.name == "review_design_outcomes":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "design outcome review requires a designing task with an updated research dataset"
    if action.name in {"diagnose_dataset", "screen_composition_effects"}:
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "first-layer research analysis requires a missed target and an updated research dataset"
    if action.name == "compile_research_analysis":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "research analysis compilation requires a missed target and an updated research dataset"
        if not state.artifacts.dataset_diagnosis or not state.artifacts.composition_effects:
            return "research analysis compilation requires both diagnosis and composition screening artifacts"
        if not str(action.arguments.get("research_question", "")).strip():
            return "research analysis compilation requires one non-empty research question"
    if action.name == "fit_local_response_model":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "local model fitting requires a missed target and an updated research dataset"
        if not state.artifacts.dataset_diagnosis or not state.artifacts.research_analysis:
            return "local model fitting requires completed diagnosis and first-layer research analysis"
    if action.name == "compare_models":
        if state.phase is not Phase.DESIGNING or len(state.artifacts.response_models) < 2:
            return "model comparison requires at least two completed local model artifacts"
    if action.name == "write_design_decision":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "design decision requires a missed target and an updated research dataset"
        if "research-iteration" not in state.active_skills:
            return "design decision requires the research-iteration Skill to be activated first"
        if action.arguments.get("selected_method") != "none":
            if not state.artifacts.research_analysis or not state.artifacts.model_comparison:
                return "a selected response model requires completed analysis and model comparison"
        if state.provisional_goal_candidate is not None:
            if action.arguments.get("strategy") != "repeat_validation":
                return "a provisional target hit requires a repeat_validation design decision"
            if action.arguments.get("selected_method") != "none":
                return "target confirmation uses an exact repeat, not a response model"
            facts = {str(value) for value in action.arguments.get("facts_used_sample_ids", [])}
            if state.provisional_goal_candidate.sample_id not in facts:
                return "target confirmation must cite the provisional target sample"
    if action.name == "generate_predicted_candidates":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "candidate generation requires a missed target and an updated research dataset"
        if not state.artifacts.design_decision or not state.artifacts.model_comparison:
            return "candidate generation requires an Agent design decision and model comparison"
        if "research-iteration" not in state.active_skills:
            return "candidate generation requires the research-iteration Skill to be activated first"
    if action.name == "design_followup_batch":
        if state.phase is not Phase.DESIGNING or not state.artifacts.predicted_candidates:
            return "follow-up CPWL design requires a generated predicted-candidates artifact"
        if state.round >= state.max_rounds:
            return "maximum experiment rounds reached"
        if not {"experiment-plan-xlsx", "measurement-data-return"}.issubset(state.active_skills):
            return "follow-up CPWL design requires experiment-plan-xlsx and measurement-data-return Skills"
        if state.provisional_goal_candidate is not None:
            return "a provisional target hit requires an exact repeat_validation batch"
    if action.name == "design_exploratory_followup_batch":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset or not state.artifacts.design_decision:
            return "exploratory CPWL design requires a missed target, research dataset, and Agent design decision"
        if state.round >= state.max_rounds:
            return "maximum experiment rounds reached"
        if not {"experiment-plan-xlsx", "measurement-data-return", "research-iteration"}.issubset(state.active_skills):
            return "exploratory CPWL design requires research-iteration, experiment-plan-xlsx, and measurement-data-return Skills"
        if (
            state.provisional_goal_candidate is not None
            and str(action.arguments.get("reference_sample_id", "")).strip()
            != state.provisional_goal_candidate.sample_id
        ):
            return "target confirmation must repeat the provisional target sample"
    if action.name == "propose_followup_batch":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "follow-up draft requires a designing task with measured research data"
        if state.round >= state.max_rounds:
            return "maximum experiment rounds reached"
        if state.provisional_goal_candidate is not None:
            return "a provisional target hit uses the exact repeat_validation path"
        if state.artifacts.batch_draft:
            return "the current design stage already has a saved Director draft"
        if not {"experiment-plan-xlsx", "measurement-data-return", "research-iteration"}.issubset(state.active_skills):
            return "follow-up draft requires research-iteration and both CPWL plan Skills"
    if action.name == "finalize_followup_batch":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "final follow-up submission requires a designing task with measured research data"
        if not all([state.artifacts.batch_draft, state.artifacts.batch_draft_evaluation, state.artifacts.critic_review]):
            return "final follow-up submission requires the saved draft, evaluation, and Scientific Critic review"
        if state.provisional_goal_candidate is not None:
            return "a provisional target hit uses the exact repeat_validation path"
    if action.name == "propose_unreachable_request":
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "an unreachable draft requires a designing task with measured research data"
        if state.provisional_goal_candidate is not None:
            return "a provisional target hit must be independently confirmed, not declared unreachable"
        if state.artifacts.batch_draft:
            return "finish the active follow-up draft workflow before proposing unreachable status"
        if state.artifacts.human_unreachable_review:
            return "a rejected application requires new measured data before another application"
        if state.artifacts.unreachable_draft or state.artifacts.unreachable_decision:
            return "the current dataset already has an active unreachable application workflow"
        if "research-iteration" not in state.active_skills:
            return "unreachable application requires the research-iteration Skill to be activated first"
    if action.name in {
        "continue_after_unreachable_review",
        "submit_unreachable_application",
    }:
        if state.phase is not Phase.DESIGNING or not state.artifacts.research_dataset:
            return "an unreachable review decision requires a designing task with measured data"
        if not all(
            [
                state.artifacts.unreachable_draft,
                state.artifacts.unreachable_evaluation,
                state.artifacts.unreachable_critic_review,
            ]
        ):
            return "responding to unreachable review requires the draft, evaluation, and Critic review"
        if state.artifacts.unreachable_decision:
            return "the active unreachable Critic review already has a Director decision"
    if action.name == "check_goal":
        if state.phase is not Phase.ANALYZING or not state.artifacts.analysis_result or not state.artifacts.research_dataset:
            return "goal checking requires analyzed measurements recorded in the research dataset"
    if action.name == "search_crossref":
        if not action.arguments.get("identity_confirmed"):
            return "Crossref requires confirmed PubChem identity"
        if not action.arguments.get("information_insufficient"):
            return "Crossref requires insufficient PubChem information"
    return None
