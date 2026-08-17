"""Durable Director state transitions for completed or rejected tool calls."""

from __future__ import annotations

import hashlib

from ..models import (
    Action,
    LastAction,
    Observation,
    Phase,
    ProvisionalGoalCandidate,
    TaskState,
    ToolResult,
)
from ..storage import TaskStore


def mark_running(store: TaskStore, action: Action) -> None:
    """Persist an in-progress marker so an interrupted call remains visible on restart."""
    store.update(
        lambda state: state.model_copy(
            update={"last_action": LastAction(name=action.name, status="running")}
        )
    )


def persist_pubchem_result(store: TaskStore, action: Action, result: ToolResult) -> None:
    """Archive successful raw PubChem identity payloads before later evidence assembly."""
    if action.name != "query_pubchem" or result.status != "success":
        return
    identity = result.data
    digest = hashlib.sha256(
        f"{identity.get('material')}::{identity.get('query_name')}".encode("utf-8")
    ).hexdigest()[:16]
    result.artifacts.append(
        store.write_artifact_json(f"artifacts/evidence/queries/{digest}.json", identity)
    )


def apply_result(store: TaskStore, action: Action, result: ToolResult) -> TaskState:
    """Append an observation and apply the bounded state fields owned by one tool."""
    observation_status = "success" if result.status == "success" else "failed"

    def mutate(state: TaskState) -> TaskState:
        artifacts = state.artifacts
        active_skills = state.active_skills
        confirmed_materials = state.confirmed_materials
        phase = state.phase
        completed_round = state.round
        pending_measurement_path = state.pending_measurement_path
        pending_data_origin = state.pending_data_origin
        injected_research_briefing = state.injected_research_briefing
        injected_complete_batch_history = state.injected_complete_batch_history
        injected_candidate_briefing = state.injected_candidate_briefing
        injected_critic_review = state.injected_critic_review
        injected_unreachable_critic_review = state.injected_unreachable_critic_review
        provisional_goal_candidate = state.provisional_goal_candidate
        result_summary = state.result

        if action.name == "read_skill" and result.status == "success":
            active_skills = list(dict.fromkeys([*active_skills, result.data["skill_name"]]))
        elif action.name == "save_material_evidence" and result.status == "success":
            artifacts = artifacts.model_copy(update={"material_evidence": result.artifacts[0]})
            confirmed_materials = list(dict.fromkeys(result.data["confirmed_materials"]))
            phase = Phase.DESIGNING
        elif action.name in {
            "design_initial_batch",
            "design_followup_batch",
            "design_exploratory_followup_batch",
        } and result.status == "success":
            plan_artifact = next((item for item in result.artifacts if item.endswith(".xlsx")), result.artifacts[0])
            updates: dict[str, object] = {"experiment_plan": plan_artifact}
            if action.name == "design_exploratory_followup_batch":
                updates["exploratory_selection"] = result.artifacts[-1]
            artifacts = artifacts.model_copy(update=updates)
            phase = Phase.WAITING_FOR_DATA
            injected_candidate_briefing = None
        elif action.name == "propose_followup_batch" and result.status == "success":
            artifacts = artifacts.model_copy(
                update={
                    "batch_draft": result.artifacts[0],
                    "batch_draft_evaluation": result.artifacts[1],
                    "critic_review": None,
                }
            )
            injected_critic_review = None
        elif action.name == "scientific_critic" and result.status == "success":
            artifacts = artifacts.model_copy(update={"critic_review": result.artifacts[0]})
            injected_critic_review = result.data
        elif action.name == "propose_unreachable_request" and result.status == "success":
            artifacts = artifacts.model_copy(
                update={
                    "unreachable_draft": result.artifacts[0],
                    "unreachable_evaluation": result.artifacts[1],
                    "unreachable_critic_review": None,
                    "unreachable_decision": None,
                }
            )
            injected_unreachable_critic_review = None
        elif action.name == "unreachable_scientific_critic" and result.status == "success":
            artifacts = artifacts.model_copy(update={"unreachable_critic_review": result.artifacts[0]})
            injected_unreachable_critic_review = result.data
        elif action.name == "continue_after_unreachable_review" and result.status == "success":
            artifacts = artifacts.model_copy(update={"unreachable_decision": result.artifacts[0]})
            # The Director is allowed to design a normal next batch from this same dataset.
            artifacts = artifacts.model_copy(
                update={
                    "unreachable_draft": None,
                    "unreachable_evaluation": None,
                    "unreachable_critic_review": None,
                }
            )
            injected_unreachable_critic_review = None
        elif action.name == "submit_unreachable_application" and result.status == "success":
            artifacts = artifacts.model_copy(update={"unreachable_decision": result.artifacts[0]})
            phase = Phase.AWAITING_HUMAN_REVIEW
        elif action.name == "finalize_followup_batch" and result.status == "success":
            plan_artifact = next(item for item in result.artifacts if item.endswith(".xlsx"))
            decision_artifact = next(
                item for item in result.artifacts if item.endswith("/design_decision.json")
            )
            artifacts = artifacts.model_copy(
                update={
                    "experiment_plan": plan_artifact,
                    "design_decision": decision_artifact,
                }
            )
            phase = Phase.WAITING_FOR_DATA
            injected_candidate_briefing = None
        elif action.name == "ingest_spectra" and result.status == "success":
            artifacts = artifacts.model_copy(update={"spectra_manifest": result.artifacts[0]})
            phase = Phase.ANALYZING
            completed_round = int(result.data["round"])
            pending_measurement_path = None
            pending_data_origin = None
        elif action.name == "calculate_cie" and result.status == "success":
            artifacts = artifacts.model_copy(update={"measurement_result": result.artifacts[0]})
        elif action.name == "analyze_results" and result.status == "success":
            artifacts = artifacts.model_copy(update={"analysis_result": result.artifacts[0]})
        elif action.name == "update_research_dataset" and result.status == "success":
            artifacts = artifacts.model_copy(
                update={
                    "research_dataset": result.artifacts[0],
                    "research_index": result.artifacts[1],
                    # Each updated dataset begins a new analysis and design-decision stage.
                    "dataset_diagnosis": None,
                    "composition_effects": None,
                    "research_analysis": None,
                    "response_models": [],
                    "model_comparison": None,
                    "design_decision": None,
                    "predicted_candidates": None,
                    "exploratory_selection": None,
                    "batch_draft": None,
                    "batch_draft_evaluation": None,
                    "critic_review": None,
                    "unreachable_draft": None,
                    "unreachable_evaluation": None,
                    "unreachable_critic_review": None,
                    "unreachable_decision": None,
                    "human_unreachable_review": None,
                    "research_briefing": None,
                    "complete_batch_history": None,
                }
            )
            injected_research_briefing = None
            injected_complete_batch_history = None
            injected_candidate_briefing = None
            injected_critic_review = None
            injected_unreachable_critic_review = None
        elif action.name == "diagnose_dataset" and result.status == "success":
            artifacts = artifacts.model_copy(update={"dataset_diagnosis": result.artifacts[0]})
        elif action.name == "screen_composition_effects" and result.status == "success":
            artifacts = artifacts.model_copy(update={"composition_effects": result.artifacts[0]})
        elif action.name == "compile_research_analysis" and result.status == "success":
            artifacts = artifacts.model_copy(update={"research_analysis": result.artifacts[0]})
        elif (
            action.name == "fit_local_response_model"
            and result.status == "success"
            and result.data.get("fit_status") == "supported"
        ):
            artifacts = artifacts.model_copy(
                update={"response_models": list(dict.fromkeys([*artifacts.response_models, result.artifacts[0]]))}
            )
        elif action.name == "compare_models" and result.status == "success":
            artifacts = artifacts.model_copy(update={"model_comparison": result.artifacts[0]})
        elif action.name == "write_design_decision" and result.status == "success":
            artifacts = artifacts.model_copy(update={"design_decision": result.artifacts[0]})
        elif action.name == "generate_predicted_candidates" and result.status == "success":
            artifacts = artifacts.model_copy(update={"predicted_candidates": result.artifacts[0]})
            injected_candidate_briefing = result.data
        elif action.name == "get_research_briefing" and result.status == "success":
            artifacts = artifacts.model_copy(update={"research_briefing": result.artifacts[0]})
            injected_research_briefing = result.data
        elif action.name == "get_complete_batch_history" and result.status == "success":
            artifacts = artifacts.model_copy(update={"complete_batch_history": result.artifacts[0]})
            injected_complete_batch_history = result.data
        elif action.name == "check_goal" and result.status == "success":
            verification_status = str(result.data.get("verification_status", "not_met"))
            if verification_status == "confirmed":
                phase = Phase.FINISHED
                provisional_goal_candidate = None
                result_summary = "target met and independently confirmed by deterministic CIE checks"
            elif verification_status == "provisional_hit":
                provisional_goal_candidate = ProvisionalGoalCandidate.model_validate(
                    result.data["provisional_candidate"]
                )
                if completed_round >= state.max_rounds:
                    phase = Phase.STOPPED
                    result_summary = (
                        "target tolerance reached but independent confirmation could not be "
                        "completed before the maximum experiment round"
                    )
                else:
                    phase = Phase.DESIGNING
                    result_summary = None
            elif verification_status == "confirmation_failed":
                provisional_goal_candidate = None
                if completed_round >= state.max_rounds:
                    phase = Phase.STOPPED
                    result_summary = (
                        "independent target confirmation failed at the maximum experiment round"
                    )
                else:
                    phase = Phase.DESIGNING
                    result_summary = None
            elif verification_status == "confirmation_ineligible":
                # Synthetic or otherwise ineligible data cannot consume the real candidate.
                if completed_round >= state.max_rounds:
                    phase = Phase.STOPPED
                    result_summary = (
                        "target candidate remained unconfirmed at the maximum experiment round"
                    )
                else:
                    phase = Phase.DESIGNING
                    result_summary = None
            elif completed_round >= state.max_rounds:
                phase = Phase.STOPPED
                result_summary = "maximum experiment rounds reached without meeting the target"
            else:
                phase = Phase.DESIGNING
                result_summary = None
            artifacts = artifacts.model_copy(update={"final_report": result.artifacts[0]})

        return state.model_copy(
            update={
                "phase": phase,
                "artifacts": artifacts,
                "confirmed_materials": confirmed_materials,
                "active_skills": active_skills,
                "round": completed_round,
                "pending_measurement_path": pending_measurement_path,
                "pending_data_origin": pending_data_origin,
                "injected_research_briefing": injected_research_briefing,
                "injected_complete_batch_history": injected_complete_batch_history,
                "injected_candidate_briefing": injected_candidate_briefing,
                "injected_critic_review": injected_critic_review,
                "injected_unreachable_critic_review": injected_unreachable_critic_review,
                "provisional_goal_candidate": provisional_goal_candidate,
                "history": [
                    *state.history,
                    Observation(
                        action=action.name,
                        status=observation_status,
                        summary=result.summary,
                        artifacts=result.artifacts,
                    ),
                ],
                "last_action": LastAction(
                    name=action.name,
                    status="success" if result.status == "success" else "failed",
                ),
                "result": result_summary,
            }
        )

    return store.update(mutate)


def record_rejection(store: TaskStore, action: Action, reason: str) -> TaskState:
    """Keep an invalid Director proposal durable before another decision or pause."""
    return store.update(
        lambda state: state.model_copy(
            update={
                "history": [
                    *state.history,
                    Observation(action=action.name, status="rejected", summary=reason),
                ],
                "last_action": LastAction(name=action.name, status="rejected"),
            }
        )
    )
