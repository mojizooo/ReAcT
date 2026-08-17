"""The thin ReAct orchestration layer for one Experiment Director Agent."""

from __future__ import annotations

import json
import hashlib
import time
from collections.abc import Callable
from typing import Any

from ..models import Action, Phase, TaskState, ToolResult
from ..storage import TaskStore
from ..evaluation.trace_recorder import TraceRecorder
from ..evaluation.protocol_fingerprint import build_protocol_fingerprint
from .clients import DecisionClient
from .critic import ScientificCritic, build_critic_packet
from .context import enrich_arguments, refresh_research_briefing
from .guards import can_execute
from .transitions import apply_result, mark_running, persist_pubchem_result, record_rejection
from .unreachable import unreachable_critic_packet
from tools.unreachable_tools import measured_records

# These tools only retrieve already-recorded evidence.  A bounded streak prevents
# a Director from spending the whole run changing filters on the same index.
_DESIGN_READ_ONLY_TOOLS = frozenset(
    {
        "query_research_index",
        "get_experiment_record",
        "get_spectrum_data",
        "get_analysis_record",
        "get_complete_batch_history",
        "get_research_briefing",
    }
)

# Only these actions demonstrate scientific progress and reset retrieval budgets.
_DESIGN_PROGRESS_TOOLS = frozenset(
    {
        "update_research_dataset",
        "check_goal",
        "review_design_outcomes",
        "extract_spectral_features",
        "diagnose_dataset",
        "screen_composition_effects",
        "compile_research_analysis",
        "fit_local_response_model",
        "compare_models",
        "write_design_decision",
        "generate_predicted_candidates",
        "design_followup_batch",
        "design_exploratory_followup_batch",
        "propose_followup_batch",
        "finalize_followup_batch",
    }
)

_CRITIC_NO_ACTION_RETRY_LIMIT = 1
_CRITIC_RESPONSE_REDIRECT = (
    "The Scientific Critic review is already saved, so this reviewed design stage cannot pause "
    "without a final Director submission. Inspect every Critic finding, state whether it is accepted, "
    "partially accepted, or rejected with evidence, and call finalize_followup_batch with the complete "
    "final batch. You may first use a currently offered read-only evidence tool only when a specific "
    "finding requires it. Do not return plain text or no tool call."
)


class DirectorRuntime:
    """Run Director-selected tools until laboratory data or a terminal state is needed."""

    def __init__(
        self,
        store: TaskStore,
        tools: dict[str, Any],
        client: DecisionClient,
        trace: Callable[[str], None] | None = None,
        recorder: TraceRecorder | None = None,
        critic: ScientificCritic | None = None,
        max_read_only_steps: int = 10,
        max_index_queries: int = 5,
    ) -> None:
        """Keep orchestration dependencies limited to one store, tool map, and decision client."""
        if max_read_only_steps < 1:
            raise ValueError("max_read_only_steps must be at least 1")
        if max_index_queries < 1:
            raise ValueError("max_index_queries must be at least 1")
        self.store = store
        self.tools = tools
        self.client = client
        self.trace = trace
        self.recorder = recorder
        self.critic = critic
        self.max_read_only_steps = max_read_only_steps
        self.max_index_queries = max_index_queries

    def run_until_pause(self, max_steps: int = 40, resume_waiting: bool = False) -> TaskState:
        """Execute bounded ReAct turns while preserving every result in the task store."""
        protocol = self._protocol_fingerprint(max_steps)
        self._event(
            "runtime_started",
            model=getattr(self.client, "model", None),
            tool_count=len(self.tools),
            protocol_sha256=protocol.get("protocol_sha256") if protocol else None,
            protocol=protocol,
        )
        consumed_waiting_resume = False
        read_only_streak: int | None = None
        index_query_count: int | None = None
        seen_index_arguments: set[str] = set()
        previous_index_result: str | None = None
        repeated_index_result_streak = 0
        critic_no_action_retries = 0
        for _ in range(max_steps):
            state = self.store.load()
            if read_only_streak is None:
                read_only_streak = self._historical_read_only_streak(state)
                index_query_count = self._historical_index_query_count(state)
            state = refresh_research_briefing(self.store, self.tools, state, self._trace)
            if state.phase is Phase.AWAITING_HUMAN_REVIEW:
                self._trace("paused: waiting for explicit human review of the unreachable application")
                self._event(
                    "task_paused",
                    reason="awaiting_human_review",
                    phase=state.phase.value,
                    round=state.round,
                )
                return state
            if self._unreachable_critic_is_pending(state):
                if self.critic is None:
                    self._trace("paused: Unreachable Scientific Critic is required for the saved application")
                    self._event(
                        "task_paused",
                        reason="unreachable_critic_not_configured",
                        phase=state.phase.value,
                        round=state.round,
                    )
                    return state
                self._run_unreachable_scientific_critic(state)
                continue
            if self._critic_is_pending(state):
                if self.critic is None:
                    self._trace("paused: Scientific Critic is required for the saved batch draft")
                    self._event("task_paused", reason="critic_not_configured", phase=state.phase.value, round=state.round)
                    return state
                self._run_scientific_critic(state)
                # Start a fresh Director decision with the durable Critic result in task state.
                continue
            self._trace_state(state)
            self._event(
                "state_observed",
                phase=state.phase.value,
                round=state.round,
                history_count=len(state.history),
            )
            if state.phase in {Phase.FINISHED, Phase.STOPPED}:
                self._trace(f"task reached terminal phase {state.phase.value}")
                self._event("task_terminal", phase=state.phase.value, round=state.round)
                return state
            if state.phase is Phase.WAITING_FOR_DATA:
                if not resume_waiting or consumed_waiting_resume:
                    self._trace("paused: waiting for user measurement data")
                    self._event("task_paused", reason="waiting_for_data", phase=state.phase.value, round=state.round)
                    return state
                consumed_waiting_resume = True

            self._trace("requesting next action from Director")
            self._event("decision_requested", phase=state.phase.value, round=state.round)
            decision_started = time.perf_counter()
            decision_tools = self._decision_tool_names(state, read_only_streak, index_query_count)
            try:
                action = self.client.decide(state, decision_tools)
            except Exception as error:
                self._event(
                    "decision_failed",
                    error_type=type(error).__name__,
                    elapsed_ms=round((time.perf_counter() - decision_started) * 1000, 3),
                )
                raise
            self._event(
                "decision_finished",
                status="action" if action is not None else "no_action",
                elapsed_ms=round((time.perf_counter() - decision_started) * 1000, 3),
                token_usage=getattr(self.client, "last_usage", None),
            )
            if action is None:
                if self._requires_unreachable_response(state):
                    if critic_no_action_retries < _CRITIC_NO_ACTION_RETRY_LIMIT:
                        critic_no_action_retries += 1
                        self._trace(
                            "Director returned no tool call after the unreachable Critic review; "
                            "requesting the required continue-or-submit decision once more"
                        )
                        self._event(
                            "unreachable_critic_no_action_retry",
                            phase=state.phase.value,
                            round=state.round,
                            attempt=critic_no_action_retries,
                            limit=_CRITIC_NO_ACTION_RETRY_LIMIT,
                        )
                        self._redirect(
                            "The Unreachable Scientific Critic review is saved. Respond to every finding and call "
                            "continue_after_unreachable_review or submit_unreachable_application; do not return plain text."
                        )
                        continue
                    self._event(
                        "unreachable_critic_no_action_retry_exhausted",
                        phase=state.phase.value,
                        round=state.round,
                        limit=_CRITIC_NO_ACTION_RETRY_LIMIT,
                    )
                    self._event(
                        "task_paused",
                        reason="unreachable_critic_no_action_retry_exhausted",
                        phase=state.phase.value,
                        round=state.round,
                    )
                    return state
                if self._requires_critic_response(state):
                    if critic_no_action_retries < _CRITIC_NO_ACTION_RETRY_LIMIT:
                        critic_no_action_retries += 1
                        self._trace(
                            "Director returned no tool call after Scientific Critic review; "
                            "requesting the required final submission once more"
                        )
                        self._event(
                            "critic_no_action_retry",
                            phase=state.phase.value,
                            round=state.round,
                            attempt=critic_no_action_retries,
                            limit=_CRITIC_NO_ACTION_RETRY_LIMIT,
                        )
                        self._redirect(_CRITIC_RESPONSE_REDIRECT)
                        continue
                    self._trace(
                        "paused: Director again returned no tool call after Scientific Critic review; "
                        "the bounded correction retry is exhausted"
                    )
                    self._event(
                        "critic_no_action_retry_exhausted",
                        phase=state.phase.value,
                        round=state.round,
                        limit=_CRITIC_NO_ACTION_RETRY_LIMIT,
                    )
                    self._event(
                        "task_paused",
                        reason="critic_no_action_retry_exhausted",
                        phase=state.phase.value,
                        round=state.round,
                    )
                    return state
                self._trace("Director returned no tool call; task remains paused")
                self._event("task_paused", reason="director_returned_no_action", phase=state.phase.value, round=state.round)
                return state
            self._trace(f"selected {action.name} {self._format_arguments(action.arguments)}")
            self._event(
                "action_selected",
                action=action.name,
                arguments=action.arguments,
                phase=state.phase.value,
                round=state.round,
            )
            if action.name == "query_research_index":
                argument_fingerprint = self._fingerprint(action.arguments)
                if argument_fingerprint in seen_index_arguments:
                    reason = "identical research-index query already executed in this Director run; reuse its result"
                    self._reject_for_retry(action, state, reason, "duplicate_index_arguments")
                    read_only_streak += 1
                    index_query_count += 1
                    continue
                seen_index_arguments.add(argument_fingerprint)

            if action.name not in decision_tools:
                # OpenAI-compatible providers can occasionally return a historical tool
                # call even after that tool has been removed from the current request.
                if self._is_design_read_only(action, state) and read_only_streak >= self.max_read_only_steps:
                    reason = (
                        "read-only design-tool budget exhausted after "
                        f"{read_only_streak} consecutive calls; choose an analysis/design tool "
                        "or pause for a human decision"
                    )
                    event_reason = "read_only_budget_exceeded"
                elif action.name == "query_research_index" and index_query_count >= self.max_index_queries:
                    reason = (
                        "research-index query budget exhausted after "
                        f"{index_query_count} calls without scientific progress; inspect specific records, "
                        "analyze the available evidence, make a design decision, or pause"
                    )
                    event_reason = "index_query_budget_exceeded"
                else:
                    reason = (
                        f"{action.name} is not available for this decision; use one of the currently "
                        "offered tools or return no tool call to pause"
                    )
                    event_reason = "tool_not_available_for_decision"
                self._reject_for_retry(action, state, reason, event_reason)
                if self._is_design_read_only(action, state):
                    read_only_streak += 1
                    if action.name == "query_research_index":
                        index_query_count += 1
                continue
            rejection = can_execute(action, state)
            if rejection:
                self._trace(f"rejected {action.name}: {rejection}")
                # A failed precondition is a recoverable ReAct observation: the Director
                # can use the concrete reason to select the missing prerequisite next.
                self._reject_for_retry(action, state, rejection, "action_precondition_failed")
                continue
            if action.name not in self.tools:
                self._trace(f"rejected {action.name}: tool is not registered")
                self._reject_for_retry(action, state, "tool is not registered", "tool_not_registered")
                continue

            mark_running(self.store, action)
            result = self._run_tool(action)
            if action.name == "query_research_index" and result.status == "success":
                result_fingerprint = self._fingerprint(result.data.get("matches", result.data))
                if result_fingerprint == previous_index_result:
                    repeated_index_result_streak += 1
                else:
                    previous_index_result = result_fingerprint
                    repeated_index_result_streak = 1
                if repeated_index_result_streak >= 2:
                    result = result.model_copy(
                        update={
                            "summary": (
                                f"{result.summary} Runtime note: this is the same indexed record set as "
                                "the previous query; use the existing evidence instead of changing filters again."
                            )
                        }
                    )
                    self._event(
                        "index_result_repeated",
                        phase=state.phase.value,
                        round=state.round,
                        repeated_count=repeated_index_result_streak,
                    )
            self._trace(f"result {action.name} status={result.status}: {result.summary}")
            persist_pubchem_result(self.store, action, result)
            apply_result(self.store, action, result)
            self._observe(action, result)
            if self._is_design_read_only(action, state):
                read_only_streak += 1
                if action.name == "query_research_index":
                    index_query_count += 1
            elif self._is_design_progress(action, result):
                read_only_streak = 0
                index_query_count = 0
                seen_index_arguments.clear()
                previous_index_result = None
                repeated_index_result_streak = 0

        state = self.store.load()
        self._trace(f"paused: Director reached the {max_steps}-decision safety budget")
        self._event(
            "task_paused",
            reason="max_steps_exceeded",
            phase=state.phase.value,
            round=state.round,
            limit=max_steps,
        )
        return state

    def _protocol_fingerprint(self, max_steps: int) -> dict[str, Any] | None:
        """Collect best-effort evaluation metadata without affecting task execution."""
        try:
            return build_protocol_fingerprint(
                state=self.store.load(),
                tools=self.tools,
                client=self.client,
                max_steps=max_steps,
                max_read_only_steps=self.max_read_only_steps,
                max_index_queries=self.max_index_queries,
            )
        except (OSError, TypeError, ValueError):
            # Protocol collection is evaluation-only and must never block a laboratory run.
            return None

    @staticmethod
    def _is_design_read_only(action: Action, state: TaskState) -> bool:
        """Identify only retrieval loops in DESIGNING; preparation and analysis stay unrestricted."""
        return state.phase is Phase.DESIGNING and action.name in _DESIGN_READ_ONLY_TOOLS

    def _decision_tool_names(
        self,
        state: TaskState,
        read_only_streak: int,
        index_query_count: int,
    ) -> tuple[str, ...]:
        """Constrain only exhausted retrieval routes while leaving scientific choices available."""
        if state.phase is Phase.AWAITING_HUMAN_REVIEW:
            return ()
        names = tuple(self.tools)
        if state.phase is not Phase.DESIGNING:
            return names
        # A recovered or branched measured run already owns durable identity evidence
        # and B1 data. Hide preparation tools so the model cannot restart the task.
        hidden_completed_stage_tools: set[str] = set()
        if state.artifacts.material_evidence:
            hidden_completed_stage_tools.update(
                {"query_pubchem", "search_crossref", "save_material_evidence"}
            )
        if state.round > 0 or state.artifacts.research_dataset:
            hidden_completed_stage_tools.add("design_initial_batch")
        if hidden_completed_stage_tools:
            names = tuple(name for name in names if name not in hidden_completed_stage_tools)
        reviewed_workflow_available = {
            "propose_followup_batch",
            "finalize_followup_batch",
        }.issubset(self.tools)
        unreachable_workflow_active = bool(state.artifacts.unreachable_draft)
        if unreachable_workflow_active:
            if state.artifacts.unreachable_critic_review:
                allowed_after_review = {
                    "continue_after_unreachable_review",
                    "submit_unreachable_application",
                    "query_research_index",
                    "get_experiment_record",
                    "get_spectrum_data",
                    "get_analysis_record",
                    "get_complete_batch_history",
                    "get_research_briefing",
                }
                names = tuple(name for name in names if name in allowed_after_review)
            else:
                # Runtime executes the pending Critic before asking Director again.
                names = tuple(name for name in names if name.startswith("get_") or name == "query_research_index")
            hidden_unreachable: set[str] = set()
            if index_query_count >= self.max_index_queries:
                hidden_unreachable.add("query_research_index")
            if read_only_streak >= self.max_read_only_steps:
                hidden_unreachable.update(_DESIGN_READ_ONLY_TOOLS)
            if hidden_unreachable:
                names = tuple(name for name in names if name not in hidden_unreachable)
            return names
        if state.provisional_goal_candidate is None and reviewed_workflow_available:
            # Normal follow-up plans must pass through draft, evaluation, Critic, and finalization.
            hidden_direct_submit = {
                "write_design_decision",
                "generate_predicted_candidates",
                "design_followup_batch",
                "design_exploratory_followup_batch",
            }
            names = tuple(name for name in names if name not in hidden_direct_submit)
            if state.artifacts.critic_review:
                # Keep focused spectral diagnostics available when the Critic challenges mechanism evidence.
                allowed_after_review = {
                    "finalize_followup_batch",
                    "query_research_index",
                    "get_experiment_record",
                    "get_spectrum_data",
                    "extract_spectral_features",
                    "get_analysis_record",
                    "get_complete_batch_history",
                    "get_research_briefing",
                }
                names = tuple(name for name in names if name in allowed_after_review)
            elif state.artifacts.batch_draft:
                # Runtime handles the pending review before another Director call.
                names = tuple(name for name in names if name == "finalize_followup_batch")
            else:
                names = tuple(name for name in names if name != "finalize_followup_batch")
        elif reviewed_workflow_available:
            # Target confirmation keeps the existing exact-repeat workflow.
            names = tuple(name for name in names if name not in {"propose_followup_batch", "finalize_followup_batch"})
        hidden: set[str] = set()
        if index_query_count >= self.max_index_queries:
            hidden.add("query_research_index")
        if read_only_streak >= self.max_read_only_steps:
            hidden.update(_DESIGN_READ_ONLY_TOOLS)
        if not hidden:
            return names
        constrained = tuple(name for name in names if name not in hidden)
        self._trace("retrieval budget reached; requesting an analysis, design, or pause decision")
        self._event(
            "decision_tools_constrained",
            phase=state.phase.value,
            round=state.round,
            hidden_tools=sorted(hidden.intersection(names)),
            available_tool_count=len(constrained),
        )
        return constrained

    @staticmethod
    def _critic_is_pending(state: TaskState) -> bool:
        """Detect the single durable point where the independent review must run."""
        return (
            state.phase is Phase.DESIGNING
            and state.provisional_goal_candidate is None
            and bool(state.artifacts.batch_draft)
            and bool(state.artifacts.batch_draft_evaluation)
            and not state.artifacts.critic_review
        )

    @staticmethod
    def _unreachable_critic_is_pending(state: TaskState) -> bool:
        """Detect the one-shot review checkpoint for an unreachable application."""
        return (
            state.phase is Phase.DESIGNING
            and bool(state.artifacts.unreachable_draft)
            and bool(state.artifacts.unreachable_evaluation)
            and not state.artifacts.unreachable_critic_review
        )

    @staticmethod
    def _requires_critic_response(state: TaskState) -> bool:
        """Detect a reviewed draft that still requires the Director's final tool submission."""
        return (
            state.phase is Phase.DESIGNING
            and state.provisional_goal_candidate is None
            and bool(state.artifacts.batch_draft)
            and bool(state.artifacts.batch_draft_evaluation)
            and bool(state.artifacts.critic_review)
        )

    @staticmethod
    def _requires_unreachable_response(state: TaskState) -> bool:
        """Require a Director continue-or-submit tool after the dedicated Critic review."""
        return (
            state.phase is Phase.DESIGNING
            and bool(state.artifacts.unreachable_draft)
            and bool(state.artifacts.unreachable_evaluation)
            and bool(state.artifacts.unreachable_critic_review)
            and not state.artifacts.unreachable_decision
        )

    def _run_scientific_critic(self, state: TaskState) -> None:
        """Execute and record one advisory review without entering a second Agent loop."""
        assert self.critic is not None
        self._trace(f"requesting one Scientific Critic review with model {self.critic.model}")
        self._event(
            "critic_requested",
            model=self.critic.model,
            phase=state.phase.value,
            round=state.round,
            draft_artifact=state.artifacts.batch_draft,
        )
        started = time.perf_counter()
        try:
            packet = build_critic_packet(self.store, state)
            review = self.critic.review(packet)
        except Exception as error:
            self._event(
                "critic_failed",
                model=self.critic.model,
                error_type=type(error).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            raise
        evaluation = json.loads(
            self.store.artifact_path(str(state.artifacts.batch_draft_evaluation)).read_text(encoding="utf-8")
        )
        review_artifact_payload = {
            "schema_version": 1,
            "round": state.round,
            "batch_id": f"B{state.round + 1}",
            "model": self.critic.model,
            "source_draft": state.artifacts.batch_draft,
            "source_draft_evaluation": state.artifacts.batch_draft_evaluation,
            "draft_sha256": evaluation["draft_sha256"],
            **review,
            "authority": "advisory_only_director_retains_final_decision",
        }
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{state.round}/batch_{state.round + 1:03}_critic.json",
            review_artifact_payload,
        )
        result = ToolResult(
            status="success",
            summary=(
                f"Scientific Critic returned {review['verdict']} with "
                f"{len(review['findings'])} findings; Director response is required."
            ),
            data=review_artifact_payload,
            artifacts=[artifact],
        )
        apply_result(self.store, Action(name="scientific_critic"), result)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        self._trace(f"Scientific Critic finished verdict={review['verdict']} findings={len(review['findings'])}")
        self._event(
            "critic_finished",
            model=self.critic.model,
            verdict=review["verdict"],
            finding_count=len(review["findings"]),
            token_usage=self.critic.last_usage,
            elapsed_ms=elapsed_ms,
            artifact=artifact,
        )

    def _run_unreachable_scientific_critic(self, state: TaskState) -> None:
        """Execute one evidence-bounded unreachable review and persist its source hashes."""
        assert self.critic is not None
        self._trace(
            f"requesting one Unreachable Scientific Critic review with model {self.critic.model}"
        )
        self._event(
            "unreachable_critic_requested",
            model=self.critic.model,
            phase=state.phase.value,
            round=state.round,
            draft_artifact=state.artifacts.unreachable_draft,
        )
        started = time.perf_counter()
        try:
            draft = json.loads(
                self.store.artifact_path(str(state.artifacts.unreachable_draft)).read_text(encoding="utf-8")
            )
            evaluation = json.loads(
                self.store.artifact_path(str(state.artifacts.unreachable_evaluation)).read_text(encoding="utf-8")
            )
            dataset = json.loads(
                self.store.artifact_path(str(state.artifacts.research_dataset)).read_text(encoding="utf-8")
            )
            records_by_id = {item["sample_id"]: item for item in measured_records(dataset)}
            cited_records = [
                records_by_id[sample_id]
                for sample_id in draft.get("evidence_sample_ids", [])
                if sample_id in records_by_id
            ]
            packet = unreachable_critic_packet(state, draft, evaluation, cited_records)
            review = self.critic.review_unreachable(packet)
        except Exception as error:
            self._event(
                "unreachable_critic_failed",
                model=self.critic.model,
                error_type=type(error).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            raise
        def canonical(payload: Any) -> str:
            """Hash a JSON artifact with the same canonicalization used by research tools."""
            return hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        review_payload = {
            "schema_version": 1,
            "kind": "unreachable_critic_review",
            "task_id": state.task_id,
            "round": state.round,
            "model": self.critic.model,
            "source_dataset": state.artifacts.research_dataset,
            "source_dataset_sha256": canonical(dataset),
            "source_draft": state.artifacts.unreachable_draft,
            "source_draft_sha256": canonical(draft),
            "source_evaluation": state.artifacts.unreachable_evaluation,
            **review,
            "authority": "advisory_only_human_approval_required_to_stop",
        }
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{state.round}/unreachable/critic_review.json", review_payload
        )
        result = ToolResult(
            status="success",
            summary=(
                f"Unreachable Scientific Critic returned {review['recommendation']} with "
                f"{len(review['findings'])} findings; Director must continue or submit."
            ),
            data=review_payload,
            artifacts=[artifact],
        )
        apply_result(self.store, Action(name="unreachable_scientific_critic"), result)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        self._trace(
            f"Unreachable Scientific Critic finished recommendation={review['recommendation']} findings={len(review['findings'])}"
        )
        self._event(
            "unreachable_critic_finished",
            model=self.critic.model,
            recommendation=review["recommendation"],
            finding_count=len(review["findings"]),
            token_usage=self.critic.last_usage,
            elapsed_ms=elapsed_ms,
            artifact=artifact,
        )

    @staticmethod
    def _is_design_progress(action: Action, result: ToolResult) -> bool:
        """Reset retrieval counters only for successful, usable scientific output."""
        if result.status != "success" or action.name not in _DESIGN_PROGRESS_TOOLS:
            return False
        if action.name == "fit_local_response_model":
            return result.data.get("fit_status") == "supported"
        return True

    @staticmethod
    def _historical_read_only_streak(state: TaskState) -> int:
        """Continue the guard across CLI resumes when the previous run ended in a read-only loop."""
        streak = 0
        for observation in reversed(state.history):
            if DirectorRuntime._observation_is_design_progress(observation, state):
                break
            if observation.action in _DESIGN_READ_ONLY_TOOLS:
                streak += 1
        return streak

    @staticmethod
    def _historical_index_query_count(state: TaskState) -> int:
        """Restore the current design-stage index-query budget from durable observations."""
        count = 0
        for observation in reversed(state.history):
            if DirectorRuntime._observation_is_design_progress(observation, state):
                break
            if observation.action == "query_research_index":
                count += 1
        return count

    @staticmethod
    def _observation_is_design_progress(observation: Any, state: TaskState) -> bool:
        """Recognize only durable successful output when restoring retrieval guards."""
        if observation.status != "success" or observation.action not in _DESIGN_PROGRESS_TOOLS:
            return False
        if observation.action == "fit_local_response_model":
            return any(artifact in state.artifacts.response_models for artifact in observation.artifacts)
        return True

    def _observe(self, action: Action, result: ToolResult) -> None:
        """Return a completed tool payload to clients that support ReAct observations."""
        observe = getattr(self.client, "observe", None)
        if callable(observe):
            observe(action, result)

    def _redirect(self, message: str) -> None:
        """Return bounded workflow guidance to clients that retain a live conversation."""
        redirect = getattr(self.client, "redirect", None)
        if callable(redirect):
            redirect(message)

    def _reject_for_retry(
        self,
        action: Action,
        state: TaskState,
        reason: str,
        event_reason: str,
    ) -> None:
        """Persist an invalid choice and return it to ReAct without terminating the run."""
        self._trace(f"rejected {action.name}: {reason}; requesting another decision")
        self._event(
            "action_rejected",
            action=action.name,
            reason=event_reason,
            recoverable=True,
            phase=state.phase.value,
            round=state.round,
        )
        record_rejection(self.store, action, reason)
        self._observe(action, ToolResult(status="failed", summary=reason))

    @staticmethod
    def _fingerprint(value: Any) -> str:
        """Serialize query arguments or result records into a stable equality fingerprint."""
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def _run_tool(self, action: Action) -> ToolResult:
        """Convert unexpected local tool failures into observations the Agent can inspect."""
        state = self.store.load()
        arguments = enrich_arguments(action, state)
        self._trace(f"executing {action.name} {self._format_arguments(arguments)}")
        started = time.perf_counter()
        self._event(
            "tool_started",
            action=action.name,
            arguments=arguments,
            phase=state.phase.value,
            round=state.round,
        )
        try:
            result = self.tools[action.name].run(arguments)
        except Exception as error:  # Tool failures are task observations, not runtime corruption.
            result = ToolResult(status="failed", summary=f"{action.name} failed: {error}")
        self._event(
            "tool_finished",
            action=action.name,
            status=result.status,
            summary=result.summary,
            artifacts=result.artifacts,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return result

    def _trace_state(self, state: TaskState) -> None:
        """Report compact operational state while durable details stay in task artifacts."""
        self._trace(f"state phase={state.phase.value} round={state.round} history={len(state.history)}")

    def _trace(self, message: str) -> None:
        """Emit one live trace record only when the caller requested it."""
        if self.trace is not None:
            self.trace(f"[director] {message}")

    def _event(self, event: str, **fields: Any) -> None:
        """Send optional structured observations to the isolated evaluation sidecar."""
        if self.recorder is not None:
            self.recorder.record(event, **fields)

    @staticmethod
    def _format_arguments(arguments: dict[str, Any]) -> str:
        """Serialize non-secret action arguments deterministically for live trace output."""
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
