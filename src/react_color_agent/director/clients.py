"""Decision clients that translate one task state into one requested tool action."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Protocol
from urllib.parse import urlparse

from ..models import Action, TaskState, ToolResult
from ..skill_registry import SkillRegistry
from .contracts import DIRECTOR_INSTRUCTION, TOOL_SCHEMAS


class DirectorAPIError(RuntimeError):
    """An expected remote Director-provider failure that callers can report cleanly."""


class DecisionClient(Protocol):
    """Choose one next action from task state without executing any local tool."""

    def decide(self, state: TaskState, tool_names: Sequence[str]) -> Action | None:
        """Return the requested tool invocation, or pause when no choice is made."""

    def observe(self, action: Action, result: "ToolResult") -> None:
        """Receive the completed tool result before the next decision."""

    def redirect(self, message: str) -> None:
        """Receive one runtime correction when a required workflow action was omitted."""


class ScriptedDecisionClient:
    """A deterministic client for tests and reproducible local demonstrations."""

    def __init__(self, actions: Sequence[Action | None]) -> None:
        """Copy a finite sequence so each decision is consumed at most once."""
        self.actions = list(actions)
        self.redirects: list[str] = []

    def decide(self, state: TaskState, tool_names: Sequence[str]) -> Action | None:
        """Return the next scripted action while deliberately ignoring runtime details."""
        del state, tool_names
        return self.actions.pop(0) if self.actions else None

    def observe(self, action: Action, result: "ToolResult") -> None:
        """Ignore results because scripted actions are already fixed."""
        del action, result

    def redirect(self, message: str) -> None:
        """Record runtime guidance so deterministic tests can inspect recovery behavior."""
        self.redirects.append(message)


class OpenAIChatDecisionClient:
    """Optional OpenAI tool-calling client constructed only with an explicit API key."""

    def __init__(self, model: str = "gpt-4.1-mini", skill_registry: SkillRegistry | None = None) -> None:
        """Initialize the optional SDK lazily to keep package installation lightweight."""
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required to run the OpenAI Director")
        try:
            from openai import OpenAI, OpenAIError
        except ImportError as error:
            raise RuntimeError("install react-color-agent[openai] for the OpenAI Director") from error
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
        self.openai_error = OpenAIError
        self.model = model
        # Retain only the host for reproducibility; paths and query strings may contain secrets.
        self.base_url_host = urlparse(
            os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).hostname
        self.skill_registry = skill_registry or SkillRegistry({})
        # Populated after each request for the isolated evaluation recorder.
        self.last_usage: dict[str, int | None] | None = None
        # Tool exchanges live only for this CLI process. Durable task state keeps
        # compact summaries, while the next ReAct turn receives the full payload.
        self._tool_messages: list[dict[str, object]] = []
        self._pending_tool_call_id: str | None = None

    def decide(self, state: TaskState, tool_names: Sequence[str]) -> Action | None:
        """Ask the model for one tool call while exposing only registered tool names."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    **TOOL_SCHEMAS.get(
                        name,
                        {
                            "description": f"Run the registered {name} tool.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    ),
                },
            }
            for name in tool_names
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DIRECTOR_INSTRUCTION},
                    {"role": "user", "content": self._user_context(state)},
                    *self._tool_messages,
                ],
                tools=tools,
                tool_choice="auto",
            )
        except self.openai_error as error:
            raise DirectorAPIError(f"OpenAI request failed: {error}") from error
        self.last_usage = _usage_dict(getattr(response, "usage", None))
        calls = response.choices[0].message.tool_calls or []
        if not calls:
            return None
        call = calls[0]
        call_id = str(call.id or f"call_{len(self._tool_messages)}")
        raw_arguments = call.function.arguments or "{}"
        # Retain only the selected call. Returning unexecuted parallel calls in the
        # transcript would violate the Chat Completions tool-message contract.
        response_message = response.choices[0].message
        assistant_message: dict[str, object] = {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": raw_arguments,
                    },
                }
            ],
        }
        # Thinking-mode compatible APIs require their opaque reasoning payload
        # to be replayed with the assistant tool call on the following request.
        reasoning_content = getattr(response_message, "reasoning_content", None)
        if reasoning_content is not None:
            assistant_message["reasoning_content"] = reasoning_content
        self._tool_messages.append(assistant_message)
        self._pending_tool_call_id = call_id
        return Action(name=call.function.name, arguments=json.loads(raw_arguments))

    def observe(self, action: Action, result: "ToolResult") -> None:
        """Append the full result as the tool response for the pending model call."""
        if self._pending_tool_call_id is None:
            return
        self._tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": self._pending_tool_call_id,
                "content": result.model_dump_json(),
            }
        )
        self._pending_tool_call_id = None

    def redirect(self, message: str) -> None:
        """Append a workflow correction as a user turn before the next model decision."""
        self._tool_messages.append({"role": "user", "content": message})

    def _user_context(self, state: TaskState) -> str:
        """Attach only explicitly activated local Skill bodies to the task-state context."""
        skill_context = self.skill_registry.prompt_context(state.active_skills)
        progress_hint = _design_progress_hint(state)
        if not skill_context:
            return f"{state.model_dump_json(indent=2)}{progress_hint}"
        return f"Task state:\n{state.model_dump_json(indent=2)}{progress_hint}\n\n{skill_context}"


def _design_progress_hint(state: TaskState) -> str:
    """Tell the model when retrieval has become a loop and a scientific decision is due."""
    if state.phase.value != "DESIGNING":
        return ""
    retrieval_tools = {
        "query_research_index",
        "get_experiment_record",
        "get_spectrum_data",
        "get_analysis_record",
        "get_complete_batch_history",
        "get_research_briefing",
    }
    progress_tools = {
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
    streak = 0
    for observation in reversed(state.history):
        if (
            observation.action in progress_tools
            and observation.status == "success"
            and not (
                observation.action == "fit_local_response_model"
                and observation.summary.startswith("Did not fit ")
            )
        ):
            break
        if observation.action in retrieval_tools:
            streak += 1
    if streak < 3:
        return ""
    return (
        "\n\nDecision guidance: the last "
        f"{streak} Director actions were retrieval-only. Treat the available evidence as sufficient "
        "for broad lookup. Read only specific records or spectra required by the current hypothesis, "
        "then choose an analysis/design-decision/follow-up-design tool, or return no tool call to pause. "
        "Do not issue another broad or filter-variant index query."
    )


def _usage_dict(usage: object) -> dict[str, int | None] | None:
    """Extract provider-reported token counts without retaining the full response."""
    if usage is None:
        return None
    getter = usage.get if isinstance(usage, Mapping) else lambda key, default=None: getattr(usage, key, default)
    result: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getter(name)
        result[name] = int(value) if isinstance(value, (int, float)) else None
    # Some providers expose cached prompt tokens as a nested usage detail.
    prompt_details = getter("prompt_tokens_details")
    if isinstance(prompt_details, Mapping):
        cached = prompt_details.get("cached_tokens")
    else:
        cached = getattr(prompt_details, "cached_tokens", None) if prompt_details is not None else None
    result["cached_prompt_tokens"] = int(cached) if isinstance(cached, (int, float)) else None
    return result
