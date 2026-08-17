"""One-shot Scientific Critic for a Director-authored follow-up batch draft."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import urlparse

from ..cie import CIE_TOLERANCE
from ..models import TaskState
from ..storage import TaskStore
from .unreachable import validate_unreachable_review

CPWL_RECIPE_SLOT_COUNT = 5

CRITIC_INSTRUCTION = """
You are a lightweight Scientific Critic reviewing one colour-experiment batch draft.
Your purpose is to improve the probability of reaching the target CIE quickly under a
small experiment budget. Review the actual recipes against the supplied measured facts
and deterministic draft evaluation. Check whether the batch concentrates effort in the
best-supported region, whether recipe changes match the stated reasoning, whether a
better-supported measured region was overlooked, whether extrapolation or weak trends
are overstated, whether recipes are redundant, and whether the competing option has
real opportunity value. Do not invent optical mechanisms or measured facts. Do not
submit an alternative batch and do not claim authority over the Director. Return one
structured review through submit_critic_review. A pass may contain no findings; every
finding must be actionable and evidence-linked where evidence exists. `material_slots`
and every recipe's `recipe_components` are the canonical composition facts. Never infer
material identity from an array position. Before reporting a
`recipe_composition_mismatch`, compare the declared recipe wording with the named
component records. Its reason must identify the relevant slot, material, concentration,
and molar fraction.
""".strip()

UNREACHABLE_CRITIC_INSTRUCTION = """
You are a lightweight Scientific Critic reviewing a bounded claim that further colour
experiments are not justified for the current materials, explored region, constraints,
and remaining budget. This is not a claim of universal physical impossibility. Check
whether the Director cited only supplied measured evidence, represented the strongest
target-near results and attempted routes fairly, acknowledged plausible remaining
options, and separated measured facts from interpretations. Recommend
support_application only when the bounded case is adequately supported; otherwise use
revise or continue_experiments. Do not invent measurements, mechanisms, or alternative
recipes. Return one structured review through submit_unreachable_review. The review is
advisory: the Director chooses whether to continue or submit, and only a human approval
can stop the task early.
""".strip()


class ScientificCritic(Protocol):
    """Review one bounded evidence packet without modifying task state."""

    model: str
    last_usage: dict[str, int | None] | None

    def review(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Return a validated structured review."""

    def review_unreachable(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Return a validated review of one bounded unreachable application."""


class ScriptedScientificCritic:
    """Return one predetermined review for offline demonstrations and unit tests."""

    def __init__(
        self,
        review: dict[str, Any],
        unreachable_review: dict[str, Any] | None = None,
    ) -> None:
        """Store a copy so caller mutation cannot change the scripted response."""
        self._review = json.loads(json.dumps(review))
        self._unreachable_review = json.loads(
            json.dumps(
                unreachable_review
                or {
                    "recommendation": "support_application",
                    "findings": [],
                    "acceptable_aspects": ["The application is evidence-bounded."],
                    "unresolved_uncertainties": [],
                }
            )
        )
        self.model = "scripted-scientific-critic"
        self.last_usage = None

    def review(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Validate the scripted review against the current packet."""
        del packet
        return _validate_review(self._review)

    def review_unreachable(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Validate the separately scripted unreachable review."""
        del packet
        return validate_unreachable_review(self._unreachable_review)


class OpenAIScientificCritic:
    """Use one isolated OpenAI-compatible request for scientific review."""

    def __init__(self, model: str) -> None:
        """Construct a separate stateless client using the same configured API endpoint."""
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required to run the Scientific Critic")
        try:
            from openai import OpenAI, OpenAIError
        except ImportError as error:
            raise RuntimeError("install react-color-agent[openai] for the Scientific Critic") from error
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
        self.openai_error = OpenAIError
        self.model = model
        self.base_url_host = urlparse(
            os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).hostname
        self.last_usage: dict[str, int | None] | None = None

    def review(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Request one review, preferring a tool call with a JSON-text compatibility fallback."""
        tool = {
            "type": "function",
            "function": {
                "name": "submit_critic_review",
                "description": "Submit the single advisory scientific review.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["pass", "pass_with_warning", "revise"]},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "finding_id": {"type": "string"},
                                    "severity": {"type": "string", "enum": ["critical", "warning"]},
                                    "category": {"type": "string"},
                                    "claim": {"type": "string"},
                                    "evidence_sample_ids": {"type": "array", "items": {"type": "string"}},
                                    "reason": {"type": "string"},
                                    "suggestion": {"type": "string"},
                                },
                                "required": ["finding_id", "severity", "category", "claim", "evidence_sample_ids", "reason", "suggestion"],
                                "additionalProperties": False,
                            },
                        },
                        "acceptable_aspects": {"type": "array", "items": {"type": "string"}},
                        "unresolved_uncertainties": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["verdict", "findings", "acceptable_aspects", "unresolved_uncertainties"],
                    "additionalProperties": False,
                },
            },
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CRITIC_INSTRUCTION},
                    {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
                ],
                tools=[tool],
                # Some thinking-mode OpenAI-compatible providers reject a forced
                # function tool_choice even though automatic tool calling works.
                tool_choice="auto",
            )
        except self.openai_error as error:
            raise RuntimeError(f"Scientific Critic API request failed: {error}") from error
        self.last_usage = _usage_dict(getattr(response, "usage", None))
        message = response.choices[0].message
        calls = message.tool_calls or []
        if len(calls) == 1 and calls[0].function.name == "submit_critic_review":
            raw_payload = calls[0].function.arguments or "{}"
        elif not calls and isinstance(message.content, str):
            # Keep the Critic one-shot: parse a JSON response from the same call
            # when the provider elects not to use the offered function.
            raw_payload = _strip_json_fence(message.content)
        else:
            raise RuntimeError("Scientific Critic did not return one structured review")
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as error:
            raise RuntimeError("Scientific Critic returned invalid JSON") from error
        return _validate_review(payload)

    def review_unreachable(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Request one advisory review using the dedicated unreachable contract."""
        tool = {
            "type": "function",
            "function": {
                "name": "submit_unreachable_review",
                "description": "Submit the single advisory review of the bounded unreachable application.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recommendation": {
                            "type": "string",
                            "enum": ["support_application", "revise", "continue_experiments"],
                        },
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "finding_id": {"type": "string"},
                                    "severity": {"type": "string", "enum": ["critical", "warning"]},
                                    "category": {"type": "string"},
                                    "claim": {"type": "string"},
                                    "evidence_sample_ids": {"type": "array", "items": {"type": "string"}},
                                    "reason": {"type": "string"},
                                    "suggestion": {"type": "string"},
                                },
                                "required": ["finding_id", "severity", "category", "claim", "evidence_sample_ids", "reason", "suggestion"],
                                "additionalProperties": False,
                            },
                        },
                        "acceptable_aspects": {"type": "array", "items": {"type": "string"}},
                        "unresolved_uncertainties": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["recommendation", "findings", "acceptable_aspects", "unresolved_uncertainties"],
                    "additionalProperties": False,
                },
            },
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": UNREACHABLE_CRITIC_INSTRUCTION},
                    {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
                ],
                tools=[tool],
                tool_choice="auto",
            )
        except self.openai_error as error:
            raise RuntimeError(f"Unreachable Scientific Critic API request failed: {error}") from error
        self.last_usage = _usage_dict(getattr(response, "usage", None))
        message = response.choices[0].message
        calls = message.tool_calls or []
        if len(calls) == 1 and calls[0].function.name == "submit_unreachable_review":
            raw_payload = calls[0].function.arguments or "{}"
        elif not calls and isinstance(message.content, str):
            # Preserve the one-shot boundary for providers that answer with JSON text.
            raw_payload = _strip_json_fence(message.content)
        else:
            raise RuntimeError("Unreachable Scientific Critic did not return one structured review")
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as error:
            raise RuntimeError("Unreachable Scientific Critic returned invalid JSON") from error
        return validate_unreachable_review(payload)


def _material_slots(materials: list[str]) -> list[dict[str, str]]:
    """Return the canonical ordered material mapping for one task."""
    return [
        {"slot": chr(ord("A") + index), "material": material}
        for index, material in enumerate(materials)
    ]


def _recipe_components(
    recipe: Mapping[str, Any], material_slots: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Expand one fixed A-E recipe into active named task components."""
    material_count = len(material_slots)
    if not 1 <= material_count <= CPWL_RECIPE_SLOT_COUNT:
        raise ValueError("task materials must occupy one to five A-E slots")
    concentrations = recipe.get("concentrations_mmol_ml")
    if not isinstance(concentrations, list) or len(concentrations) != CPWL_RECIPE_SLOT_COUNT:
        raise ValueError("recipe concentration array must match the task material slots")
    # CPWL persists five A-E positions even when a task uses fewer materials.
    if any(float(value) != 0 for value in concentrations[material_count:]):
        raise ValueError("recipe concentration array uses a non-zero unused material slot")
    total = sum(float(value) for value in concentrations)
    if total <= 0:
        raise ValueError("recipe concentration array must contain a positive total")
    fractions = recipe.get("molar_fractions")
    if fractions is not None and (
        not isinstance(fractions, list) or len(fractions) != CPWL_RECIPE_SLOT_COUNT
    ):
        raise ValueError("recipe molar-fraction array must match the task material slots")
    if fractions is not None and any(
        float(value) != 0 for value in fractions[material_count:]
    ):
        raise ValueError("recipe molar-fraction array uses a non-zero unused material slot")
    return [
        {
            "slot": slot["slot"],
            "material": slot["material"],
            "concentration_mmol_ml": float(concentration),
            "molar_fraction": (
                float(fractions[index]) if fractions is not None else float(concentration) / total
            ),
        }
        for index, (slot, concentration) in enumerate(
            zip(material_slots, concentrations[:material_count], strict=True)
        )
        if float(concentration) > 0
    ]


def build_critic_packet(store: TaskStore, state: TaskState) -> dict[str, Any]:
    """Build a compact facts-first packet from controlled task artifacts."""
    if not state.artifacts.research_dataset or not state.artifacts.batch_draft or not state.artifacts.batch_draft_evaluation:
        raise ValueError("critic packet requires dataset, draft, and draft evaluation artifacts")
    dataset = _load_json(store, state.artifacts.research_dataset)
    draft = _load_json(store, state.artifacts.batch_draft)
    evaluation = _load_json(store, state.artifacts.batch_draft_evaluation)
    material_slots = _material_slots(state.materials)
    critic_draft = json.loads(json.dumps(draft))
    for recipe in critic_draft.get("recipes", []):
        recipe["recipe_components"] = _recipe_components(recipe, material_slots)
    measured = [
        observation
        for batch in dataset.get("batches", [])
        if bool(batch.get("origin", {}).get("scientific_eligible"))
        for observation in batch.get("observations", [])
    ]
    measured_by_id = {
        str(observation["identity"]["sample_id"]): observation for observation in measured
    }
    nearest = sorted(measured, key=lambda item: float(item["evaluation"]["distance"]))[:5]
    cited = [
        measured_by_id[sample_id]
        for sample_id in draft.get("facts_used_sample_ids", [])
        if sample_id in measured_by_id
    ]
    return {
        "task": {
            "task_id": state.task_id,
            "target_cie": list(state.target),
            "target_tolerance": CIE_TOLERANCE,
            "target_tolerance_mode": "coordinate_wise",
            "target_tolerance_rule": (
                "|x-x_target| <= 0.005 and |y-y_target| <= 0.005; "
                "distance remains Euclidean for ranking/reporting"
            ),
            "completed_round": state.round,
            "next_batch_limit": 12,
        },
        "material_slots": material_slots,
        "nearest_measured_samples": [
            _compact_observation(item, material_slots) for item in nearest
        ],
        "director_cited_measured_records": [
            _compact_observation(item, material_slots) for item in cited
        ],
        "director_draft": critic_draft,
        "deterministic_draft_evaluation": evaluation,
        "authority_boundary": (
            "Critic is advisory. The Director must respond to every finding and retains final authority."
        ),
    }


def _compact_observation(
    observation: dict[str, Any], material_slots: list[dict[str, str]]
) -> dict[str, Any]:
    """Keep only the measured formulation and CIE facts needed for batch review."""
    return {
        "sample_id": observation["identity"]["sample_id"],
        "concentrations_mmol_ml": observation["recipe"]["concentrations_mmol_ml"],
        "molar_fractions": observation["recipe"]["molar_fractions"],
        "recipe_components": _recipe_components(observation["recipe"], material_slots),
        "cie": observation["measurement"]["cie"],
        "target_distance": observation["evaluation"]["distance"],
    }


def _load_json(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Read JSON through the run-local artifact path guard."""
    try:
        return json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load critic source artifact {artifact}") from error


def _strip_json_fence(content: str) -> str:
    """Remove one optional Markdown fence from an otherwise structured JSON response."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _validate_review(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the Critic response before it can become durable evidence of Agent behavior."""
    verdict = str(payload.get("verdict", ""))
    if verdict not in {"pass", "pass_with_warning", "revise"}:
        raise ValueError("critic verdict must be pass, pass_with_warning, or revise")
    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list) or not all(isinstance(item, dict) for item in raw_findings):
        raise ValueError("critic findings must be a list")
    findings = []
    for item in raw_findings:
        finding = {
            "finding_id": str(item.get("finding_id", "")).strip(),
            "severity": str(item.get("severity", "")).strip(),
            "category": str(item.get("category", "")).strip(),
            "claim": str(item.get("claim", "")).strip(),
            "evidence_sample_ids": [str(value).strip() for value in item.get("evidence_sample_ids", [])],
            "reason": str(item.get("reason", "")).strip(),
            "suggestion": str(item.get("suggestion", "")).strip(),
        }
        if finding["severity"] not in {"critical", "warning"}:
            raise ValueError("critic finding severity must be critical or warning")
        if any(not finding[key] for key in ("finding_id", "category", "claim", "reason", "suggestion")):
            raise ValueError("critic findings require non-empty structured fields")
        findings.append(finding)
    finding_ids = [item["finding_id"] for item in findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ValueError("critic finding IDs must be unique")
    if verdict == "pass" and findings:
        raise ValueError("a pass verdict must not contain findings")
    return {
        "verdict": verdict,
        "findings": findings,
        "acceptable_aspects": [str(value) for value in payload.get("acceptable_aspects", [])],
        "unresolved_uncertainties": [str(value) for value in payload.get("unresolved_uncertainties", [])],
    }


def _usage_dict(usage: object) -> dict[str, int | None] | None:
    """Extract provider token counts without retaining a remote response body."""
    if usage is None:
        return None
    getter = usage.get if isinstance(usage, Mapping) else lambda key, default=None: getattr(usage, key, default)
    result: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getter(name)
        result[name] = int(value) if isinstance(value, (int, float)) else None
    return result
