"""Pydantic models shared by the Director, tools, and file storage."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Phase(str, Enum):
    """The small set of task phases required by the scientific loop."""

    PREPARING = "PREPARING"
    DESIGNING = "DESIGNING"
    WAITING_FOR_DATA = "WAITING_FOR_DATA"
    ANALYZING = "ANALYZING"
    AWAITING_HUMAN_REVIEW = "AWAITING_HUMAN_REVIEW"
    FINISHED = "FINISHED"
    STOPPED = "STOPPED"


class ArtifactRefs(BaseModel):
    """Relative paths to durable artifacts produced during one task."""

    material_evidence: str | None = None
    experiment_plan: str | None = None
    spectra_manifest: str | None = None
    measurement_result: str | None = None
    analysis_result: str | None = None
    research_dataset: str | None = None
    research_index: str | None = None
    dataset_diagnosis: str | None = None
    composition_effects: str | None = None
    research_analysis: str | None = None
    response_models: list[str] = Field(default_factory=list)
    model_comparison: str | None = None
    design_decision: str | None = None
    predicted_candidates: str | None = None
    exploratory_selection: str | None = None
    batch_draft: str | None = None
    batch_draft_evaluation: str | None = None
    critic_review: str | None = None
    unreachable_draft: str | None = None
    unreachable_evaluation: str | None = None
    unreachable_critic_review: str | None = None
    unreachable_decision: str | None = None
    human_unreachable_review: str | None = None
    research_briefing: str | None = None
    complete_batch_history: str | None = None
    final_report: str | None = None


class Observation(BaseModel):
    """A short, human-readable result of one action or user data return."""

    action: str
    status: Literal["success", "failed", "waiting", "rejected"]
    summary: str
    artifacts: list[str] = Field(default_factory=list)


class LastAction(BaseModel):
    """The latest action gives restart logic enough context without an event ledger."""

    name: str
    status: Literal["running", "success", "failed", "rejected"]


class Action(BaseModel):
    """One Director-selected tool invocation."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """A bounded tool response that can be recorded as an observation."""

    status: Literal["success", "not_found", "ambiguous", "network_error", "failed"]
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class ProvisionalGoalCandidate(BaseModel):
    """Persist the first measured target hit until an independent repeat confirms it."""

    source_round: int = Field(ge=1)
    sample_id: str
    recipe_id: str
    cie: tuple[float, float]
    distance: float = Field(ge=0)
    concentrations_mmol_ml: tuple[float, float, float, float, float]


class TaskState(BaseModel):
    """The complete mutable state for one recoverable research task."""

    task_id: str
    materials: list[str]
    target: tuple[float, float]
    max_rounds: int
    phase: Phase = Phase.PREPARING
    round: int = 0
    confirmed_materials: list[str] = Field(default_factory=list)
    artifacts: ArtifactRefs = Field(default_factory=ArtifactRefs)
    active_skills: list[str] = Field(default_factory=list)
    pending_measurement_path: str | None = None
    pending_data_origin: Literal["measured", "synthetic_dry_run"] | None = None
    injected_research_briefing: dict[str, Any] | None = None
    injected_complete_batch_history: dict[str, Any] | None = None
    injected_candidate_briefing: dict[str, Any] | None = None
    injected_critic_review: dict[str, Any] | None = None
    injected_unreachable_critic_review: dict[str, Any] | None = None
    provisional_goal_candidate: ProvisionalGoalCandidate | None = None
    history: list[Observation] = Field(default_factory=list)
    last_action: LastAction | None = None
    result: str | None = None

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: tuple[float, float]) -> tuple[float, float]:
        """CIE chromaticity coordinates must lie in the physical unit square."""
        if any(component < 0 or component > 1 for component in value):
            raise ValueError("CIE target coordinates must be between 0 and 1")
        return value

    @classmethod
    def new(
        cls,
        task_id: str,
        materials: list[str],
        target: tuple[float, float],
        max_rounds: int,
    ) -> "TaskState":
        """Create an untouched task from the user's scientific request."""
        return cls(
            task_id=task_id,
            materials=materials,
            target=target,
            max_rounds=max_rounds,
        )

    @property
    def all_materials_confirmed(self) -> bool:
        """Return true only when every user-supplied material has PubChem identity."""
        return bool(self.materials) and set(self.materials).issubset(self.confirmed_materials)
