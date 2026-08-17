"""Agent-callable tools for activating repository-local scientific Skills."""

from __future__ import annotations

from typing import Any

from react_color_agent.models import ToolResult
from react_color_agent.skill_registry import SkillRegistry
from react_color_agent.storage import TaskStore


class ReadSkillTool:
    """Activate one registered local Skill and preserve a run-local audit artifact."""

    name = "read_skill"

    def __init__(self, store: TaskStore, registry: SkillRegistry) -> None:
        """Bind Skill audit records to one task and allow only registered Skill names."""
        self.store = store
        self.registry = registry

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Persist Skill identity; the Director injects its body on the next model decision."""
        skill = self.registry.get(str(arguments["name"]))
        # Store only stable provenance; full instructions remain managed by the registry.
        artifact = self.store.write_artifact_json(
            f"artifacts/skills/{skill.name}.json",
            {
                "name": skill.name,
                "description": skill.description,
                "source_path": str(skill.path),
            },
        )
        return ToolResult(
            status="success",
            summary=f"Activated local Skill {skill.name}.",
            data={"skill_name": skill.name},
            artifacts=[artifact],
        )
