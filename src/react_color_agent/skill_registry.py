"""Discovery and prompt-context management for repository-local scientific Skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalSkill:
    """A human-authored local Skill and its small front-matter metadata."""

    name: str
    description: str
    path: Path
    body: str


class SkillRegistry:
    """Discover explicit local Skill contracts and render activated instruction context."""

    def __init__(self, skills: dict[str, LocalSkill]) -> None:
        """Keep a stable name-to-Skill mapping for one Director process."""
        self.skills = skills

    @classmethod
    def from_directory(cls, root: Path) -> "SkillRegistry":
        """Load only `SKILL.md` files beneath the supplied repository resource root."""
        skills: dict[str, LocalSkill] = {}
        if not root.is_dir():
            return cls(skills)
        for path in sorted(root.rglob("SKILL.md")):
            text = path.read_text(encoding="utf-8")
            metadata, body = _split_front_matter(text)
            name = metadata.get("name")
            if not name:
                continue
            if name in skills:
                raise ValueError(f"duplicate local Skill name: {name}")
            skills[name] = LocalSkill(
                name=name,
                description=metadata.get("description", ""),
                path=path,
                body=body,
            )
        return cls(skills)

    def get(self, name: str) -> LocalSkill:
        """Resolve one registered Skill or report the available bounded capability names."""
        try:
            return self.skills[name]
        except KeyError as error:
            available = ", ".join(sorted(self.skills)) or "none"
            raise ValueError(f"unknown Skill {name!r}; available Skills: {available}") from error

    def prompt_context(self, active_names: list[str]) -> str:
        """Render only explicitly activated Skills into the next model decision context."""
        sections = []
        for name in active_names:
            skill = self.get(name)
            sections.append(f"# Active local Skill: {skill.name}\n\n{skill.body.strip()}")
        return "\n\n---\n\n".join(sections)


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse the small YAML-like metadata shape used by the migrated Skill files."""
    if not text.startswith("---\n"):
        return {}, text
    closing_index = text.find("\n---\n", 4)
    if closing_index < 0:
        return {}, text
    metadata: dict[str, str] = {}
    for line in text[4:closing_index].splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip('"')
    return metadata, text[closing_index + 5 :]
