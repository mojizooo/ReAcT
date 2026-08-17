"""Public compatibility facade for the split Experiment Director package."""

from .clients import DecisionClient, DirectorAPIError, OpenAIChatDecisionClient, ScriptedDecisionClient
from .contracts import DIRECTOR_INSTRUCTION, RECIPE_DISCRIMINATION_SCHEMA, TOOL_SCHEMAS
from .critic import OpenAIScientificCritic, ScriptedScientificCritic, ScientificCritic
from .guards import can_execute
from .runtime import DirectorRuntime

__all__ = [
    "DIRECTOR_INSTRUCTION",
    "RECIPE_DISCRIMINATION_SCHEMA",
    "TOOL_SCHEMAS",
    "DecisionClient",
    "DirectorAPIError",
    "OpenAIChatDecisionClient",
    "OpenAIScientificCritic",
    "ScriptedScientificCritic",
    "ScientificCritic",
    "ScriptedDecisionClient",
    "can_execute",
    "DirectorRuntime",
]
