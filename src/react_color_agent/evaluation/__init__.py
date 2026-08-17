"""Read-only runtime and paper-evidence collection helpers.

The evaluation package is deliberately outside the Agent tool registry.  Its
outputs describe a run after or alongside execution, but never decide it.
"""

from .paper_evidence import collect_paper_evidence
from .protocol_fingerprint import build_protocol_fingerprint
from .trace_recorder import TraceRecorder

__all__ = ["TraceRecorder", "build_protocol_fingerprint", "collect_paper_evidence"]
