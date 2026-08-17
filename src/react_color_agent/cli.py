"""Command-line entry points for creating, inspecting, and running one task."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from .director import (
    DirectorAPIError,
    DirectorRuntime,
    OpenAIChatDecisionClient,
    OpenAIScientificCritic,
)
from .evaluation import TraceRecorder, collect_paper_evidence
from .models import ArtifactRefs, Observation, Phase, TaskState
from .human_review import review_unreachable_application
from .skill_registry import SkillRegistry
from .storage import TaskStore
from tools.experiment_tools import (
    AnalyzeResultsTool,
    CheckGoalTool,
    CrossrefTool,
    DesignExploratoryFollowupBatchTool,
    DesignFollowupBatchTool,
    DesignInitialBatchTool,
    PubChemTool,
    SaveMaterialEvidenceTool,
)
from tools.skill_tools import ReadSkillTool
from tools.spectrum_tools import CalculateCieTool, IngestSpectraTool
from tools.spectral_tools import ExtractSpectralFeaturesTool
from tools.research_outcome_tools import ReviewDesignOutcomesTool
from tools.research_tools import (
    GetExperimentRecordTool,
    GetSpectrumDataTool,
    QueryResearchIndexTool,
    UpdateResearchDatasetTool,
    _build_research_index,
    _render_notebook,
)
from tools.analysis_tools import (
    CompileResearchAnalysisTool,
    DiagnoseDatasetTool,
    ScreenCompositionEffectsTool,
)
from tools.model_tools import CompareModelsTool, FitLocalResponseModelTool, WriteDesignDecisionTool
from tools.candidate_tools import GeneratePredictedCandidatesTool
from tools.batch_review_tools import FinalizeFollowupBatchTool, ProposeFollowupBatchTool
from tools.unreachable_tools import (
    ContinueAfterUnreachableReviewTool,
    ProposeUnreachableRequestTool,
    SubmitUnreachableApplicationTool,
)
from tools.research_context_tools import (
    GetAnalysisRecordTool,
    GetCompleteBatchHistoryTool,
    GetResearchBriefingTool,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a CLI command and return a process-compatible status code."""
    _load_dotenv(Path.cwd() / ".env")
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "create":
        return _create_task(arguments)
    if arguments.command == "show":
        return _show_task(arguments)
    if arguments.command == "run":
        return _run_task(arguments)
    if arguments.command == "collect-evidence":
        return _collect_evidence(arguments)
    if arguments.command == "branch-run":
        return _branch_run(arguments)
    if arguments.command == "review-unreachable":
        return _review_unreachable(arguments)
    parser.error("a command is required")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    """Define only the three commands required for a single local research task."""
    parser = argparse.ArgumentParser(prog="color-agent")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a colour experiment task")
    create.add_argument("--runs-root", type=Path, default=Path("runs"))
    create.add_argument("--task-id", required=True)
    create.add_argument(
        "--material",
        action="append",
        required=True,
        help="one whitespace-separated batch, or repeat this option for complete names with spaces",
    )
    create.add_argument("--target", required=True, help="CIE x,y; for example 0.345,0.345")
    create.add_argument("--max-rounds", type=int, default=3)

    show = commands.add_parser("show", help="show the current task state")
    show.add_argument("--run-dir", type=Path, required=True)

    run = commands.add_parser("run", help="run or resume a task through the OpenAI Director")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--data", type=Path, help="returned laboratory spectrum directory")
    run.add_argument(
        "--data-origin",
        choices=("measured", "synthetic_dry_run"),
        default="measured",
        help="scientific origin of --data; synthetic data remains ineligible for candidate design",
    )
    run.add_argument("--model", help="override BASE_MODEL from .env")
    run.add_argument("--no-trace", action="store_true", help="suppress live Director trace output")

    collect = commands.add_parser(
        "collect-evidence",
        help="derive paper evidence files from an existing run without executing the Agent",
    )
    collect.add_argument("--run-dir", type=Path, required=True)

    branch = commands.add_parser(
        "branch-run",
        help="create a new design-stage run from measured facts through one completed round",
    )
    branch.add_argument("--source-run", type=Path, required=True)
    branch.add_argument("--through-round", type=int, required=True)
    branch.add_argument("--run-dir", type=Path, required=True)

    review = commands.add_parser(
        "review-unreachable",
        help="approve or reject a submitted bounded unreachable application",
    )
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--decision", choices=("approve", "reject"), required=True)
    review.add_argument("--reason", required=True)
    return parser


def _create_task(arguments: argparse.Namespace) -> int:
    """Persist the original user request without attempting evidence retrieval."""
    try:
        x_text, y_text = arguments.target.split(",", maxsplit=1)
        target = (float(x_text), float(y_text))
    except ValueError:
        print("--target must use the form x,y", file=sys.stderr)
        return 2
    if arguments.max_rounds < 1:
        print("--max-rounds must be at least 1", file=sys.stderr)
        return 2
    materials = _parse_material_arguments(arguments.material)
    if not materials:
        print("--material must contain at least one non-empty material name", file=sys.stderr)
        return 2
    store = TaskStore.create(
        arguments.runs_root / arguments.task_id,
        TaskState.new(arguments.task_id, materials, target, arguments.max_rounds),
    )
    print(store.state_path)
    return 0


def _parse_material_arguments(values: Sequence[str]) -> list[str]:
    """Accept one whitespace-separated list while preserving repeated complete names."""
    if len(values) == 1:
        # A single shell argument is the convenient batch form for Chinese material lists.
        return values[0].split()
    return [value.strip() for value in values if value.strip()]


def _load_dotenv(path: Path) -> None:
    """Load simple project-local environment entries without executing shell content."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isidentifier():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", maxsplit=1)[0].rstrip()
        # Shell-provided configuration stays higher priority than a local .env file.
        os.environ.setdefault(key, value)
    if "OPENAI_BASE_URL" not in os.environ and os.environ.get("BASE_URL"):
        # The OpenAI SDK convention differs from the lightweight research .env convention.
        os.environ["OPENAI_BASE_URL"] = os.environ["BASE_URL"]


def _show_task(arguments: argparse.Namespace) -> int:
    """Print the canonical file state so users can inspect a paused task directly."""
    state = TaskStore.open(arguments.run_dir).load()
    print(state.model_dump_json(indent=2))
    return 0


def _run_task(arguments: argparse.Namespace) -> int:
    """Start a real Director only when an explicit OpenAI credential is available."""
    store = TaskStore.open(arguments.run_dir)
    current_state = store.load()
    if current_state.phase is Phase.AWAITING_HUMAN_REVIEW:
        if arguments.data:
            print(
                "the task is awaiting human review; --data cannot bypass approval or rejection",
                file=sys.stderr,
            )
            return 2
        print(current_state.model_dump_json(indent=2))
        return 0
    if arguments.data:
        _record_returned_data(store, arguments.data, arguments.data_origin)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for color-agent run", file=sys.stderr)
        return 2
    try:
        model = arguments.model or os.environ.get("BASE_MODEL", "gpt-4.1-mini")
        skill_registry = SkillRegistry.from_directory(_local_skill_root())
        client = OpenAIChatDecisionClient(model=model, skill_registry=skill_registry)
        critic_model = os.environ.get("CRITIC_MODEL") or model
        critic = OpenAIScientificCritic(model=critic_model)
        runtime = DirectorRuntime(
            store,
            _default_tools(store, skill_registry),
            client,
            trace=None if arguments.no_trace else _print_trace,
            recorder=TraceRecorder(store.run_dir),
            critic=critic,
        )
        state = runtime.run_until_pause(resume_waiting=bool(arguments.data))
    except (DirectorAPIError, RuntimeError) as error:
        print(f"Director API request failed: {error}", file=sys.stderr)
        return 2
    print(state.model_dump_json(indent=2))
    return 0


def _collect_evidence(arguments: argparse.Namespace) -> int:
    """Run the read-only paper evidence projection as a separate CLI operation."""
    try:
        files = collect_paper_evidence(arguments.run_dir)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"evidence collection failed: {error}", file=sys.stderr)
        return 2
    for name, path in files.items():
        print(f"{name}: {arguments.run_dir / path}")
    return 0


def _branch_run(arguments: argparse.Namespace) -> int:
    """Create an independent design-stage run from a bounded prefix of measured facts."""
    source_store = TaskStore.open(arguments.source_run)
    source_state = source_store.load()
    through_round = int(arguments.through_round)
    if through_round < 1 or through_round > source_state.round:
        print("--through-round must select an existing completed round", file=sys.stderr)
        return 2
    if arguments.run_dir.exists():
        print(f"branch run directory already exists: {arguments.run_dir}", file=sys.stderr)
        return 2
    if not source_state.artifacts.research_dataset:
        print("source run has no research dataset", file=sys.stderr)
        return 2
    source_dataset_path = source_store.artifact_path(source_state.artifacts.research_dataset)
    dataset = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    selected_batches = [
        batch for batch in dataset.get("batches", []) if int(batch.get("round", 0)) <= through_round
    ]
    if not selected_batches or max(int(batch["round"]) for batch in selected_batches) != through_round:
        print("source research dataset does not contain the requested completed round", file=sys.stderr)
        return 2

    new_task_id = arguments.run_dir.name
    dataset["task"]["task_id"] = new_task_id
    dataset["batches"] = selected_batches
    current_batch = next(batch for batch in selected_batches if int(batch["round"]) == through_round)
    experiment_plan = str(current_batch["plan"]["xlsx_artifact"])
    observation_artifacts = current_batch["observation_artifacts"]
    new_state = TaskState(
        task_id=new_task_id,
        materials=source_state.materials,
        target=source_state.target,
        max_rounds=source_state.max_rounds,
        phase=Phase.DESIGNING,
        round=through_round,
        confirmed_materials=source_state.confirmed_materials,
        artifacts=ArtifactRefs(
            material_evidence=source_state.artifacts.material_evidence,
            experiment_plan=experiment_plan,
            spectra_manifest=str(observation_artifacts["spectra_manifest"]),
            measurement_result=str(observation_artifacts["measurement_result"]),
            analysis_result=str(observation_artifacts["analysis_result"]),
            research_dataset="artifacts/research_dataset.json",
            research_index="artifacts/research_index.json",
        ),
        active_skills=source_state.active_skills,
        history=[
            Observation(
                action="branch_run",
                status="success",
                summary=(
                    f"Derived this run from {arguments.source_run} using measured facts through "
                    f"round {through_round}; later source-run decisions were intentionally excluded."
                ),
            )
        ],
    )
    destination_store = TaskStore.create(arguments.run_dir, new_state)

    # Copy immutable factual inputs and activated Skill snapshots without touching the source run.
    for relative_directory in ("artifacts/evidence", "artifacts/skills"):
        source = source_store.artifact_path(relative_directory)
        if source.is_dir():
            shutil.copytree(source, destination_store.artifact_path(relative_directory), dirs_exist_ok=True)
    plan_root = source_store.artifact_path("artifacts/experiment_plans")
    destination_plan_root = destination_store.artifact_path("artifacts/experiment_plans")
    destination_plan_root.mkdir(parents=True, exist_ok=True)
    for batch_number in range(1, through_round + 1):
        for source in plan_root.glob(f"batch_{batch_number:03}.*"):
            shutil.copy2(source, destination_plan_root / source.name)
        design = plan_root / f"batch_{batch_number:03}_design.json"
        if design.is_file():
            shutil.copy2(design, destination_plan_root / design.name)
    for round_number in range(1, through_round + 1):
        source_round = source_store.artifact_path(f"artifacts/round-{round_number}")
        destination_round = destination_store.artifact_path(f"artifacts/round-{round_number}")
        destination_round.mkdir(parents=True, exist_ok=True)
        for name in (
            "raw",
            "spectra_manifest.json",
            "measurement_result.json",
            "analysis_result.json",
            "design_outcome_review.json",
            "SYNTHETIC_DRY_RUN.md",
        ):
            source = source_round / name
            destination = destination_round / name
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            elif source.is_file():
                shutil.copy2(source, destination)

    dataset_ref = destination_store.write_artifact_json("artifacts/research_dataset.json", dataset)
    index = _build_research_index(dataset, dataset_ref)
    destination_store.write_artifact_json("artifacts/research_index.json", index)
    destination_store.write_artifact_text(
        "artifacts/research_notebook.md", _render_notebook(dataset, index)
    )
    print(destination_store.run_dir)
    return 0


def _print_trace(message: str) -> None:
    """Keep live operational trace on stderr so stdout remains a machine-readable state JSON."""
    print(message, file=sys.stderr, flush=True)


def _local_skill_root() -> Path:
    """Resolve packaged local Skill resources independently of the caller's directory."""
    return Path(__file__).resolve().parents[1] / "skills"


def _record_returned_data(store: TaskStore, path: Path, data_origin: str = "measured") -> None:
    """Record a user-returned spectrum directory before the Director processes it."""
    if not path.is_dir():
        raise FileNotFoundError(f"measurement directory does not exist: {path}")
    store.update(
        lambda state: state.model_copy(
            update={
                "pending_measurement_path": str(path),
                "pending_data_origin": data_origin,
                "history": [
                    *state.history,
                    Observation(
                        action="user_data_return",
                        status="waiting",
                        summary=f"User returned {data_origin} measurement data at {path}.",
                    ),
                ],
            }
        )
    )


def _default_tools(store: TaskStore, skill_registry: SkillRegistry) -> dict[str, object]:
    """Register the intentionally small first-version tool set for the real Director."""
    return {
        "read_skill": ReadSkillTool(store, skill_registry),
        "query_pubchem": PubChemTool(),
        "search_crossref": CrossrefTool(),
        "save_material_evidence": SaveMaterialEvidenceTool(store),
        "design_initial_batch": DesignInitialBatchTool(store),
        "design_followup_batch": DesignFollowupBatchTool(store),
        "design_exploratory_followup_batch": DesignExploratoryFollowupBatchTool(store),
        "ingest_spectra": IngestSpectraTool(store),
        "calculate_cie": CalculateCieTool(store),
        "analyze_results": AnalyzeResultsTool(store),
        "update_research_dataset": UpdateResearchDatasetTool(store),
        "query_research_index": QueryResearchIndexTool(store),
        "get_experiment_record": GetExperimentRecordTool(store),
        "get_spectrum_data": GetSpectrumDataTool(store),
        "extract_spectral_features": ExtractSpectralFeaturesTool(store),
        "review_design_outcomes": ReviewDesignOutcomesTool(store),
        "diagnose_dataset": DiagnoseDatasetTool(store),
        "screen_composition_effects": ScreenCompositionEffectsTool(store),
        "compile_research_analysis": CompileResearchAnalysisTool(store),
        "fit_local_response_model": FitLocalResponseModelTool(store),
        "compare_models": CompareModelsTool(store),
        "write_design_decision": WriteDesignDecisionTool(store),
        "generate_predicted_candidates": GeneratePredictedCandidatesTool(store),
        "propose_followup_batch": ProposeFollowupBatchTool(store),
        "finalize_followup_batch": FinalizeFollowupBatchTool(store),
        "propose_unreachable_request": ProposeUnreachableRequestTool(store),
        "continue_after_unreachable_review": ContinueAfterUnreachableReviewTool(store),
        "submit_unreachable_application": SubmitUnreachableApplicationTool(store),
        "get_research_briefing": GetResearchBriefingTool(store),
        "get_complete_batch_history": GetCompleteBatchHistoryTool(store),
        "get_analysis_record": GetAnalysisRecordTool(store),
        "check_goal": CheckGoalTool(store),
    }


def _review_unreachable(arguments: argparse.Namespace) -> int:
    """Apply the only state transition that can approve an early unreachable stop."""
    try:
        state = review_unreachable_application(
            arguments.run_dir,
            arguments.decision,
            arguments.reason,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"unreachable review failed: {error}", file=sys.stderr)
        return 2
    print(state.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    # Keep module execution equivalent to the installed console script.
    raise SystemExit(main())
