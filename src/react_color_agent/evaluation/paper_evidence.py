"""Post-hoc paper evidence export for one completed or paused run.

This module only reads run files and writes derived reports.  It is not a
registered Agent tool and never changes ``state.json`` or any scientific fact.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .trace_recorder import TraceRecorder


EVIDENCE_DIR = Path("artifacts/paper_evidence")


def collect_paper_evidence(run_dir: Path) -> dict[str, str]:
    """Create the paper-evidence projection and return its run-relative files."""
    run_dir = run_dir.resolve()
    if not (run_dir / "state.json").is_file():
        raise FileNotFoundError(f"task state does not exist: {run_dir / 'state.json'}")
    state = _read_json(run_dir / "state.json", default={})
    dataset_path = run_dir / "artifacts" / "research_dataset.json"
    dataset = _read_json(dataset_path, default=None)
    trace = TraceRecorder(run_dir).read()
    output = run_dir / EVIDENCE_DIR
    output.mkdir(parents=True, exist_ok=True)

    manifest = _run_manifest(run_dir, state, dataset, trace)
    metrics = _run_metrics(run_dir, state, dataset, trace)
    batches = _batch_rows(run_dir, dataset)
    samples = _sample_rows(dataset)
    predictions = _prediction_rows(run_dir, dataset)
    decision_trace = _decision_trace(trace, state)
    tool_trajectory = _tool_trajectory(trace)

    files: dict[str, Any] = {
        "run_manifest.json": manifest,
        "run_metrics.json": metrics,
        "batch_metrics.csv": batches,
        "sample_metrics.csv": samples,
        "prediction_vs_measurement.csv": predictions,
        "decision_trace.json": decision_trace,
        "tool_trajectory.json": tool_trajectory,
    }
    for name, payload in files.items():
        path = output / name
        if name.endswith(".json"):
            _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        else:
            _write_csv(path, payload)
    summary_path = output / "paper_summary.md"
    _write_text(summary_path, _render_summary(manifest, metrics, batches, predictions))
    result = {name: (output / name).relative_to(run_dir).as_posix() for name in files}
    result["paper_summary.md"] = summary_path.relative_to(run_dir).as_posix()
    return result


def _run_manifest(run_dir: Path, state: dict[str, Any], dataset: dict[str, Any] | None, trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Capture run identity and reproducibility references without credentials."""
    model = next((item.get("model") for item in reversed(trace) if item.get("model")), None)
    protocol_sessions = _protocol_sessions(trace)
    latest_protocol = protocol_sessions[-1]["protocol"] if protocol_sessions else None
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": state.get("task_id"),
        "materials": state.get("materials", []),
        "target_cie": state.get("target"),
        "max_rounds": state.get("max_rounds"),
        "final_phase": state.get("phase"),
        "final_round": state.get("round", 0),
        "final_result": state.get("result"),
        "model": model,
        "dataset_present": dataset is not None,
        "dataset_sha256": _sha256(dataset) if dataset is not None else None,
        "runtime_trace": "evaluation/runtime_trace.jsonl" if trace else None,
        "run_path_name": run_dir.name,
        "protocol_sessions": protocol_sessions,
        "source_hashes": (
            latest_protocol.get("source_hashes", {})
            if isinstance(latest_protocol, dict)
            else _source_hashes(run_dir, state)
        ),
        "source_hashes_basis": (
            "latest_runtime_started_event" if protocol_sessions else "current_checkout_fallback"
        ),
    }


def _run_metrics(run_dir: Path, state: dict[str, Any], dataset: dict[str, Any] | None, trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate operational and scientific metrics while preserving origin labels."""
    batches = list(dataset.get("batches", [])) if dataset else []
    measured_batches = [batch for batch in batches if batch.get("origin", {}).get("scientific_eligible")]
    samples = [observation for batch in measured_batches for observation in batch.get("observations", [])]
    distances = [_number(item.get("evaluation", {}).get("distance")) for item in samples]
    distances = [value for value in distances if value is not None]
    events = _event_counts(trace)
    protocol_sessions = _protocol_sessions(trace)
    first_within_tolerance = next((batch.get("round") for batch in measured_batches if any(
        bool(item.get("evaluation", {}).get("within_tolerance")) for item in batch.get("observations", [])
    )), None)
    final_report_artifact = state.get("artifacts", {}).get("final_report")
    goal_report = _read_json(run_dir / str(final_report_artifact), default={}) if final_report_artifact else {}
    provisional_candidate = goal_report.get("provisional_candidate", {})
    confirmation_samples = goal_report.get("confirmation_samples", [])
    return {
        "schema_version": 1,
        "phase": state.get("phase"),
        "target_met": state.get("phase") == "FINISHED",
        "best_measured_target_distance": min(distances) if distances else None,
        "first_within_tolerance_round": first_within_tolerance,
        "goal_verification_status": goal_report.get("verification_status"),
        "provisional_hit_round": provisional_candidate.get("source_round"),
        "confirmation_round": (
            goal_report.get("round")
            if goal_report.get("verification_status") in {"confirmed", "confirmation_failed"}
            else None
        ),
        "confirmation_sample_count": len(confirmation_samples),
        "measured_batch_count": len(measured_batches),
        "all_recorded_batch_count": len(batches),
        "measured_sample_count": len(samples),
        "all_recorded_sample_count": sum(len(batch.get("observations", [])) for batch in batches),
        "max_rounds": state.get("max_rounds"),
        "snapshot_count": len(list((run_dir / "snapshots").glob("*.json"))),
        "tool_event_counts": events,
        "tool_call_count": sum(1 for item in trace if item.get("event") == "tool_finished"),
        "tool_call_sequence": [
            item.get("action") for item in trace
            if item.get("event") == "tool_finished" and item.get("action")
        ],
        "action_rejection_count": events.get("action_rejected", 0),
        "tool_failure_count": events.get("tool_finished_failed", 0),
        "tool_elapsed_ms": sum(_number(item.get("elapsed_ms")) or 0 for item in trace if item.get("event") == "tool_finished"),
        "decision_elapsed_ms": sum(_number(item.get("elapsed_ms")) or 0 for item in trace if item.get("event") in {"decision_finished", "decision_failed"}),
        "critic_elapsed_ms": sum(_number(item.get("elapsed_ms")) or 0 for item in trace if item.get("event") in {"critic_finished", "critic_failed"}),
        "critic_review_count": events.get("critic_finished", 0),
        "agent_elapsed_ms": sum(_number(item.get("elapsed_ms")) or 0 for item in trace if item.get("event") in {"tool_finished", "decision_finished", "decision_failed", "critic_finished", "critic_failed"}),
        "token_usage": _token_usage_metrics(trace),
        "runtime_session_count": events.get("runtime_started", 0),
        "protocol_record_count": len(protocol_sessions),
        "distinct_protocol_count": len(
            {item["protocol_sha256"] for item in protocol_sessions if item.get("protocol_sha256")}
        ),
        "human_wait_count": events.get("task_paused", 0),
        "strategy_sequence": _strategy_sequence(run_dir, dataset),
        "scientific_evidence_policy": "Only measured batches contribute to scientific metrics; synthetic dry runs remain lineage records.",
        "unreachable_application": _unreachable_metrics(run_dir, state, trace),
    }


def _unreachable_metrics(
    run_dir: Path,
    state: dict[str, Any],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project the bounded stop-request path for paper analysis without changing state."""
    refs = state.get("artifacts", {})
    draft = _read_json(run_dir / str(refs.get("unreachable_draft", "")), default={})
    evaluation = _read_json(run_dir / str(refs.get("unreachable_evaluation", "")), default={})
    critic = _read_json(run_dir / str(refs.get("unreachable_critic_review", "")), default={})
    decision = _read_json(run_dir / str(refs.get("unreachable_decision", "")), default={})
    human = _read_json(run_dir / str(refs.get("human_unreachable_review", "")), default={})
    events = [
        item
        for item in trace
        if str(item.get("event", "")).startswith("unreachable_critic_")
        or item.get("event") == "human_review_unreachable"
    ]
    return {
        "active": bool(draft or critic or decision or human),
        "draft_artifact": refs.get("unreachable_draft"),
        "evidence_evaluation_artifact": refs.get("unreachable_evaluation"),
        "critic_artifact": refs.get("unreachable_critic_review"),
        "director_decision_artifact": refs.get("unreachable_decision"),
        "human_review_artifact": refs.get("human_unreachable_review"),
        "draft_claim": draft.get("claim"),
        "measured_evidence_sample_count": evaluation.get("result", {}).get("cited_sample_count"),
        "critic_recommendation": critic.get("recommendation"),
        "critic_finding_count": len(critic.get("findings", [])) if isinstance(critic.get("findings"), list) else 0,
        "director_decision_kind": decision.get("decision_kind"),
        "human_decision": human.get("decision"),
        "human_reason": human.get("reason"),
        "trace_events": events,
        "scientific_boundary": "This path records a bounded, human-reviewed decision; it is not proof of universal impossibility.",
    }


def _batch_rows(run_dir: Path, dataset: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten one row per batch with measured-only quality metrics and lineage."""
    rows: list[dict[str, Any]] = []
    for batch in (dataset or {}).get("batches", []):
        observations = batch.get("observations", [])
        distances = [_number(item.get("evaluation", {}).get("distance")) for item in observations]
        distances = [value for value in distances if value is not None]
        eligible = bool(batch.get("origin", {}).get("scientific_eligible"))
        design = _read_json(run_dir / str(batch.get("plan", {}).get("design_artifact", "")), default={})
        decision = _read_json(
            run_dir / f"artifacts/round-{int(batch.get('round', 0))}/design_decision.json",
            default={},
        )
        rows.append({
            "batch_id": batch.get("batch_id"),
            "round": batch.get("round"),
            "origin": batch.get("origin", {}).get("kind"),
            "scientific_eligible": eligible,
            "sample_count": len(observations),
            "best_target_distance": min(distances) if distances else None,
            "median_target_distance": _median(distances),
            "within_tolerance_count": sum(bool(item.get("evaluation", {}).get("within_tolerance")) for item in observations),
            "plan_artifact": batch.get("plan", {}).get("xlsx_artifact"),
            "design_artifact": batch.get("plan", {}).get("design_artifact"),
            "recipe_source": design.get("design_context", {}).get("recipe_source"),
            "scientific_rationale": design.get("design_context", {}).get("scientific_rationale"),
            "strategy": decision.get("strategy"),
            "selected_method": decision.get("selected_method"),
        })
    return rows


def _sample_rows(dataset: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten recipe, CIE, QC, and raw-file hashes into a paper-friendly table."""
    rows: list[dict[str, Any]] = []
    for batch in (dataset or {}).get("batches", []):
        origin = batch.get("origin", {})
        for item in batch.get("observations", []):
            identity = item.get("identity", {})
            measurement = item.get("measurement", {})
            evaluation = item.get("evaluation", {})
            evidence = item.get("evidence", {})
            rows.append({
                "batch_id": batch.get("batch_id"),
                "round": batch.get("round"),
                "sample_id": identity.get("sample_id"),
                "recipe_id": identity.get("recipe_id"),
                "design_role": identity.get("design_role"),
                "origin": origin.get("kind"),
                "scientific_eligible": bool(origin.get("scientific_eligible")),
                "concentrations_mmol_ml": json.dumps(item.get("recipe", {}).get("concentrations_mmol_ml", []), ensure_ascii=False),
                "cie_x": _at(measurement.get("cie"), 0),
                "cie_y": _at(measurement.get("cie"), 1),
                "target_x": _at(evaluation.get("target_cie"), 0),
                "target_y": _at(evaluation.get("target_cie"), 1),
                "target_distance": evaluation.get("distance"),
                "within_tolerance": evaluation.get("within_tolerance"),
                "emission_peak_nm": item.get("qc", {}).get("emission_peak_nm"),
                "emission_sha256": evidence.get("emission_sha256"),
                "absorption_sha256": evidence.get("absorption_sha256"),
                "emission_raw_path": evidence.get("emission_raw_path"),
                "absorption_raw_path": evidence.get("absorption_raw_path"),
                "design_intent": item.get("design_intent", {}).get("purpose"),
            })
    return rows


def _prediction_rows(run_dir: Path, dataset: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Join candidate predictions to later measured recipes when the lineage exists."""
    if dataset is None:
        return []
    measured = {
        tuple(round(float(value), 12) for value in item.get("recipe", {}).get("concentrations_mmol_ml", [])): item
        for batch in dataset.get("batches", [])
        if batch.get("origin", {}).get("scientific_eligible")
        for item in batch.get("observations", [])
    }
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "artifacts").glob("round-*/predicted_candidates.json")):
        payload = _read_json(path, default={})
        if not payload or payload.get("status") != "PREDICTED_CANDIDATES_NOT_MEASURED":
            continue
        for candidate in payload.get("candidates", []):
            concentrations = tuple(round(float(value), 12) for value in candidate.get("concentrations_mmol_ml", []))
            observation = measured.get(concentrations)
            measured_cie = observation.get("measurement", {}).get("cie") if observation else None
            predicted_cie = candidate.get("predicted_cie")
            rows.append({
                "prediction_round": payload.get("round"),
                "candidate_id": candidate.get("candidate_id"),
                "selection_eligible": candidate.get("selection_eligible"),
                "measured_sample_id": observation.get("identity", {}).get("sample_id") if observation else None,
                "concentrations_mmol_ml": json.dumps(candidate.get("concentrations_mmol_ml", []), ensure_ascii=False),
                "predicted_x": _at(predicted_cie, 0),
                "predicted_y": _at(predicted_cie, 1),
                "measured_x": _at(measured_cie, 0),
                "measured_y": _at(measured_cie, 1),
                "cie_error": _cie_error(predicted_cie, measured_cie),
                "predicted_target_distance": candidate.get("predicted_target_distance"),
                "measured_target_distance": observation.get("evaluation", {}).get("distance") if observation else None,
                "coverage_classification": candidate.get("coverage", {}).get("classification"),
                "selected_method": payload.get("selected_method"),
            })
    return rows


def _decision_trace(trace: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    """Expose the ordered runtime events plus durable action outcomes."""
    return {
        "schema_version": 1,
        "events": trace,
        "state_history": state.get("history", []),
        "interpretation_notice": "Events describe execution order; state history describes persisted tool outcomes. Neither is a causal scientific claim.",
    }


def _tool_trajectory(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair tool start/finish events into an ordered, analysis-friendly trajectory."""
    pending: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for event in trace:
        event_name = event.get("event")
        if event_name == "tool_started":
            pending.append(event)
            continue
        if event_name != "tool_finished":
            continue
        action = event.get("action")
        start_index = next(
            (index for index in range(len(pending) - 1, -1, -1) if pending[index].get("action") == action),
            None,
        )
        started = pending.pop(start_index) if start_index is not None else {}
        calls.append({
            "step": len(calls) + 1,
            "action": action,
            "phase": started.get("phase", event.get("phase")),
            "round": started.get("round", event.get("round")),
            "arguments": started.get("arguments", {}),
            "status": event.get("status"),
            "summary": event.get("summary"),
            "elapsed_ms": event.get("elapsed_ms"),
            "artifacts": event.get("artifacts", []),
        })
    return {
        "schema_version": 1,
        "call_count": len(calls),
        "calls": calls,
        "unmatched_tool_start_count": len(pending),
        "interpretation_notice": "This is an execution trace, not a scientific causal explanation or a policy recommendation.",
    }


def _render_summary(manifest: dict[str, Any], metrics: dict[str, Any], batches: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> str:
    """Render a concise handoff for paper tables and figure preparation."""
    usage = metrics["token_usage"]
    token_text = (
        f"{usage['total_tokens']} tokens across {usage['request_count']} decisions"
        if usage["available"]
        else "provider usage unavailable"
    )
    lines = [
        "# Paper Evidence Summary",
        "",
        f"- Task: `{manifest.get('task_id')}`",
        f"- Target CIE: `{manifest.get('target_cie')}`",
        f"- Final phase: `{manifest.get('final_phase')}`",
        f"- Scientific batches: `{metrics['measured_batch_count']}`; measured samples: `{metrics['measured_sample_count']}`",
        f"- Best measured target distance: `{_fmt(metrics.get('best_measured_target_distance'))}`",
        f"- Agent tool time (recorded): `{metrics['agent_elapsed_ms']} ms`",
        f"- Agent token usage (provider-reported): `{token_text}`",
        f"- Goal verification: `{metrics.get('goal_verification_status')}`; provisional round: `{metrics.get('provisional_hit_round')}`; confirmation round: `{metrics.get('confirmation_round')}`",
        f"- Runtime sessions: `{metrics['runtime_session_count']}`; frozen protocol records: `{metrics['protocol_record_count']}`; distinct protocols: `{metrics['distinct_protocol_count']}`",
        "",
        "## Batch table",
        "",
        "| Batch | Origin | Samples | Best distance | Eligible |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in batches:
        lines.append(f"| {row['batch_id']} | {row['origin']} | {row['sample_count']} | {_fmt(row['best_target_distance'])} | {row['scientific_eligible']} |")
    lines.extend([
        "",
        f"Prediction/measurement joins: `{sum(row['measured_sample_id'] is not None for row in predictions)}` of `{len(predictions)}` candidate records.",
        "",
        "> Synthetic dry-run batches are retained for workflow lineage but are excluded from scientific metrics and model claims.",
        "> This report is a derived evidence projection, not an additional source of experimental facts.",
        "",
    ])
    return "\n".join(lines)


def _event_counts(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count event types and failed tool completions for operational reporting."""
    counts: dict[str, int] = {}
    for event in events:
        name = str(event.get("event", "unknown"))
        counts[name] = counts.get(name, 0) + 1
        if name == "tool_finished" and event.get("status") == "failed":
            counts["tool_finished_failed"] = counts.get("tool_finished_failed", 0) + 1
    return counts


def _token_usage_metrics(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate provider-reported usage while never estimating missing counts."""
    usage_records = [
        event.get("token_usage")
        for event in events
        if event.get("event") in {"decision_finished", "critic_finished"}
        and isinstance(event.get("token_usage"), dict)
    ]
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens")
    totals: dict[str, int | None] = {}
    for field in fields:
        values = [record.get(field) for record in usage_records if isinstance(record.get(field), (int, float))]
        totals[field] = int(sum(values)) if values else None
    return {
        "available": bool(usage_records),
        "request_count": len(usage_records),
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["total_tokens"],
        "cached_prompt_tokens": totals["cached_prompt_tokens"],
        "estimated": False,
        "notice": "Counts are copied from provider response usage; null means the provider did not return usage.",
    }


def _strategy_sequence(run_dir: Path, dataset: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return strategy labels from persisted decision artifacts when available."""
    sequence: list[dict[str, Any]] = []
    for batch in (dataset or {}).get("batches", []):
        decision = _read_json(
            run_dir / f"artifacts/round-{int(batch.get('round', 0))}/design_decision.json",
            default={},
        )
        if decision:
            sequence.append({
                "round": batch.get("round"),
                "strategy": decision.get("strategy"),
                "selected_method": decision.get("selected_method"),
            })
    return sequence


def _source_hashes(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Hash prompt/Skill/source files when the repository is available locally."""
    repository_root = Path(__file__).resolve().parents[3]
    source_files = sorted(
        [*repository_root.glob("src/**/*.py"), *repository_root.glob("src/skills/**/*.md")]
    )
    implementation_digest = hashlib.sha256()
    for path in source_files:
        try:
            implementation_digest.update(path.relative_to(repository_root).as_posix().encode("utf-8"))
            implementation_digest.update(path.read_bytes())
        except OSError:
            continue
    prompt_path = repository_root / "src/react_color_agent/director/contracts.py"
    skills: dict[str, str] = {}
    for name in state.get("active_skills", []):
        matches = []
        for candidate in (repository_root / "src/skills").glob("**/SKILL.md"):
            try:
                if f"name: {name}" in candidate.read_text(encoding="utf-8").split("\n---", 1)[0]:
                    matches.append(candidate)
            except OSError:
                continue
        if matches:
            digest = _file_sha256(matches[0])
            if digest:
                skills[str(name)] = digest
    return {
        "implementation_sha256": implementation_digest.hexdigest() if source_files else None,
        "prompt_sha256": _file_sha256(prompt_path),
        "active_skill_sha256": skills,
    }


def _protocol_sessions(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the immutable per-session protocol records from runtime trace events."""
    sessions = []
    for event in trace:
        if event.get("event") != "runtime_started" or not isinstance(event.get("protocol"), dict):
            continue
        sessions.append(
            {
                "timestamp": event.get("timestamp"),
                "protocol_sha256": event.get("protocol_sha256"),
                "protocol": event["protocol"],
            }
        )
    return sessions


def _file_sha256(path: Path) -> str | None:
    """Hash one optional source file for reproducibility metadata."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable UTF-8 CSV even when a run has no rows."""
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["record_count"]
        rows = [{"record_count": 0}]
    path.write_text("", encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    """Write derived files using a simple replace-after-write operation."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    """Read optional run artifacts without making partial runs impossible to inspect."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _sha256(payload: Any) -> str:
    """Hash canonical JSON for manifest lineage."""
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _number(value: Any) -> float | None:
    """Convert finite numeric values while keeping absent values as null."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _at(value: Any, index: int) -> float | None:
    """Read one coordinate from a possibly absent CIE vector."""
    if not isinstance(value, (list, tuple)) or len(value) <= index:
        return None
    return _number(value[index])


def _median(values: list[float]) -> float | None:
    """Calculate a tiny dependency-free median for batch summaries."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _cie_error(predicted: Any, measured: Any) -> float | None:
    """Return Euclidean CIE error only when both vectors are complete."""
    px, py = _at(predicted, 0), _at(predicted, 1)
    mx, my = _at(measured, 0), _at(measured, 1)
    if None in {px, py, mx, my}:
        return None
    return ((px - mx) ** 2 + (py - my) ** 2) ** 0.5


def _fmt(value: Any) -> str:
    """Format nullable distances for Markdown without inventing zero values."""
    number = _number(value)
    return "n/a" if number is None else f"{number:.6f}"
