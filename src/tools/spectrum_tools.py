"""Deterministic CPWL spectrum intake and CIE 1931 chromaticity tools."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore

EMISSION_COMMON_HEADER_PREFIX = (
    "Scan Mode: 发射扫描",
    "激发波长: 350 nm",
)
EMISSION_RANGE_HEADER = "发射波长范围: 360 - 760 nm"
LEGACY_EMISSION_RANGE_HEADER = "发射波长范围: 400 - 700 nm"
EMISSION_COLUMN_HEADER = "波长(nm)\t荧光强度"
EMISSION_START_NM = 360.0
EMISSION_END_NM = 760.0
DEFAULT_EMISSION_STEP_NM = 0.2
LEGACY_EMISSION_START_NM = 400.0
LEGACY_EMISSION_END_NM = 700.0
LEGACY_EMISSION_STEP_NM = 1.0
ABSORPTION_HEADER = "Wavelength(nm)\tTransmittance(%)\tAbsorbance"
ABSORPTION_WAVELENGTHS = tuple(range(700, 399, -1))
# A small amount above 100% can be caused by blank-reference noise. Larger
# values indicate that the absorption scan is unsuitable for quantitative use.
ABSORPTION_TRANSMITTANCE_QC_MAX_PERCENT = 110.0
CIE_OBSERVER = "CIE 1931 2 Degree Standard Observer"


class SpectrumContractError(ValueError):
    """Report a returned laboratory package that violates the fixed CPWL contract."""


class IngestSpectraTool:
    """Validate one complete returned spectrum directory and preserve raw evidence."""

    name = "ingest_spectra"

    def __init__(self, store: TaskStore) -> None:
        """Bind all submitted-spectrum artifacts to a single recoverable task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Require qualified emission data while retaining absorption as optional evidence."""
        source_root = Path(str(arguments["data_path"])).resolve()
        if not source_root.is_dir():
            raise SpectrumContractError("returned measurement data must be a directory")
        expected_ids = self._expected_detection_ids(str(arguments["experiment_plan"]))
        submitted_ids = {path.name for path in source_root.iterdir() if path.is_dir()}
        unexpected_entries = [path.name for path in source_root.iterdir() if not path.is_dir()]
        if submitted_ids != set(expected_ids) or unexpected_entries:
            missing = sorted(set(expected_ids) - submitted_ids)
            unexpected = sorted(submitted_ids - set(expected_ids)) + sorted(unexpected_entries)
            details = []
            if missing:
                details.append(f"missing sample directories: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected batch entries: {', '.join(unexpected)}")
            raise SpectrumContractError("; ".join(details))

        allow_legacy_emission = self._plan_uses_legacy_emission_contract(
            str(arguments["experiment_plan"])
        )
        parsed_samples = [
            self._parse_sample(
                source_root / sample_id,
                sample_id,
                allow_legacy_emission=allow_legacy_emission,
            )
            for sample_id in expected_ids
        ]
        emission_contracts = [
            _emission_grid_contract(sample["emission_wavelengths_nm"])
            for sample in parsed_samples
        ]
        if len({tuple(contract["emission_nm"]) for contract in emission_contracts}) != 1:
            raise SpectrumContractError("all samples in one batch must use the same emission wavelength grid")
        round_number = int(arguments.get("round", 0)) + 1
        data_origin = str(arguments.get("data_origin", "measured"))
        if data_origin not in {"measured", "synthetic_dry_run"}:
            raise SpectrumContractError("data_origin must be measured or synthetic_dry_run")
        notice_artifact = None
        if data_origin == "synthetic_dry_run":
            notice_artifact = f"artifacts/round-{round_number}/SYNTHETIC_DRY_RUN.md"
            # Preserve an in-task warning so copied test data cannot later look like a measurement.
            self.store.write_artifact_text(
                notice_artifact,
                "# Synthetic Dry Run\n\nThis returned spectrum package is simulation-only and is not scientific evidence.\n",
            )
        raw_root = self.store.artifact_path(f"artifacts/round-{round_number}/raw")
        for sample in parsed_samples:
            destination = raw_root / sample["sample_id"]
            destination.mkdir(parents=True, exist_ok=False)
            # Copy unmodified source files so future analysis can reproduce every result.
            shutil.copy2(Path(sample.pop("emission_source")), destination / "emission.txt")
            absorption_source = sample.pop("absorption_source")
            if absorption_source is not None:
                # Preserve a submitted absorption export even when it is scientifically unusable.
                shutil.copy2(Path(absorption_source), destination / "absorption.txt")
            sample["raw_emission_path"] = (destination / "emission.txt").relative_to(self.store.run_dir).as_posix()
            sample["raw_absorption_path"] = (
                (destination / "absorption.txt").relative_to(self.store.run_dir).as_posix()
                if absorption_source is not None
                else None
            )

        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/spectra_manifest.json",
            {
                "round": round_number,
                "experiment_plan": arguments["experiment_plan"],
                "source_path": str(source_root),
                # The source is declared by the CLI, never guessed from a directory name.
                "origin": {
                    "kind": data_origin,
                    "scientific_eligible": data_origin == "measured",
                    "notice_artifact": notice_artifact,
                },
                "spectral_contract": {
                    **emission_contracts[0],
                    "absorption_required": False,
                    "absorption_nm_when_usable": [700, 400, -1],
                    "absorption_point_count_when_usable": len(ABSORPTION_WAVELENGTHS),
                },
                "samples": parsed_samples,
            },
        )
        absorption_status_counts: dict[str, int] = {}
        for sample in parsed_samples:
            status = str(sample["absorption_qc"]["status"])
            absorption_status_counts[status] = absorption_status_counts.get(status, 0) + 1
        excluded_count = absorption_status_counts.get("excluded", 0)
        other_unavailable_count = sum(
            count
            for status, count in absorption_status_counts.items()
            if status not in {"usable", "excluded"}
        )
        summary = f"Validated and archived emission spectra for {len(parsed_samples)} detection samples."
        if excluded_count:
            summary += (
                f" Absorption remained optional; excluded {excluded_count} absorption spectra "
                "without blocking CIE analysis."
            )
        if other_unavailable_count:
            summary += (
                f" Another {other_unavailable_count} optional absorption files were missing, partial, "
                "or invalid without blocking CIE analysis."
            )
        return ToolResult(
            status="success",
            summary=summary,
            data={
                "round": round_number,
                "sample_count": len(parsed_samples),
                "absorption_status_counts": absorption_status_counts,
                # Retain the historical count while the richer status map supports optional files.
                "absorption_excluded_count": excluded_count,
            },
            artifacts=[artifact],
        )

    def _expected_detection_ids(self, experiment_plan: str) -> list[str]:
        """Recover exact detection IDs from the machine-readable plan adjacent to its workbook."""
        workbook_path = self.store.artifact_path(experiment_plan)
        design_path = workbook_path.with_name(f"{workbook_path.stem}_design.json")
        try:
            plan = json.loads(design_path.read_text(encoding="utf-8"))
            recipe_ids = [str(recipe["recipe_id"]) for recipe in plan["recipes"]]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise SpectrumContractError("could not resolve detection IDs from the experiment plan") from error
        if not recipe_ids or len(recipe_ids) != len(set(recipe_ids)):
            raise SpectrumContractError("experiment plan has no unique recipe IDs")
        return [f"{recipe_id}-D" for recipe_id in recipe_ids]

    def _plan_uses_legacy_emission_contract(self, experiment_plan: str) -> bool:
        """Permit old grids only when replaying a plan whose own explanation requested them."""
        explanation_path = self.store.artifact_path(experiment_plan).with_suffix(".md")
        try:
            explanation = explanation_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            # Missing provenance is treated as a current plan and therefore fails closed.
            return False
        return (
            LEGACY_EMISSION_RANGE_HEADER in explanation
            and EMISSION_RANGE_HEADER not in explanation
        )

    def _parse_sample(
        self, sample_dir: Path, sample_id: str, *, allow_legacy_emission: bool
    ) -> dict[str, Any]:
        """Read required emission and best-effort optional absorption without repairing data."""
        emission_path = sample_dir / "emission.txt"
        absorption_path = sample_dir / "absorption.txt"
        emission_wavelengths, intensities = _parse_emission(
            emission_path, allow_legacy=allow_legacy_emission
        )
        absorption_wavelengths: list[int] | None = None
        transmittances: list[float] | None = None
        absorbances: list[float | None] | None = None
        absorption_sha256: str | None = None
        absorption_source: str | None = None
        if not absorption_path.is_file():
            absorption_qc = _absorption_qc(
                "not_provided",
                ["absorption.txt was not provided"],
                scientific_use="optional absorption evidence was not provided",
            )
        else:
            absorption_source = str(absorption_path)
            absorption_sha256 = _sha256_file(absorption_path)
            try:
                absorption_wavelengths, transmittances, absorbances, absorption_qc = _parse_absorption(
                    absorption_path
                )
            except SpectrumContractError as error:
                # Invalid optional absorption evidence is archived but cannot reject valid emission data.
                absorption_qc = _absorption_qc(
                    "invalid_format",
                    [str(error)],
                    scientific_use="raw evidence retained; invalid for absorption-derived analysis",
                )
        return {
            "sample_id": sample_id,
            "emission_source": str(emission_path),
            "absorption_source": absorption_source,
            "emission_sha256": _sha256_file(emission_path),
            "absorption_sha256": absorption_sha256,
            "emission_wavelengths_nm": emission_wavelengths,
            "emission_intensities": intensities,
            "absorption_wavelengths_nm": absorption_wavelengths,
            "transmittance_percent": transmittances,
            "absorbance": absorbances,
            "absorption_qc": absorption_qc,
        }


class CalculateCieTool:
    """Calculate CIE 1931 2-degree chromaticity from previously qualified emission spectra."""

    name = "calculate_cie"

    def __init__(self, store: TaskStore) -> None:
        """Bind calculated chromaticity artifacts to the active scientific task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Integrate all archived emission spectra and write the existing measurement-result contract."""
        manifest_path = self.store.artifact_path(str(arguments["spectra_artifact"]))
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SpectrumContractError(f"could not read qualified spectra manifest: {error}") from error
        target = tuple(float(value) for value in arguments["target"])
        if len(target) != 2:
            raise SpectrumContractError("target CIE requires exactly two values")
        measurements = []
        for sample in manifest.get("samples", []):
            xyz, cie = _calculate_cie_1931(
                sample["emission_wavelengths_nm"], sample["emission_intensities"]
            )
            measurements.append(
                {
                    "sample_id": sample["sample_id"],
                    "cie": list(cie),
                    "xyz_relative": list(xyz),
                    "observer": CIE_OBSERVER,
                    "emission_raw_path": sample["raw_emission_path"],
                    "absorption_raw_path": sample["raw_absorption_path"],
                }
            )
        if not measurements:
            raise SpectrumContractError("qualified spectra manifest contains no samples")
        round_number = int(manifest["round"])
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/measurement_result.json",
            {
                "round": round_number,
                "experiment_plan": manifest["experiment_plan"],
                "spectra_manifest": str(arguments["spectra_artifact"]),
                "target_cie": list(target),
                "measurements": measurements,
            },
        )
        return ToolResult(
            status="success",
            summary=f"Calculated CIE 1931 2-degree chromaticity for {len(measurements)} samples.",
            data={"round": round_number, "sample_count": len(measurements)},
            artifacts=[artifact],
        )


def _parse_emission(path: Path, *, allow_legacy: bool = True) -> tuple[list[float], list[float]]:
    """Parse the current emission export or either supported historical grid."""
    lines = _read_text_lines(path)
    if (
        len(lines) < 5
        or tuple(lines[:2]) != EMISSION_COMMON_HEADER_PREFIX
        or lines[4] != EMISSION_COLUMN_HEADER
    ):
        raise SpectrumContractError(f"{path}: emission header or row count violates the fixed contract")
    expected_wavelengths = _emission_wavelengths_for_header(
        lines[2], lines[3], path, allow_legacy=allow_legacy
    )
    if len(lines) != 5 + len(expected_wavelengths):
        raise SpectrumContractError(f"{path}: emission header or row count violates the fixed contract")
    wavelengths: list[float] = []
    intensities: list[float] = []
    for row_number, line in enumerate(lines[5:], start=6):
        fields = line.split("\t")
        if len(fields) != 2:
            raise SpectrumContractError(f"{path}:{row_number}: emission rows require two tab-delimited columns")
        wavelength = _parse_finite_float(fields[0], path, row_number, "wavelength")
        intensity = _parse_finite_float(fields[1], path, row_number, "fluorescence intensity")
        if intensity < 0:
            raise SpectrumContractError(f"{path}:{row_number}: fluorescence intensity must be non-negative")
        wavelengths.append(wavelength)
        intensities.append(intensity)
    if not _wavelengths_match(wavelengths, expected_wavelengths):
        step_nm = expected_wavelengths[1] - expected_wavelengths[0]
        raise SpectrumContractError(
            f"{path}: emission wavelengths must be {expected_wavelengths[0]:g}-"
            f"{expected_wavelengths[-1]:g} nm ascending at {step_nm:g} nm"
        )
    if sum(intensities) <= 0:
        raise SpectrumContractError(f"{path}: emission intensity integral must be positive")
    return wavelengths, intensities


def _parse_absorption(
    path: Path,
) -> tuple[list[int], list[float], list[float | None], dict[str, Any]]:
    """Parse a fixed-grid UV-visible export and flag unusable absorption values."""
    lines = _read_text_lines(path)
    if not lines or lines[0] != ABSORPTION_HEADER or len(lines) != 1 + len(ABSORPTION_WAVELENGTHS):
        raise SpectrumContractError(f"{path}: absorption header or row count violates the fixed contract")
    wavelengths: list[int] = []
    transmittances: list[float] = []
    absorbances: list[float | None] = []
    non_finite_wavelengths: list[int] = []
    excessive_transmittance_wavelengths: list[int] = []
    for row_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != 3:
            raise SpectrumContractError(f"{path}:{row_number}: absorption rows require three tab-delimited columns")
        wavelength = _parse_integer_wavelength(fields[0], path, row_number)
        transmittance = _parse_finite_float(fields[1], path, row_number, "transmittance")
        absorbance = _parse_optional_finite_float(fields[2], path, row_number, "absorbance")
        if transmittance < 0:
            raise SpectrumContractError(f"{path}:{row_number}: transmittance must be non-negative")
        if absorbance is None:
            non_finite_wavelengths.append(wavelength)
        if transmittance > ABSORPTION_TRANSMITTANCE_QC_MAX_PERCENT:
            excessive_transmittance_wavelengths.append(wavelength)
        wavelengths.append(wavelength)
        transmittances.append(transmittance)
        absorbances.append(absorbance)
    if tuple(wavelengths) != ABSORPTION_WAVELENGTHS:
        raise SpectrumContractError(f"{path}: absorption wavelengths must be 700-400 nm descending at 1 nm")
    reasons: list[str] = []
    finite_absorbances = [value for value in absorbances if value is not None]
    if non_finite_wavelengths:
        reasons.append("non-finite absorbance values")
    if excessive_transmittance_wavelengths:
        reasons.append(
            f"transmittance exceeds {ABSORPTION_TRANSMITTANCE_QC_MAX_PERCENT:g}%"
        )
    if excessive_transmittance_wavelengths or not finite_absorbances:
        status = "excluded"
    elif non_finite_wavelengths:
        status = "partial"
    else:
        status = "usable"
    qc = _absorption_qc(
        status,
        reasons,
        scientific_use=(
            "qualified for absorption-derived analysis"
            if status == "usable"
            else (
                "finite maximum absorbance retained; full absorption analysis disabled"
                if status == "partial"
                else "raw evidence retained; excluded from absorption-derived analysis"
            )
        ),
        non_finite_absorbance_wavelengths_nm=non_finite_wavelengths,
        excessive_transmittance_wavelengths_nm=excessive_transmittance_wavelengths,
        transmittance_range_percent=[min(transmittances), max(transmittances)],
        max_finite_absorbance=max(finite_absorbances) if finite_absorbances else None,
    )
    return wavelengths, transmittances, absorbances, qc


def _absorption_qc(
    status: str,
    reasons: list[str],
    *,
    scientific_use: str,
    **details: Any,
) -> dict[str, Any]:
    """Create one stable optional-absorption status object for every intake outcome."""
    return {
        "status": status,
        "reasons": reasons,
        "scientific_use": scientific_use,
        **details,
    }


def _read_text_lines(path: Path) -> list[str]:
    """Read one UTF-8 instrument export, accepting a BOM only where the device emits one."""
    try:
        return path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise SpectrumContractError(f"{path}: could not read UTF-8 spectrum file") from error


def _parse_integer_wavelength(value: str, path: Path, row_number: int) -> int:
    """Require wavelength cells to be integer nanometres, preserving the fixed grid exactly."""
    parsed = _parse_finite_float(value, path, row_number, "wavelength")
    if not parsed.is_integer():
        raise SpectrumContractError(f"{path}:{row_number}: wavelength must be an integer nanometre")
    return int(parsed)


def _emission_wavelengths_for_header(
    range_header: str, step_header: str, path: Path, *, allow_legacy: bool
) -> tuple[float, ...]:
    """Map range and step declarations to the current or read-only historical grids."""
    if range_header == EMISSION_RANGE_HEADER and step_header == "步长: 0.2 nm":
        return _emission_wavelengths(EMISSION_START_NM, EMISSION_END_NM, DEFAULT_EMISSION_STEP_NM)
    if (
        allow_legacy
        and range_header == LEGACY_EMISSION_RANGE_HEADER
        and step_header == "步长: 0.2 nm"
    ):
        return _emission_wavelengths(
            LEGACY_EMISSION_START_NM, LEGACY_EMISSION_END_NM, DEFAULT_EMISSION_STEP_NM
        )
    if (
        allow_legacy
        and range_header == LEGACY_EMISSION_RANGE_HEADER
        and step_header == "步长: 1 nm"
    ):
        return _emission_wavelengths(
            LEGACY_EMISSION_START_NM, LEGACY_EMISSION_END_NM, LEGACY_EMISSION_STEP_NM
        )
    raise SpectrumContractError(
        f"{path}: emission range/step must be 360-760 nm at 0.2 nm; "
        "400-700 nm at 0.2 or 1 nm is accepted only for historical data"
    )


def _emission_wavelengths(start_nm: float, end_nm: float, step_nm: float) -> tuple[float, ...]:
    """Build supported grids with integer tenths to avoid floating-point accumulation."""
    start_tenths = round(start_nm * 10)
    end_tenths = round(end_nm * 10)
    step_tenths = round(step_nm * 10)
    if step_tenths <= 0 or not math.isclose(step_tenths / 10, step_nm, abs_tol=1e-12):
        raise SpectrumContractError("unsupported emission wavelength grid")
    return tuple(value / 10 for value in range(start_tenths, end_tenths + 1, step_tenths))


def _supported_emission_grids() -> tuple[tuple[float, ...], ...]:
    """List the active laboratory contract first, followed by read-only historical grids."""
    return (
        _emission_wavelengths(EMISSION_START_NM, EMISSION_END_NM, DEFAULT_EMISSION_STEP_NM),
        _emission_wavelengths(
            LEGACY_EMISSION_START_NM, LEGACY_EMISSION_END_NM, DEFAULT_EMISSION_STEP_NM
        ),
        _emission_wavelengths(
            LEGACY_EMISSION_START_NM, LEGACY_EMISSION_END_NM, LEGACY_EMISSION_STEP_NM
        ),
    )


def _wavelengths_match(actual: list[float], expected: tuple[float, ...]) -> bool:
    """Compare instrument wavelength values with a narrow numeric tolerance."""
    return len(actual) == len(expected) and all(
        math.isclose(value, target, abs_tol=1e-9) for value, target in zip(actual, expected)
    )


def _emission_grid_contract(wavelengths: list[float]) -> dict[str, Any]:
    """Validate a known emission grid and expose its compact artifact representation."""
    for expected in _supported_emission_grids():
        if _wavelengths_match(wavelengths, expected):
            # Persist the declared decimal step, not its binary floating-point residue.
            step_nm = round(expected[1] - expected[0], 10)
            return {
                "emission_nm": [expected[0], expected[-1], step_nm],
                "emission_point_count": len(expected),
                # Keep the former generic field readable by older run consumers.
                "point_count": len(expected),
            }
    raise SpectrumContractError("CIE calculation requires a supported emission wavelength grid")


def _parse_finite_float(value: str, path: Path, row_number: int, field: str) -> float:
    """Convert one instrument cell to a finite float and retain its source location on failure."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise SpectrumContractError(f"{path}:{row_number}: {field} is not numeric") from error
    if not math.isfinite(parsed):
        raise SpectrumContractError(f"{path}:{row_number}: {field} must be finite")
    return parsed


def _parse_optional_finite_float(
    value: str, path: Path, row_number: int, field: str
) -> float | None:
    """Return null for an instrument non-finite value while rejecting malformed text."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise SpectrumContractError(f"{path}:{row_number}: {field} is not numeric") from error
    return parsed if math.isfinite(parsed) else None


def _sha256_file(path: Path) -> str:
    """Create an immutable raw-file identity for audit without interpreting file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calculate_cie_1931(wavelengths: list[float], intensities: list[float]) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Integrate relative XYZ and convert it to CIE 1931 xy on a qualified emission grid."""
    _emission_grid_contract(wavelengths)
    if len(intensities) != len(wavelengths):
        raise SpectrumContractError("CIE calculation requires matching emission wavelength and intensity counts")
    intensity_array = np.asarray(intensities, dtype=float)
    if not bool(np.isfinite(intensity_array).all()) or bool(np.any(intensity_array < 0)):
        raise SpectrumContractError("CIE calculation requires finite, non-negative emission intensities")
    cmf_values = _cie_1931_cmf(np.asarray(wavelengths, dtype=float))
    wavelength_array = np.asarray(wavelengths, dtype=float)
    if hasattr(np, "trapezoid"):
        xyz = np.asarray(np.trapezoid(intensity_array[:, None] * cmf_values, wavelength_array, axis=0), dtype=float)
    else:
        xyz = np.asarray(np.trapz(intensity_array[:, None] * cmf_values, wavelength_array, axis=0), dtype=float)
    denominator = float(np.sum(xyz))
    if not bool(np.isfinite(xyz).all()) or denominator <= 0:
        raise SpectrumContractError("integrated CIE XYZ values must be finite and positive")
    return (float(xyz[0]), float(xyz[1]), float(xyz[2])), (float(xyz[0] / denominator), float(xyz[1] / denominator))


def _cie_1931_cmf(wavelengths: np.ndarray) -> np.ndarray:
    """Read the standard observer from colour-science and interpolate its published tabulation."""
    try:
        # colour-science may warn about optional Matplotlib; no plotting API is used here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='.*"Matplotlib" related API.*')
            import colour
    except ImportError as error:
        raise SpectrumContractError("install colour-science to calculate CIE 1931 chromaticity") from error
    cmfs = colour.MSDS_CMFS[CIE_OBSERVER]
    cmf_wavelengths = np.asarray(cmfs.wavelengths, dtype=float)
    cmf_values = np.asarray(cmfs.values, dtype=float)
    return np.column_stack(
        [np.interp(wavelengths, cmf_wavelengths, cmf_values[:, index]) for index in range(3)]
    )
