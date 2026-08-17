"""Controlled, measured-only spectral feature extraction for Director diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from react_color_agent.models import ToolResult
from react_color_agent.storage import TaskStore
from tools.spectrum_tools import _parse_absorption, _parse_emission


class ExtractSpectralFeaturesTool:
    """Extract fixed spectral descriptors without making an optical-mechanism claim."""

    name = "extract_spectral_features"

    def __init__(self, store: TaskStore) -> None:
        """Bind controlled dataset and raw-spectrum access to one research task."""
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Calculate requested measured-spectrum features and write one reproducible artifact."""
        sample_ids = _validate_sample_ids(arguments.get("sample_ids"))
        kind = _validate_kind(arguments.get("kind"))
        band_nm = _validate_band(arguments.get("band_nm"))
        compare_to = _validate_optional_sample_id(arguments.get("compare_to_sample_id"))
        dataset_artifact = str(arguments["research_dataset_artifact"])
        dataset = _load_json(self.store, dataset_artifact)
        observations = _measured_observations(dataset)
        requested_ids = list(dict.fromkeys([*sample_ids, *([compare_to] if compare_to else [])]))
        missing = [sample_id for sample_id in requested_ids if sample_id not in observations]
        if missing:
            raise ValueError(
                "sample_ids must identify measured observations in the current dataset: " + ", ".join(missing)
            )

        feature_rows = [
            self._features_for_observation(observations[sample_id], kind, band_nm)
            for sample_id in sample_ids
        ]
        features_by_id = {row["sample_id"]: row for row in feature_rows}
        if compare_to and compare_to not in features_by_id:
            features_by_id[compare_to] = self._features_for_observation(
                observations[compare_to], kind, band_nm
            )
        comparisons = _comparisons(feature_rows, features_by_id.get(compare_to)) if compare_to else []
        limitations = list(
            dict.fromkeys(
                limitation
                for row in feature_rows
                for limitation in row["limitations"]
            )
        )
        if compare_to:
            limitations.extend(
                limitation
                for comparison in comparisons
                for limitation in comparison["limitations"]
                if limitation not in limitations
            )
        public_features = [_public_feature_row(row) for row in feature_rows]

        round_number = int(arguments["round"])
        input_contract = {
            "sample_ids": sample_ids,
            "kind": kind,
            "band_nm": band_nm,
            "compare_to_sample_id": compare_to,
        }
        digest = _canonical_sha256(input_contract)[:16]
        artifact = self.store.write_artifact_json(
            f"artifacts/round-{round_number}/spectral_features_{digest}.json",
            {
                "schema_version": 1,
                "round": round_number,
                "source_dataset": dataset_artifact,
                "source_dataset_sha256": _canonical_sha256(dataset),
                "origin_policy": "measured_only",
                **input_contract,
                "sample_features": public_features,
                "comparisons": comparisons,
                "limitations": limitations,
                "interpretation_boundary": (
                    "These are deterministic spectral descriptors, not a claim about optical mechanism, "
                    "quantum yield, or CIE prediction."
                ),
            },
        )
        return ToolResult(
            status="success",
            summary=(
                f"Extracted {kind} spectral features for {len(feature_rows)} measured samples"
                + (" and compared them to the requested reference." if compare_to else ".")
            ),
            data={
                "round": round_number,
                "kind": kind,
                "sample_count": len(feature_rows),
                "comparison_count": len(comparisons),
                "sample_features": public_features,
                "comparisons": comparisons,
                "limitations": limitations,
            },
            artifacts=[artifact],
        )

    def _features_for_observation(
        self,
        observation: dict[str, Any],
        kind: str,
        band_nm: list[float] | None,
    ) -> dict[str, Any]:
        """Read one archived spectrum through dataset evidence and calculate its fixed descriptors."""
        sample_id = str(observation["identity"]["sample_id"])
        evidence = observation["evidence"]
        absorption_qc = evidence.get("absorption_qc", {"status": "usable", "reasons": []})
        if kind == "absorption" and absorption_qc.get("status") != "usable":
            status = str(absorption_qc.get("status", "unknown"))
            raise ValueError(
                f"absorption spectrum for {sample_id} is excluded from analysis "
                f"because QC status is {status}: "
                + "; ".join(str(reason) for reason in absorption_qc.get("reasons", []))
            )
        raw_reference = evidence.get(f"{kind}_raw_path")
        if not raw_reference:
            raise ValueError(f"no {kind} spectrum was provided for {sample_id}")
        raw_path = self.store.artifact_path(str(raw_reference))
        wavelengths, values = _read_spectrum(raw_path, kind)
        feature = _spectrum_features(wavelengths, values, kind, band_nm)
        return {
            "sample_id": sample_id,
            "raw_path": raw_path.relative_to(self.store.run_dir).as_posix(),
            "sha256": evidence[f"{kind}_sha256"],
            "features": feature["features"],
            "comparison_vector": feature["comparison_vector"],
            "limitations": feature["limitations"],
        }


def _load_json(store: TaskStore, artifact: str) -> dict[str, Any]:
    """Load one run-local JSON artifact through TaskStore path validation."""
    try:
        return json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load JSON artifact {artifact}") from error


def _canonical_sha256(payload: Any) -> str:
    """Hash a JSON-compatible value with the repository's stable artifact convention."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measured_observations(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index only measured observations so dry-run spectra cannot become scientific evidence."""
    return {
        str(observation["identity"]["sample_id"]): observation
        for batch in dataset.get("batches", [])
        if bool(batch.get("origin", {}).get("scientific_eligible"))
        for observation in batch.get("observations", [])
    }


def _validate_sample_ids(value: Any) -> list[str]:
    """Require a bounded, duplicate-free sequence of stable sample identifiers."""
    if not isinstance(value, list) or not 1 <= len(value) <= 12 or not all(isinstance(item, str) for item in value):
        raise ValueError("sample_ids must contain one to twelve sample IDs")
    sample_ids = [item.strip() for item in value]
    if any(not item for item in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be non-empty and unique")
    return sample_ids


def _validate_kind(value: Any) -> str:
    """Limit diagnostic calculations to the two fixed CPWL spectrum types."""
    kind = str(value or "").strip()
    if kind not in {"emission", "absorption"}:
        raise ValueError("kind must be emission or absorption")
    return kind


def _validate_band(value: Any) -> list[float] | None:
    """Accept one finite closed wavelength interval when the Agent needs a focused band."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("band_nm must be a two-value [start_nm, end_nm] interval")
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise ValueError("band_nm values must be finite numbers") from error
    if not math.isfinite(start) or not math.isfinite(end) or start > end:
        raise ValueError("band_nm requires finite start_nm less than or equal to end_nm")
    return [start, end]


def _validate_optional_sample_id(value: Any) -> str | None:
    """Normalize an optional single comparison reference without allowing blank identifiers."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("compare_to_sample_id must be a non-empty sample ID when provided")
    return value.strip()


def _read_spectrum(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse original fixed-contract files without modifying or resampling their grids."""
    if kind == "emission":
        wavelengths, values = _parse_emission(path)
    else:
        wavelengths, _transmittance, values, absorption_qc = _parse_absorption(path)
        if absorption_qc.get("status") == "excluded":
            raise ValueError(
                "absorption spectrum is excluded from analysis: "
                + "; ".join(str(reason) for reason in absorption_qc.get("reasons", []))
            )
    return np.asarray(wavelengths, dtype=float), np.asarray(values, dtype=float)


def _spectrum_features(
    wavelengths: np.ndarray,
    values: np.ndarray,
    kind: str,
    band_nm: list[float] | None,
) -> dict[str, Any]:
    """Calculate descriptors from one unaltered spectrum and retain its comparison vector."""
    peak_index = int(np.argmax(values))
    integral = _integrate(wavelengths, values)
    limitations: list[str] = []
    if int(np.count_nonzero(values == values[peak_index])) > 1:
        limitations.append("multiple equal maximum values; peak wavelength uses the first maximum on file order")
    centroid = _spectral_centroid(wavelengths, values, integral)
    if centroid is None:
        limitations.append("integrated signal is zero, so spectral centroid is undefined")

    features: dict[str, Any] = {
        "wavelength_range_nm": [float(wavelengths.min()), float(wavelengths.max())],
        "sample_count": int(len(wavelengths)),
        "spectral_centroid_nm": centroid,
    }
    if kind == "emission":
        fwhm, fwhm_limitations = _fwhm(wavelengths, values, peak_index)
        limitations.extend(fwhm_limitations)
        features.update(
            {
                "peak_wavelength_nm": float(wavelengths[peak_index]),
                "peak_intensity": float(values[peak_index]),
                "integrated_intensity": integral,
                "fwhm_nm": fwhm,
            }
        )
    else:
        features.update(
            {
                "max_absorbance_wavelength_nm": float(wavelengths[peak_index]),
                "max_absorbance": float(values[peak_index]),
                "integrated_absorbance": integral,
            }
        )
    if band_nm is not None:
        features["band"] = _band_features(wavelengths, values, band_nm, integral, kind)
    return {"features": features, "comparison_vector": _normalized_vector(values, integral), "limitations": limitations}


def _spectral_centroid(wavelengths: np.ndarray, values: np.ndarray, integral: float) -> float | None:
    """Return the intensity-weighted wavelength centroid when the spectrum has nonzero area."""
    if math.isclose(integral, 0.0, abs_tol=1e-15):
        return None
    return float(_integrate(wavelengths, wavelengths * values) / integral)


def _integrate(wavelengths: np.ndarray, values: np.ndarray) -> float:
    """Integrate on ascending wavelength order so absorption-area values stay non-negative."""
    if float(wavelengths[0]) > float(wavelengths[-1]):
        return float(np.trapezoid(values[::-1], wavelengths[::-1]))
    return float(np.trapezoid(values, wavelengths))


def _band_features(
    wavelengths: np.ndarray,
    values: np.ndarray,
    band_nm: list[float],
    total_integral: float,
    kind: str,
) -> dict[str, Any]:
    """Calculate a focused-band projection only from points present in the archived grid."""
    start, end = band_nm
    mask = (wavelengths >= start) & (wavelengths <= end)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError("band_nm must contain at least two archived spectrum points")
    band_wavelengths, band_values = wavelengths[mask], values[mask]
    peak_index = int(np.argmax(band_values))
    integral = _integrate(band_wavelengths, band_values)
    fraction = None if math.isclose(total_integral, 0.0, abs_tol=1e-15) else float(integral / total_integral)
    value_key = "peak_intensity" if kind == "emission" else "max_absorbance"
    return {
        "requested_range_nm": [start, end],
        "resolved_range_nm": [float(band_wavelengths.min()), float(band_wavelengths.max())],
        "integral": integral,
        "fraction_of_total_integral": fraction,
        "peak_wavelength_nm": float(band_wavelengths[peak_index]),
        value_key: float(band_values[peak_index]),
    }


def _fwhm(wavelengths: np.ndarray, values: np.ndarray, peak_index: int) -> tuple[float | None, list[str]]:
    """Use adjacent half-height crossings to estimate emission FWHM without curve fitting."""
    peak = float(values[peak_index])
    if peak <= 0:
        return None, ["emission peak is not positive, so FWHM is undefined"]
    half = peak / 2.0
    left = _half_crossing(wavelengths, values, peak_index, -1, half)
    right = _half_crossing(wavelengths, values, peak_index, 1, half)
    if left is None or right is None:
        return None, ["FWHM is undefined because the emission peak lacks two half-height crossings"]
    return float(right - left), []


def _half_crossing(
    wavelengths: np.ndarray,
    values: np.ndarray,
    peak_index: int,
    direction: int,
    half: float,
) -> float | None:
    """Locate one nearest half-height crossing by linear interpolation on the file grid."""
    if direction < 0:
        indices = range(peak_index, 0, -1)
        pairs = ((index, index - 1) for index in indices)
    else:
        indices = range(peak_index, len(values) - 1)
        pairs = ((index, index + 1) for index in indices)
    for first, second in pairs:
        first_value, second_value = float(values[first]), float(values[second])
        if (first_value >= half >= second_value) or (first_value <= half <= second_value):
            if math.isclose(first_value, second_value):
                return float(wavelengths[first])
            ratio = (half - first_value) / (second_value - first_value)
            return float(wavelengths[first] + ratio * (wavelengths[second] - wavelengths[first]))
    return None


def _normalized_vector(values: np.ndarray, integral: float) -> list[float] | None:
    """Normalize by spectral area for shape comparison while preserving zero-signal limitations."""
    if math.isclose(integral, 0.0, abs_tol=1e-15):
        return None
    return [float(value / integral) for value in values]


def _comparisons(
    sample_rows: list[dict[str, Any]], reference: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Compare requested spectra to one measured reference without interpreting the differences."""
    if reference is None:
        return []
    rows = []
    reference_features = reference["features"]
    reference_integral = _integral(reference_features)
    for row in sample_rows:
        if row["sample_id"] == reference["sample_id"]:
            continue
        features = row["features"]
        limitations: list[str] = []
        integral = _integral(features)
        if math.isclose(reference_integral, 0.0, abs_tol=1e-15):
            ratio = None
            limitations.append("reference integrated signal is zero, so integral ratio is undefined")
        else:
            ratio = float(integral / reference_integral)
        similarity = _cosine_similarity(row["comparison_vector"], reference["comparison_vector"])
        if similarity is None:
            limitations.append("one spectrum has zero integrated signal, so shape similarity is undefined")
        rows.append(
            {
                "sample_id": row["sample_id"],
                "reference_sample_id": reference["sample_id"],
                "peak_wavelength_shift_nm": float(
                    _peak_wavelength(features) - _peak_wavelength(reference_features)
                ),
                "integral_ratio_to_reference": ratio,
                "normalized_shape_cosine_similarity": similarity,
                "limitations": limitations,
            }
        )
    return rows


def _public_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop the internal normalized vector so artifacts stay compact and Agent-readable."""
    return {key: value for key, value in row.items() if key != "comparison_vector"}


def _integral(features: dict[str, Any]) -> float:
    """Read the kind-specific integral field through one common comparison helper."""
    return float(features.get("integrated_intensity", features.get("integrated_absorbance")))


def _peak_wavelength(features: dict[str, Any]) -> float:
    """Read the kind-specific peak coordinate through one common comparison helper."""
    return float(features.get("peak_wavelength_nm", features.get("max_absorbance_wavelength_nm")))


def _cosine_similarity(first: list[float] | None, second: list[float] | None) -> float | None:
    """Return shape-only cosine similarity after area normalization on the fixed common grid."""
    if first is None or second is None:
        return None
    first_values, second_values = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first_values.shape != second_values.shape:
        raise ValueError("spectra use different grids and cannot be compared without resampling")
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return float(np.dot(first_values, second_values) / denominator)
