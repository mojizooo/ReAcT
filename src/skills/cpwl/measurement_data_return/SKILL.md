---
name: measurement-data-return
description: Use when requesting or accepting required CPWL emission spectra and optional absorption spectra for an approved experiment batch.
version: 1
domain: cpwl
---
# CPWL Measurement Data Return

Use this Skill after a CPWL XLSX experiment plan is generated and before returned
spectra are processed. It defines the only accepted laboratory data package. Do
not ask the laboratory user to calculate or submit CIE coordinates: the
deterministic `calculate_cie` tool calculates them from the qualified emission
spectra.

## Required Agent Feedback

When a batch enters `WAITING_FOR_DATA`, give the laboratory user these sections
in this exact order. The matching CPWL plan explanation file must contain the
same information.

1. `回传批次`: batch identifier, target CIE, and the expected data-root name.
2. `必交样品`: one detection-result identifier for every synthesis formulation in the approved XLSX plan.
3. `目录结构`: the fixed directory tree below.
4. `发射谱文件合同`: the required `emission.txt` structure.
5. `吸收谱文件合同`: the optional `absorption.txt` structure.
6. `系统自动处理`: state that CIE 1931 2-degree `x,y` is calculated by the system.
7. `拒收条件`: list missing samples, unexpected samples, invalid names, malformed
   columns, invalid wavelength grids, and invalid emission values. Non-finite
   absorbance values are retained as raw evidence and excluded from absorption analysis.

## Directory Contract

Each approved plan automatically creates an empty data-root directory at
`artifacts/measurement_returns/round-{batch:03}/`. Submit that directory after
placing required `emission.txt` and any available optional `absorption.txt` into every matching detection-result folder.
Its immediate children must be exactly the detection-result IDs corresponding
to the XLSX synthesis rows. A detection-result ID uses `B{batch}-N{number}-D`;
`B` is the batch, `N` is the formulation number, and `D` marks a manually
measured result, not an XLSX row.

```text
round-001/
├── B1-N1-D/
│   ├── emission.txt
│   └── absorption.txt  # optional
├── B1-N2-D/
│   ├── emission.txt
│   └── absorption.txt
└── ...
```

Do not submit a manually assembled CIE JSON file. Do not omit a detection
sample or reuse one spectrum for multiple sample directories. Do not add an
unknown sample directory to the batch.

## Emission File Contract

`emission.txt` is a UTF-8 text file. It contains exactly these five header
lines, followed by 2001 tab-delimited numeric rows:

```text
Scan Mode: 发射扫描
激发波长: 350 nm
发射波长范围: 360 - 760 nm
步长: 0.2 nm
波长(nm)\t荧光强度
```

- The wavelength rows are the grid `360.0, 360.2, ..., 760.0` in ascending
  order, exactly once each.
- Each row has exactly two columns separated by one tab: wavelength in `nm`
  and fluorescence intensity.
- Intensities must be finite, non-negative numbers and their total must be
  positive.
- The instrument filename may contain another nominal scan range, but its
  contents must meet this fixed contract. Do not interpolate, reorder,
  smooth, baseline-correct, or repair submitted values.
- The parser accepts the former `400-700 nm` grids at `0.2 nm` or `1 nm` only
  to read historical runs. New laboratory returns must use the grid above.

## Optional Absorption File Contract

`absorption.txt` is optional. When supplied, it is UTF-8 text (an optional BOM is accepted). Its first line
is exactly:

```text
Wavelength(nm)\tTransmittance(%)\tAbsorbance
```

- It has 301 data rows on the integer grid `700, 699, ..., 400` in descending
  order, exactly once each.
- Every row has exactly three tab-delimited fields: wavelength in `nm`,
  transmittance in percent, and absorbance. Wavelength and transmittance must be
  finite numbers. Instrument-produced `NaN`/`Inf` absorbance is retained as raw
  evidence but makes that sample's absorption spectrum ineligible for analysis.
- Transmittance must be non-negative. Small negative absorbance values caused
  by blank correction are retained as measured values.
- Transmittance above 110% is treated as a gross QC failure. The original file
  is retained, but the absorption spectrum is excluded from derived features.
- A missing file is recorded as `not_provided` and never blocks emission-based
  CIE. A partially finite file is recorded as `partial` and contributes only
  its finite maximum absorbance; full absorption feature analysis requires
  `usable` status. Invalid-format files are archived when present.
- The absorption spectrum is used for optional quality control and later optical
  analysis; it is not used to calculate CIE `x,y`.

## CIE Boundary

The `calculate_cie` tool uses the CIE 1931 2 Degree Standard Observer and the
qualified emission values on the fixed 360-760 nm, 0.2 nm grid. It integrates
relative `X`, `Y`, and `Z`, then reports `x = X/(X+Y+Z)` and
`y = Y/(X+Y+Z)`. The Agent must cite only its persisted result and must not
calculate, adjust, or claim CIE values in prose.
