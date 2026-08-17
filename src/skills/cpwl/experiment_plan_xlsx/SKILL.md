---
name: experiment-plan-xlsx
description: Use when a CPWL agent must turn a bounded, tool-validated CPWL recipe batch into the repository's constrained XLSX experiment-plan artifact and matching explanation file.
version: 1
domain: cpwl
constraint_path: constraints/constraint.md
template_path: constraints/exp_template.xlsx
---
# CPWL Experiment Plan XLSX Skill

Use this skill after the Agent and deterministic tools have selected a bounded recipe batch and before returning experiment-plan artifacts. Its responsibility is the file contract: validate concentrations, expand each formulation into one synthesis row, map experiments to the fixed workbook columns, and produce the required files without changing the scientific intent.

## Immutable Authority

The following repository files are human-owned constraints and must never be modified by an agent or by self-evolution:

- `constraints/constraint.md`
- `constraints/exp_template.xlsx`

Use this precedence when the files appear inconsistent:

1. Explicit fixed rules in `constraints/constraint.md`.
2. Field definitions in the `填写说明` worksheet.
3. Example rows in the `导入实验模板` worksheet.

The example rows are demonstrations of field shapes, not compliant defaults. Do not copy their temperatures, times, wavelength ranges, simultaneous module flags, or concentration values.

An evolution proposal may improve explanations, validation, or error messages in this skill, but it must not alter a fixed value from `constraints/` unless a human first changes the constraint source.

## Required Artifacts

Every finalized experiment plan must produce exactly these two matching artifacts:

1. One `.xlsx` workbook created from `constraints/exp_template.xlsx`.
2. One explanation file with the same basename, preferably Markdown (`.md`).

Example:

```text
artifacts/experiment_plans/experiment_plan_001.xlsx
artifacts/experiment_plans/experiment_plan_001.md
```

Do not claim completion when only prose, JSON, CSV, Markdown, or an in-memory table was produced. If no tool can copy and edit XLSX files, report the missing capability instead of pretending an XLSX artifact exists.

## Input Gate

Before final file generation, obtain or derive all of the following:

- Experiment objective and target CIE.
- Ordered material list A-E; unused positions remain zero.
- Molecular weight for every material used.
- Final amount or concentration of every material in every experiment.
- Mixture basis: molar fraction, mass fraction, or stock-volume fraction.
- The manual measurement-result directory convention for every formulation.
- Any safety, solvent, stock-solution, or instrument assumptions needed by the explanation file.

Never invent a molecular weight, stock concentration, density, or mixture basis. If a required value cannot be verified, mark file generation as blocked and identify the exact missing input.

## Concentration Contract

For every material used in any row, report both:

- Molar volume concentration in `mmol/ml`.
- Mass volume concentration in `mg/ml`.

Use the conversion:

```text
mass concentration (mg/ml) = molar concentration (mmol/ml) * molecular weight (g/mol)
```

Apply these hard checks:

- Every material final molar volume concentration must be `<= 0.00006 mmol/ml`.
- Concentrations must be non-negative finite numbers.
- Unit labels must appear in the explanation file; never provide unlabeled concentration values.
- The workbook field `产物体积浓度` is the final product molar volume concentration in `mmol/ml`.
- The explanation file must show the molecular weight and conversion for every material.

For a binary blend, report all of the following in the explanation file:

- Material A final molar and mass volume concentrations.
- Material B final molar and mass volume concentrations.
- The A:B ratio and its basis.
- The total final product molar volume concentration.

When concentrations refer to stocks mixed by volume, calculate the final product concentration as:

```text
c_product = (c_A * V_A + c_B * V_B) / V_total
```

When the listed concentrations are already final component concentrations, calculate:

```text
c_product = c_A_final + c_B_final
```

State which formula was used. Do not mix stock and final concentrations in the same calculation.

## Fixed Volume Contract

- Every formulation produces exactly `8 ml`.
- Every A-E stock concentration is fixed at `0.0001 mmol/ml`.
- Calculate each material stock volume with `V_i = c_i_final * 8 / 0.0001`.
- Calculate solvent volume with `V_solvent = 8 - sum(V_i)`.
- Every individual A-E stock volume and the solvent volume must be `<= 4.8 ml`.
- Preflight every recipe before calling a design tool: total final concentration must be within `0.00004..0.0001 mmol/ml`, and each individual component concentration must be `<= 0.00006 mmol/ml`.
- If a preflight fails, correct the Director-authored concentrations and recompute all five stock volumes plus solvent; never submit the same invalid recipe again.
- Persist the five stock volumes and solvent volume in the machine-readable design and explanation file.
- Reject the plan before workbook generation when any volume exceeds `4.8 ml` or solvent volume is negative.

## Workbook Preservation Rules

Create the output by copying `constraints/exp_template.xlsx`, then replace the example experiment rows. Preserve:

- Workbook type as `.xlsx`.
- Worksheet names `导入实验模板` and `填写说明`.
- The `填写说明` worksheet unchanged.
- Header text, header order, styles, column widths, and frozen header row in `导入实验模板`.
- Exactly 32 columns from A through AF.

The required header order is:

```text
实验名称,启用合成,启用后处理,启用检测,物料摩尔百分比,产物体积浓度,第一个实验前清洗,停留时间,反应温度,背压,反应清洗,收集起止,沉淀循环次数,复溶循环次数,清洗循环次数,光通量确认方式,沉淀参数,复溶参数,清洗参数,后处理收集瓶,检测模块,继续后处理,检测设备,UV起始波长,UV结束波长,扫描模式,激发波长,发射起始波长,发射结束波长,发射波长,激发起始波长,激发结束波长
```

Do not add, delete, rename, reorder, merge, or hide columns.

## Experiment Row Rules

### Naming And Manual Measurement

- `实验名称` uses only `B{batch}-N{number}-S` for synthesis, for example `B1-N1-S`.
- `B` is the experiment batch, `N` is the formulation number, and `S` marks synthesis. Human measurement results use the corresponding `B{batch}-N{number}-D` directory name; `D` is not an XLSX row.
- Each name must be unique and no longer than 20 characters.

### Device Execution Groups

- B1 contains one to twenty-four formulations; B2 and later contain no more than twelve.
- Write exactly one synthesis row for each formulation in recipe-number order.
- Manual detection is outside the XLSX import workflow and is requested through the matching explanation file.

### Module Flags

For `启用合成`, `启用后处理`, and `启用检测`:

- Use logical `true` or `false` values, not the non-compliant example text `是` or `否`.
- At most one of the three fields may be `true` in a row.
- `启用后处理` is always `false`.
- Every generated row is a synthesis row with `启用合成=true` and `启用检测=false`.

### Material Fractions

- `物料摩尔百分比` contains exactly five comma-separated non-negative values for materials A-E.
- Use `0` for an unused material position.
- The five values must not all be zero.
- Prefer normalized molar fractions that sum to `1`, for example `0.5,0.5,0,0,0`.
- If the input uses an integer ratio, normalize it before writing and preserve the original ratio in the explanation file.
- Keep the A-E material ordering identical across the workbook and explanation file.

## Fixed Column Values

Write the following immutable values for every generated experiment row unless the source constraint file is changed by a human:


| Column | Field            | Required value |
| -------- | ------------------ | ---------------- |
| C      | 启用后处理       | `false`        |
| G      | 第一个实验前清洗 | `true`         |
| H      | 停留时间         | `1`            |
| I      | 反应温度         | `25`           |
| J      | 背压             | `1`            |
| K      | 反应清洗         | `1,2`          |
| L      | 收集起止         | `4,8`          |
| M      | 沉淀循环次数     | blank          |
| N      | 复溶循环次数     | blank          |
| O      | 清洗循环次数     | blank          |
| P      | 光通量确认方式   | blank          |
| Q      | 沉淀参数         | blank          |
| R      | 复溶参数         | blank          |
| S      | 清洗参数         | blank          |
| T      | 后处理收集瓶     | blank          |
| U      | 检测模块         | `3`            |
| V      | 继续后处理       | `false`        |
| W      | 检测设备         | `1,1`          |
| X      | UV起始波长       | `400`          |
| Y      | UV结束波长       | `700`          |
| Z      | 扫描模式         | `2`            |
| AA     | 激发波长         | `350`          |
| AB     | 发射起始波长     | `360`          |
| AC     | 发射结束波长     | `760`          |
| AD     | 发射波长         | `350`          |
| AE     | 激发起始波长     | `400`          |
| AF     | 激发结束波长     | `700`          |

The duplicate final label in `constraints/constraint.md` is resolved by the actual template: AF is `激发结束波长`, not a second `激发起始波长`.

## Explanation File Contract

The matching explanation file must contain these sections:

1. `实验目标`: target CIE, scientific purpose, and experiment series identifier.
2. `物料映射`: ordered A-E material names, molecular weights, stock information, and unused positions.
3. `浓度与换算`: per-material `mmol/ml`, `mg/ml`, five stock volumes, solvent volume, formulas, and calculation inputs.
4. `共混计算`: component concentrations, ratio basis, volumes when relevant, and final product concentration.
5. `人工检测与数据回传`: the `-D` directories expected after manual measurement.
6. `固定设备参数`: the immutable values copied into columns G-AF.
7. `假设与缺失信息`: assumptions used and unresolved blockers; no hidden assumptions.
8. `安全与执行说明`: solvent, dye, PPE, light exposure, waste, and handling notes.
9. `验证结果`: every check from the final validation gate with pass/fail evidence.
10. `配方辨别意图`: for every B{batch}-N{number}, document its falsifiable hypothesis, referenced measured samples (empty only for B1), and the next experimental strategy if the hypothesis is supported or not supported.

The explanation must be sufficient to audit every numeric value in the workbook. Besides, the explanation must clarify the reason why the must be designed like above.

## Generation Workflow

1. Obtain the tool-validated initial, model-candidate, or exploratory recipe matrix.
2. Validate required material identities, molecular weights, concentrations, ratios, and units.
3. Expand each formulation into one ordered synthesis row.
4. Assign unique experiment names and document the manual `-D` result directories.
5. Copy `constraints/exp_template.xlsx` to the requested artifact path.
6. Remove all example rows below the header in `导入实验模板`.
7. Write generated rows in exact A-AF order and preserve the template structure.
8. Write the matching explanation file with auditable concentration calculations.
9. Reopen the generated workbook with an XLSX parser and run the validation gate.
10. Return artifact paths, synthesis row count, and validation status.

## Final Validation Gate

Do not return a successful result unless every applicable check passes:

- Both matching `.xlsx` and explanation files exist.
- The workbook opens as XLSX and contains exactly the two required worksheets.
- `导入实验模板` has the exact 32-column header in the required order.
- `填写说明` remains unchanged.
- Every experiment name is non-empty, unique, and no longer than 20 characters.
- Every row has at most one enabled module and `启用后处理=false`.
- Every workbook row is a synthesis row and has `启用检测=false`.
- Every formulation has one documented manual `-D` result directory.
- Every material fraction field contains exactly five values in stable A-E order.
- Every material has both `mmol/ml` and `mg/ml` in the explanation.
- No material final concentration exceeds `0.00006 mmol/ml`.
- The product volume is exactly `8 ml`, every stock uses `0.0001 mmol/ml`, the total final concentration is within `0.00004..0.0001 mmol/ml`, and every A-E stock or solvent volume is `<= 4.8 ml`.
- Every binary blend includes component concentrations, ratio basis, and product concentration.
- All fixed columns match the immutable values above.
- Required blank columns are truly empty, not `0`, `null`, `N/A`, or a space.
- No placeholder such as `TBD`, `TODO`, `unknown`, or an invented molecular weight remains.

On failure, do not silently repair scientific values. Report the row, column, invalid value, expected rule, and required user or upstream-agent action.

## Output Handoff

Return a concise structured handoff containing:

- XLSX artifact path.
- Explanation artifact path.
- Number of workbook rows.
- Number of synthesis rows.
- Material A-E mapping.
- Maximum material molar concentration observed.
- Validation status and any blockers.

The files are the primary deliverables; a prose summary is not a substitute for them.
