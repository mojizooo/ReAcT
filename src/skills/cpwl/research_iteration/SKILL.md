---
name: research-iteration
description: Use when interpreting completed CPWL rounds and designing the smallest evidence-supported batch that can reach the target CIE quickly.
version: 2
domain: cpwl
---
# CPWL Research Iteration

Use this Skill after a missed or provisional deterministic `check_goal`. The
current task's `research_dataset.json` is the only sample-level scientific fact
source. Its index and briefing are compact access paths, not additional fact
sources.

## Required Starting Evidence

For an ordinary missed target:

1. Read `injected_research_briefing` when present.
2. Call `query_research_index` with the target CIE to identify the closest
   measured samples and relevant gaps.
3. Call `review_design_outcomes` for the just-completed measured batch.
4. State one research question and choose the design approach that best answers
   it. Do not force the data into a predefined material-substitution path.
5. Retrieve or calculate only evidence that can change the proposed recipes.
6. Activate `experiment-plan-xlsx` and `measurement-data-return`, then submit
   the complete draft through `propose_followup_batch`.

Do not run every analysis tool by default. Call `get_complete_batch_history`
only for a genuine whole-task comparison; `get_analysis_record` only for a
relevant prior analysis or decision; `get_experiment_record` only for relevant
samples; and `get_spectrum_data` only for a concrete spectral question. Never
infer artifact paths.

When the question concerns peak position, shape, relative intensity, or an
absorption-band change, call `extract_spectral_features` with only the needed
measured sample IDs. Deterministic descriptors are facts, but they do not
establish a mechanism, quantum yield, or predicted CIE.

## Design Approaches

The strategy label describes the Director's main approach; it is not a fixed
state machine. A batch may combine approaches when the allocation reason is
explicit.

| Strategy | Use when | Typical action |
| --- | --- | --- |
| `local_refinement` | A measured target-near region has a useful local trend. | Refine ratios or concentrations around that region. |
| `model_guided` | A local model has acceptable validation and the candidates are interpolation. | Prioritize model-supported recipes while retaining a measured-evidence competitor. |
| `component_adjustment` | Measured evidence supports changing one component to correct the current CIE residual. | Adjust or replace one component without claiming a mechanism. |
| `region_switch` | The current region cannot plausibly improve both CIE coordinates, while another measured region is more promising. | Move the primary allocation to the better-supported region. |
| `concentration_scan` | Total concentration may matter at a fixed or nearly fixed ratio. | Change total concentration over a small justified range. |
| `multi_component_search` | More than one component must change to move toward the target. | Search a compact local multi-component region. |
| `diagnostic` | Conflicting measured patterns prevent a reliable optimization direction. | Use the fewest recipes that distinguish decision-relevant hypotheses. |

## Optional Analysis And Model Path

Use deterministic analysis only when it can improve the batch:

1. `diagnose_dataset` describes measured coverage, CIE geometry, active
   dimensions, QC, and limitations.
2. `screen_composition_effects` compares measured anchors and sparse probes
   without assuming linearity.
3. `compile_research_analysis` records one specific interpolation or
   composition question.
4. For a fixed local scope with one or two active slots, fit both
   `local_ridge` and `weighted_neighbors` on the same measured samples.
5. Use `compare_models` to inspect LOOCV errors, support, scope, and
   limitations. The comparison supplies evidence; it does not choose the
   scientific strategy.

Use `model_guided` only when the selected model is supported and the proposed
recipes remain inside measured local coverage. Otherwise reject the model and
choose a direct evidence-based approach. Do not call the legacy predicted-
candidate or direct plan tools for an ordinary missed target; the reviewed
draft workflow owns normal B2-and-later planning.

## Reviewed Batch Workflow

`propose_followup_batch` receives one complete Director-authored batch:

- Include 2 to 12 finite, non-negative A-E concentration recipes.
- Concentrate most experiments in the region most likely to reach the target.
- Include at least one scientifically supported `competing` recipe; do not use
  a fixed primary-to-competing ratio.
- Cite measured sample IDs for every factual claim and recipe reference.
- Label unmeasured optical behavior and explanations as hypotheses.
- Do not add routine repeats, same-batch controls, or pure verification recipes
  before a target hit merely to fill the batch.
- Avoid measured duplicates, batch duplicates, needlessly dense scans, and
  unsupported extrapolation unless the research question explicitly justifies
  their opportunity cost.

The tool saves the draft and evaluates the actual recipes for validity,
nearest measured neighbors, coverage, redundancy, role allocation, and any
supported model predictions. The evaluation informs the Director but never
chooses a strategy or rewrites a recipe.

The Runtime then performs exactly one independent Scientific Critic request.
After its review is injected, respond exactly once to every finding through
`finalize_followup_batch`. Accept, partially accept, or reject each finding
with measured evidence, revise the recipes when useful, and submit the complete
final object once. The Director retains final authority. Do not start a second
analysis or Critic loop merely to postpone finalization.

## Target Confirmation Exception

The first deterministic `check_goal` hit from `measured` data is provisional.
While `provisional_goal_candidate` is present, do not use the ordinary reviewed
draft workflow. Instead:

1. Call `write_design_decision` with `strategy=repeat_validation` and
   `selected_method=none`, citing the provisional sample.
2. Call `design_exploratory_followup_batch` with that same reference sample and
   an exactly matching recipe for independent preparation and measurement.
3. Finish the task only when the later deterministic `check_goal` result also
   passes the coordinate-wise tolerance: both `|x-x_target| <= 0.005` and
   `|y-y_target| <= 0.005` must hold. The stored Euclidean distance is still
   useful for ranking and reporting, but it is not the acceptance rule.

If confirmation fails, preserve both measurements as facts and return to the
ordinary reviewed optimization workflow.

## Scientific Boundaries

Keep facts separate from hypotheses. Facts are persisted recipes, CIE values,
QC fields, and referenced spectra. A hypothesis may motivate a recipe but must
not be presented as a measured optical mechanism.

`synthetic_dry_run` observations validate the workflow only. They cannot select
a scientific candidate, justify a material claim, or establish target success.

Emission spectra are required for CIE. Absorption spectra are optional auxiliary
evidence: `not_provided`, `partial`, `excluded`, or `invalid_format` must never
block the loop. Use full absorption features only when the stored QC status is
`usable`; a `partial` spectrum contributes at most its finite maximum
absorbance. The 400--700 nm absorption contract does not support a quantitative
350 nm inner-filter assessment.
