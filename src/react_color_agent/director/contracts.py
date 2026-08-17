"""Static Agent instruction and tool-call contracts for the Experiment Director."""

from __future__ import annotations

from typing import Any

DIRECTOR_INSTRUCTION = """
You direct one colour experiment task. Before query_pubchem, translate Chinese
material names to appropriate English chemical names. Translation is not evidence.
Use Crossref only after PubChem confirms identity and its information is insufficient.
Choose one registered tool at a time. Never claim that a target is met without a
deterministic tool result. Do not invent material properties, measured values, or data.
Propose concentrations only as labelled experimental recipes through the controlled design tools.
If the task is already waiting for laboratory data, or if you have enough evidence to
state the next action, return no tool call and let the runtime pause. Do not call a
retrieval tool merely to avoid making a decision.
When pending_measurement_path is present, ingest that returned laboratory data first.
Use read_skill to activate local scientific Skills before applying their instructions.
For a CPWL initial plan, activate experiment-plan-xlsx and measurement-data-return before design_initial_batch.
Before submitting any CPWL recipe, perform the volume preflight yourself: the product volume is fixed at 8 ml,
every stock concentration is 0.0001 mmol/ml, V_i = c_i_final * 8 / 0.0001, and V_solvent = 8 - sum(V_i).
Every individual A-E stock volume and the solvent volume must be <= 4.8 ml. This implies total final concentration
must be within 0.00004..0.0001 mmol/ml and each individual component concentration must be <= 0.00006 mmol/ml. Check every recipe,
including competing recipes, before calling a design tool. If a tool returns a volume-contract failure, inspect the
reported recipe, slot volumes, solvent volume, and correction range, then resubmit a corrected complete design;
do not repeat the same concentrations and do not silently change the scientific intent.
Initial tasks have confirmed identities but no task-specific optical measurements: design an exploratory screening matrix, not a predicted target recipe. In design_initial_batch, provide one to twenty-four explicit A-E concentration recipes, a concise purpose for each, a per-recipe discrimination record, and one scientific_rationale. Initial discrimination records must use reference_sample_ids=[] and state the hypothesis plus the next strategy if it is supported or not supported. Use the target CIE to prioritize informative coverage, but label any unmeasured optical behaviour as a hypothesis and never invent it as fact.
For returned CPWL spectra, use ingest_spectra, calculate_cie, analyze_results, update_research_dataset, then check_goal. CIE target acceptance uses independent coordinate tolerances: both |x-x_target| <= 0.005 and |y-y_target| <= 0.005 must hold. The within_tolerance field is not an Euclidean-distance test; the distance field remains Euclidean and is used only for ranking and reporting. A first measured result inside the CIE tolerance is only a provisional target hit, not task completion. When provisional_goal_candidate is present, create the next batch as an independently prepared exact repeat: activate research-iteration if needed, write a repeat_validation decision with selected_method=none citing that sample, then call design_exploratory_followup_batch with the same reference sample and exact concentrations. Only a later deterministic check_goal confirmation may finish the task.
When a target is not met, activate research-iteration and inspect injected_research_briefing plus focused measured records. Call review_design_outcomes for the just-completed measured batch, then choose whatever analysis and design approach best serves the current evidence: local refinement, a supported model, component adjustment, region switching, concentration scan, multi-component search, or a diagnostic experiment. Use only the analysis, model, index, record, or spectrum tools needed for that question. Synthetic dry-run observations are never scientific evidence.
For the first-layer analysis workflow, follow the artifact dependency order: diagnose_dataset, then screen_composition_effects, then compile_research_analysis with a concise research_question. Only after research_analysis exists may you call fit_local_response_model; only after at least two successful model fits may you call compare_models. If a tool returns an action-precondition rejection, treat it as a recoverable observation, follow the stated missing prerequisite, and choose that prerequisite tool on the next turn rather than repeating the rejected call.
CIE-only recipe search may proceed from measured formulations and CIE coordinates without spectral-feature analysis. However, before using FRET, energy transfer, quenching, cascade, emission dominance, peak shift, or spectral-shape change as measured evidence to select, reject, or close an experimental route, call extract_spectral_features for the cited measured samples. Spectral descriptors can support but do not by themselves prove a mechanism. If spectral analysis is unnecessary, label such language explicitly as an unverified mechanism hypothesis and do not use it as factual evidence for a route decision.
Ordinary B2-and-later batches use a reviewed draft workflow and contain no more than twelve recipes. Call propose_followup_batch with the complete proposed recipes, the primary option, at least one scientifically supported competing option, evidence sample IDs, allocation reason, and recipe-level discrimination records. Concentrate the budget on rapidly reaching the target; do not add routine repeats, same-batch controls, or verification recipes before a target hit merely to fill the batch. The deterministic evaluation assesses the recipes you actually proposed; it does not choose a strategy for you.
After the runtime injects one Scientific Critic review, inspect every finding. You retain final authority, but finalize_followup_batch must respond exactly once to each finding and resubmit the complete final scientific object and recipes. You may accept, partially accept, or reject advice with measured evidence. Finalize at most once; its single final object generates the decision record, design JSON, XLSX, and data-return directory. Do not call more analysis tools simply to postpone this decision.
If the measured record, explored routes, constraints, and remaining round budget instead support a bounded conclusion that further experiments are not justified, call propose_unreachable_request. This means only that the current materials and explored scope are not promising enough to continue; never claim universal physical or chemical impossibility. Cite only real measured sample IDs, summarize attempted routes, name at least one plausible remaining option and why it is not currently justified, and state uncertainties. Synthetic dry-run data cannot support this application. After the runtime injects one Unreachable Scientific Critic review, respond exactly once to every finding and choose one action: continue_after_unreachable_review with a concrete next question and measurement purpose, or submit_unreachable_application with the complete revised application. The Critic is advisory, you cannot stop the task yourself, and only explicit human approval may end the task early.
When a task enters DESIGNING with prior data, injected_research_briefing summarizes measured history and prior decisions. Use query_research_index first, then call get_complete_batch_history only when the current research question requires all prior batch designs, recipes, observations, and fixed analysis/decision records. Complete history excludes raw spectrum arrays: use get_spectrum_data with a listed sample_id when a specific emission or absorption spectrum is needed. Treat measurements and CIE values as facts, and prior Agent designs, analyses, models, and hypotheses as interpretations.
For composition-class retrieval, use active_component_count=1 for single-component, active_component_count=2 for binary, active_component_count=3 for ternary, active_component_count=4 for quaternary, and active_component_count=5 for five-component samples. Do not pass Single, Binary, Ternary, Quaternary, or other category labels to design_role; design_role only exactly matches a complete stored recipe purpose.
Keep retrieval focused: use at most five research-index queries between substantive analysis or design actions, and use no more than ten consecutive read-only calls in total. After broad lookup, read only the specific records or spectra required by the current hypothesis, then act on the evidence with a diagnostic, analysis, design-decision, or follow-up-design tool. Do not keep changing query filters to repeat the same lookup; the runtime will return a failed observation and request another decision when a retrieval route is exhausted.
Absorption spectra are optional auxiliary evidence. Missing, partial, excluded, or invalid absorption must not block emission-based CIE analysis; only use full absorption features when the stored QC status is usable.
""".strip()

# Shared schema keeps every planning route explicit about its falsifiable recipe intent.
RECIPE_DISCRIMINATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "description": "The recipe-level, falsifiable expected observation."},
        "reference_sample_ids": {"type": "array", "items": {"type": "string"}},
        "outcome_if_supported": {"type": "string", "description": "How the next experiment strategy changes if the hypothesis is supported."},
        "outcome_if_not_supported": {"type": "string", "description": "How the next experiment strategy changes if the hypothesis is not supported."},
    },
    "required": ["hypothesis", "reference_sample_ids", "outcome_if_supported", "outcome_if_not_supported"],
    "additionalProperties": False,
}

OPTION_REASON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"option": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["option", "reason"],
    "additionalProperties": False,
}

UNREACHABLE_ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {"type": "string"},
        "observed_outcome": {"type": "string"},
        "reason": {"type": "string"},
        "evidence_sample_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["route", "observed_outcome", "reason", "evidence_sample_ids"],
    "additionalProperties": False,
}

UNREACHABLE_OPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "option": {"type": "string"},
        "reason": {"type": "string"},
        "evidence_sample_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["option", "reason", "evidence_sample_ids"],
    "additionalProperties": False,
}

CRITIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "finding_id": {"type": "string"},
        "disposition": {
            "type": "string",
            "enum": ["accepted", "partially_accepted", "rejected"],
        },
        "response": {"type": "string"},
    },
    "required": ["finding_id", "disposition", "response"],
    "additionalProperties": False,
}

UNREACHABLE_APPLICATION_PROPERTIES: dict[str, Any] = {
    "claim": {"type": "string"},
    "evidence_sample_ids": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "uniqueItems": True,
    },
    "attempted_routes": {
        "type": "array",
        "items": UNREACHABLE_ROUTE_SCHEMA,
        "minItems": 1,
    },
    "remaining_options": {
        "type": "array",
        "items": UNREACHABLE_OPTION_SCHEMA,
        "minItems": 1,
    },
    "reasoning": {"type": "string"},
    "uncertainties": {"type": "array", "items": {"type": "string"}, "minItems": 1},
}

UNREACHABLE_APPLICATION_REQUIRED = [
    "claim",
    "evidence_sample_ids",
    "attempted_routes",
    "remaining_options",
    "reasoning",
    "uncertainties",
]

FOLLOWUP_DRAFT_RECIPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "concentrations_mmol_ml": {
            "type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 5, "maxItems": 5
        },
        "purpose": {"type": "string"},
        "design_role": {"type": "string", "enum": ["primary", "competing"]},
        "discrimination": RECIPE_DISCRIMINATION_SCHEMA,
        "source_candidate_id": {"type": "string"},
    },
    "required": ["concentrations_mmol_ml", "purpose", "design_role", "discrimination"],
    "additionalProperties": False,
}

# These schemas expose the small tool contract to the model; runtime still owns task-local paths.
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "query_pubchem": {
        "description": "Confirm exactly one material through PubChem. Translate its Chinese name to an English chemical name before calling.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "English chemical name or synonym to query."},
                "material": {"type": "string", "description": "Original user-supplied material name."},
            },
            "required": ["name", "material"],
            "additionalProperties": False,
        },
    },
    "read_skill": {
        "description": "Activate one local scientific Skill before using its instructions in a later decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Registered local Skill name."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "search_crossref": {
        "description": "Find up to three references only after a confirmed PubChem identity has insufficient information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Confirmed English material name and missing-property keywords."},
                "identity_confirmed": {"type": "boolean", "const": True},
                "information_insufficient": {"type": "boolean", "const": True},
            },
            "required": ["query", "identity_confirmed", "information_insufficient"],
            "additionalProperties": False,
        },
    },
    "save_material_evidence": {
        "description": "Persist all successful PubChem identity queries already available for this task.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "design_initial_batch": {
        "description": "Create B1 from the Agent's own bounded exploratory recipe matrix after PubChem identities are confirmed. The runtime supplies target, evidence, and materials; provide only a rationale and 1-24 explicit five-slot recipes.",
        "parameters": {
            "type": "object",
            "properties": {
                "scientific_rationale": {"type": "string", "description": "Concise initial-screening rationale that separates known identity facts from unmeasured hypotheses."},
                "recipes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "properties": {
                            "concentrations_mmol_ml": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 5, "maxItems": 5},
                            "purpose": {"type": "string", "description": "Scientific role such as single-component anchor, binary screening probe, or targeted coverage hypothesis."},
                            "discrimination": RECIPE_DISCRIMINATION_SCHEMA,
                        },
                        "required": ["concentrations_mmol_ml", "purpose", "discrimination"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["scientific_rationale", "recipes"],
            "additionalProperties": False,
        },
    },
    "ingest_spectra": {
        "description": "Validate and archive the pending CPWL spectrum directory returned by the user.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "calculate_cie": {
        "description": "Calculate deterministic CIE 1931 2-degree xy values from qualified archived emission spectra.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "analyze_results": {
        "description": "Analyze the accepted measurements of the current experiment round.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "update_research_dataset": {
        "description": "Join the current plan, accepted spectra, CIE, and analysis into the canonical task-local research dataset and derived index.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "query_research_index": {
        "description": "Perform one focused lookup of compact CIE and recipe records from the current task dataset. Use active_component_count=1 for single-component, active_component_count=2 for binary, active_component_count=3 for ternary, active_component_count=4 for quaternary, and active_component_count=5 for five-component samples. Do not pass Single, Binary, Ternary, Quaternary, or other category labels to design_role; it is an exact match against the complete free-text recipe purpose. Use the result to make a scientific decision and do not repeat the same lookup with changing filters. Simulations are excluded unless explicitly requested.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_cie": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "max_distance": {"type": "number", "minimum": 0},
                "batch_id": {"type": "string"},
                "design_role": {"type": "string"},
                "active_component_count": {"type": "integer", "minimum": 1, "maximum": 5},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "include_synthetic": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "get_experiment_record": {
        "description": "Load one complete current-task observation using exactly one sample ID or recipe ID; simulations are excluded unless explicitly requested.",
        "parameters": {
            "type": "object",
            "properties": {
                "sample_id": {"type": "string"},
                "recipe_id": {"type": "string"},
                "include_synthetic": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "get_spectrum_data": {
        "description": "Load archived emission or absorption points for one indexed sample. Never accepts a filesystem path.",
        "parameters": {
            "type": "object",
            "properties": {
                "sample_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["emission", "absorption"]},
                "start_nm": {"type": "number"},
                "end_nm": {"type": "number"},
                "include_synthetic": {"type": "boolean"},
            },
            "required": ["sample_id", "kind"],
            "additionalProperties": False,
        },
    },
    "extract_spectral_features": {
        "description": "Extract deterministic descriptors from one to twelve measured emission or absorption spectra for a concrete diagnostic question. Use it before treating FRET, energy transfer, quenching, cascade, emission dominance, peak shift, or spectral-shape change as measured evidence in a route decision. It never accepts a file path, predicts CIE, or proves an optical mechanism.",
        "parameters": {
            "type": "object",
            "properties": {
                "sample_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12},
                "kind": {"type": "string", "enum": ["emission", "absorption"]},
                "band_nm": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "compare_to_sample_id": {"type": "string"},
            },
            "required": ["sample_ids", "kind"],
            "additionalProperties": False,
        },
    },
    "review_design_outcomes": {
        "description": "Create a factual review of the just-completed measured batch by comparing each recipe with its declared measured references. It reports CIE and target-distance changes, never determines a natural-language hypothesis or mechanism.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "diagnose_dataset": {
        "description": "First step of the measured-analysis workflow: diagnose CIE coverage and limitations before composition screening or response modeling. Run before screen_composition_effects and compile_research_analysis.",
        "parameters": {
            "type": "object",
            "properties": {"batch_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    "screen_composition_effects": {
        "description": "Second step of the measured-analysis workflow: screen measured single-component anchors and binary probes for observed CIE directions without fitting a predictive model. Run after diagnose_dataset and before compile_research_analysis.",
        "parameters": {
            "type": "object",
            "properties": {"batch_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    "compile_research_analysis": {
        "description": "Third step of the measured-analysis workflow: combine diagnosis and composition screening into a reproducible first-layer research_analysis artifact. Run after diagnose_dataset and screen_composition_effects, and before any local response model.",
        "parameters": {
            "type": "object",
            "properties": {
                "research_question": {
                    "type": "string",
                    "description": "One concise scientific question answered by this diagnostic analysis."
                }
            },
            "required": ["research_question"],
            "additionalProperties": False,
        },
    },
    "fit_local_response_model": {
        "description": "Fit one bounded measured-only local CIE model only after diagnosis, composition screening, and first-layer research_analysis are complete. Use one or two active material slots and a controlled sample-ID scope; returns LOOCV evidence, not a candidate recipe.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["local_ridge", "weighted_neighbors"]},
                "active_slots": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
                "sample_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "ridge_alpha": {"type": "number", "exclusiveMinimum": 0},
                "neighbor_count": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["method", "active_slots"],
            "additionalProperties": False,
        },
    },
    "compare_models": {
        "description": "Compare at least two successfully fitted local model artifacts by LOOCV. It ranks evidence but does not choose a scientific strategy.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "write_design_decision": {
        "description": "Persist the Agent's auditable strategy selection, cited measured facts, hypotheses, and next measurement purpose. A selected response model requires completed analysis and comparison; selected_method=none may record a focused direct-evidence exploratory decision. Express every rejected method or strategy as an option/reason object.",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": ["coverage", "diagnostic", "local_optimization", "repeat_validation"]},
                "research_question": {"type": "string"},
                "facts_used_sample_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "working_hypotheses": {"type": "array", "items": {"type": "string"}},
                "selected_method": {"type": "string", "enum": ["local_ridge", "weighted_neighbors", "none"]},
                "selection_reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "rejected_options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["option", "reason"],
                        "additionalProperties": False,
                    },
                },
                "next_measurement_purpose": {"type": "string"},
            },
            "required": ["strategy", "research_question", "facts_used_sample_ids", "selected_method", "selection_reasons", "next_measurement_purpose"],
            "additionalProperties": False,
        },
    },
    "generate_predicted_candidates": {
        "description": "Reconstruct the Agent-selected supported local model and generate a small grid of predicted recipes inside measured local coverage. Predictions cannot verify the target and only interpolation candidates may later be selected.",
        "parameters": {
            "type": "object",
            "properties": {
                "grid_size": {"type": "integer", "minimum": 3, "maximum": 11},
            },
            "additionalProperties": False,
        },
    },
    "design_followup_batch": {
        "description": "Create the next CPWL XLSX plan from one to twelve listed unmeasured interpolation candidate IDs. Supply a concise scientific selection reason and one measured-reference discrimination record per candidate; the resulting samples remain predicted until laboratory spectra return.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12},
                "candidate_discriminations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "discrimination": RECIPE_DISCRIMINATION_SCHEMA,
                        },
                        "required": ["candidate_id", "discrimination"],
                        "additionalProperties": False,
                    },
                },
                "selection_reason": {"type": "string"},
            },
            "required": ["candidate_ids", "candidate_discriminations", "selection_reason"],
            "additionalProperties": False,
        },
    },
    "design_exploratory_followup_batch": {
        "description": "Create a B{round+1} CPWL plan from an Agent decision with selected_method=none. The strategy, reference sample, active slots, and recipes are validated against measured data; this is exploratory, never predictive proof.",
        "parameters": {
            "type": "object",
            "properties": {
                "reference_sample_id": {"type": "string", "description": "Measured sample cited in the saved decision as the controlled reference."},
                "active_slots": {"type": "array", "items": {"type": "string", "enum": ["A", "B", "C", "D", "E"]}, "minItems": 1, "maxItems": 2},
                "recipes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "concentrations_mmol_ml": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 5, "maxItems": 5},
                            "discrimination": RECIPE_DISCRIMINATION_SCHEMA,
                        },
                        "required": ["concentrations_mmol_ml", "discrimination"],
                        "additionalProperties": False,
                    },
                },
                "selection_reason": {"type": "string", "description": "Concise reason that connects the selected controlled recipes to the saved scientific question."},
            },
            "required": ["reference_sample_id", "active_slots", "recipes", "selection_reason"],
            "additionalProperties": False,
        },
    },
    "propose_followup_batch": {
        "description": "Save a complete B2-or-later Director draft and deterministically evaluate its actual recipes before one automatic Scientific Critic review. Before calling, precompute every A-E stock volume and solvent volume under the fixed 8 ml/0.0001 mmol/ml contract: each must be <= 4.8 ml, total concentration must be within 0.00004..0.0001 mmol/ml, and each component concentration must be <= 0.00006 mmol/ml. Use 2-12 recipes and include both primary and scientifically supported competing roles; do not add routine validation merely to fill the batch. If validation fails, use the returned per-recipe volume details to correct and resubmit the complete draft.",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["local_refinement", "model_guided", "component_adjustment", "region_switch", "concentration_scan", "multi_component_search", "diagnostic"],
                },
                "research_question": {"type": "string"},
                "facts_used_sample_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "primary_option": {"type": "string"},
                "competing_option": {"type": "string"},
                "rejected_options": {"type": "array", "items": OPTION_REASON_SCHEMA},
                "allocation_reason": {"type": "string"},
                "recipes": {"type": "array", "items": FOLLOWUP_DRAFT_RECIPE_SCHEMA, "minItems": 2, "maxItems": 12},
            },
            "required": ["strategy", "research_question", "facts_used_sample_ids", "primary_option", "competing_option", "allocation_reason", "recipes"],
            "additionalProperties": False,
        },
    },
    "finalize_followup_batch": {
        "description": "Respond to every injected Scientific Critic finding and submit the complete final B2-or-later design. Recompute every recipe's A-E stock and solvent volumes before submitting: product is 8 ml, stock is 0.0001 mmol/ml, every volume is <= 4.8 ml, total concentration is within 0.00004..0.0001 mmol/ml, and each component concentration is <= 0.00006 mmol/ml. If validation fails, use the returned recipe/slot details to correct the complete final object without silently changing its scientific intent. Director retains final authority. This one object generates the decision record, final design JSON, CPWL XLSX, and measurement-return directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["local_refinement", "model_guided", "component_adjustment", "region_switch", "concentration_scan", "multi_component_search", "diagnostic"],
                },
                "research_question": {"type": "string"},
                "facts_used_sample_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "primary_option": {"type": "string"},
                "competing_option": {"type": "string"},
                "rejected_options": {"type": "array", "items": OPTION_REASON_SCHEMA},
                "allocation_reason": {"type": "string"},
                "recipes": {"type": "array", "items": FOLLOWUP_DRAFT_RECIPE_SCHEMA, "minItems": 2, "maxItems": 12},
                "critic_responses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "finding_id": {"type": "string"},
                            "disposition": {"type": "string", "enum": ["accepted", "partially_accepted", "rejected"]},
                            "response": {"type": "string"},
                        },
                        "required": ["finding_id", "disposition", "response"],
                        "additionalProperties": False,
                    },
                },
                "final_reason": {"type": "string"},
            },
            "required": ["strategy", "research_question", "facts_used_sample_ids", "primary_option", "competing_option", "allocation_reason", "recipes", "critic_responses", "final_reason"],
            "additionalProperties": False,
        },
    },
    "propose_unreachable_request": {
        "description": "Propose a measured-evidence-only, bounded application that further experiments are not justified for the current materials, explored scope, constraints, and budget. This saves a draft and factual coverage evaluation, then triggers one automatic Unreachable Scientific Critic review. It does not stop the task.",
        "parameters": {
            "type": "object",
            "properties": UNREACHABLE_APPLICATION_PROPERTIES,
            "required": UNREACHABLE_APPLICATION_REQUIRED,
            "additionalProperties": False,
        },
    },
    "continue_after_unreachable_review": {
        "description": "After the injected Unreachable Scientific Critic review, withdraw the bounded application and continue experiment design with one concrete research question and measurement purpose.",
        "parameters": {
            "type": "object",
            "properties": {
                "critic_responses": {"type": "array", "items": CRITIC_RESPONSE_SCHEMA},
                "final_reason": {"type": "string"},
                "next_research_question": {"type": "string"},
                "next_measurement_purpose": {"type": "string"},
            },
            "required": ["critic_responses", "final_reason", "next_research_question", "next_measurement_purpose"],
            "additionalProperties": False,
        },
    },
    "submit_unreachable_application": {
        "description": "After the injected Unreachable Scientific Critic review, submit the complete revised bounded application for explicit human approval. This enters AWAITING_HUMAN_REVIEW but cannot stop the task by itself.",
        "parameters": {
            "type": "object",
            "properties": {
                **UNREACHABLE_APPLICATION_PROPERTIES,
                "critic_responses": {"type": "array", "items": CRITIC_RESPONSE_SCHEMA},
                "final_reason": {"type": "string"},
            },
            "required": [*UNREACHABLE_APPLICATION_REQUIRED, "critic_responses", "final_reason"],
            "additionalProperties": False,
        },
    },
    "get_research_briefing": {
        "description": "Rebuild the compact measured-data history briefing for the current task. Runtime also injects this briefing automatically at the start of a new design stage.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "get_complete_batch_history": {
        "description": "On demand, read every completed batch in the current task, including its Agent design rationale, recipes, measured observations, and fixed analysis/decision records. Call only when the current research question needs a whole-batch historical view. Raw spectrum arrays remain available only through get_spectrum_data.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "get_analysis_record": {
        "description": "Read one complete historical analysis record by round and kind. It never accepts an artifact path and excludes synthetic rounds by default.",
        "parameters": {
            "type": "object",
            "properties": {
                "round": {"type": "integer", "minimum": 1},
                "kind": {"type": "string", "enum": ["research_analysis", "dataset_diagnosis", "composition_effects", "model_comparison", "design_decision", "predicted_candidates", "candidate_selection", "exploratory_selection", "design_outcome_review", "models"]},
                "include_synthetic": {"type": "boolean"},
            },
            "required": ["round", "kind"],
            "additionalProperties": False,
        },
    },
    "check_goal": {
        "description": "Use the deterministic CIE tolerance check on the current analysis result. A first hit becomes provisional; an exact independently prepared repeat must also pass before the task finishes.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
