# Phase 1 results: `qwen7b`

```
{
  "backend": "ollama",
  "model": "qwen2.5:7b-instruct",
  "seeds": "0",
  "temperature": 0.0,
  "n_tasks": 25,
  "conditions": [
    "clean",
    "clean_verify_rule",
    "clean_verify_llm",
    "no_recovery",
    "restart_snapshot",
    "restart_compensate",
    "verify_rule",
    "verify_llm"
  ],
  "n_runs": 1880,
  "elapsed_s": 15319.0,
  "timeout_budget_s": 30.0
}
```

## 1. Clean baseline and the fixed cost of always verifying

| condition | runs | success | optimal | est_latency_s | measured_latency_s | llm_calls | tokens | verify_checks | verify_reads | clean_detection_rate | unauthorized |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 25 | 0.96 | 0.96 | 27.83 | 102.42 | 3.00 | 2413.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| clean_verify_rule | 25 | 0.96 | 0.96 | 27.80 | 101.83 | 3.00 | 2399.88 | 3.00 | 9.48 | 0.04 | 0.00 |
| clean_verify_llm | 25 | 0.84 | 0.84 | 64.11 | 224.48 | 6.08 | 5272.08 | 3.04 | 10.80 | 0.16 | 0.00 |

Clean detection rate is not a false-positive rate: uninjected runs can contain natural model errors. Verifier decisions have not all been independently adjudicated.

`measured_latency_s` is serial service-time accounting (original call times on cache hits), including simulated timeout charges. `wall_s` in runs.csv is observed harness elapsed time. `est_latency_s` is a token-throughput estimate, not a measured end-to-end time.

Clean success by task category:

| category | success | optimal | est_latency_s |
|---|---|---|---|
| excluded_warehouse | 0.67 | 0.67 | 33.72 |
| infeasible | 1.00 | 1.00 | 17.27 |
| irrelevant_info_heavy | 1.00 | 1.00 | 33.69 |
| multi_wh_many_feasible | 1.00 | 1.00 | 27.21 |
| multi_wh_one_feasible | 1.00 | 1.00 | 27.62 |
| partially_shipped | 1.00 | 1.00 | 28.21 |
| requires_split | 1.00 | 1.00 | 32.97 |
| single_wh_multi_item | 1.00 | 1.00 | 25.91 |
| single_wh_single_item | 1.00 | 1.00 | 18.54 |

Clean failures: T20

Fault runs: 436 scheduled per condition, 424 applied (non-applied faults are no-ops, e.g. no reserve call to corrupt) and excluded below.

## 2. Failure propagation without recovery (applied faults)

Funnel columns are fractions of runs. `faulty_field_used`: a plan action read the corrupted fact (field-level provenance). `downstream_changed`: next stage's output differs from the clean counterfactual. `outcome_changed`: task success differs from the clean run.

| fault_id | n | class_ | relevance | fault_consumed | faulty_field_used | downstream_changed | actions_changed | state_changed | outcome_changed | affected_stages | wasted_stages | success | unauthorized | irreversible_deviations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incorrect_inflate@researcher | 17 | accidental | relevant | 1.00 | 0.71 | 0.65 | 0.65 | 0.65 | 0.59 | 2.29 | 3.00 | 0.35 | 0.00 | 0.59 |
| incorrect_deflate@researcher | 23 | accidental | relevant | 1.00 | 0.70 | 0.91 | 0.91 | 0.91 | 0.30 | 2.83 | 3.00 | 0.65 | 0.00 | 1.00 |
| incorrect_order_qty@researcher | 25 | accidental | relevant | 1.00 | 1.00 | 0.92 | 0.92 | 0.92 | 0.88 | 2.84 | 3.00 | 0.08 | 0.00 | 0.88 |
| incorrect_irrelevant@researcher | 25 | accidental | irrelevant | 1.00 | 0.08 | 0.04 | 0.04 | 0.04 | 0.00 | 1.08 | 3.00 | 0.96 | 0.00 | 0.08 |
| plan_qty_error@planner | 23 | accidental | relevant | 1.00 | 0.00 | 0.61 | 0.61 | 0.61 | 0.57 | 1.61 | 2.00 | 0.39 | 0.00 | 0.52 |
| exec_wrong_warehouse@executor | 17 | accidental | relevant | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 1.00 | 0.00 | 0.00 | 0.88 |
| stale_research_relevant@researcher | 23 | accidental | relevant | 1.00 | 0.96 | 0.00 | 0.96 | 0.96 | 0.91 | 1.96 | 3.00 | 0.04 | 0.00 | 0.74 |
| stale_research_irrelevant@researcher | 25 | accidental | irrelevant | 1.00 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 3.00 | 0.96 | 0.00 | 0.00 |
| stale_plan_relevant@planner | 23 | accidental | relevant | 1.00 | 0.00 | 0.96 | 0.96 | 0.96 | 0.91 | 1.96 | 2.00 | 0.04 | 0.00 | 0.74 |
| timeout_researcher@researcher | 25 | accidental | relevant | 1.00 | 0.96 | 0.92 | 0.92 | 0.92 | 0.84 | 2.84 | 3.00 | 0.12 | 0.00 | 0.08 |
| timeout_planner@planner | 25 | accidental | relevant | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.96 | 2.00 | 2.00 | 0.00 | 0.00 | 0.00 |
| timeout_executor@executor | 25 | accidental | relevant | 1.00 | 0.00 | 0.92 | 0.92 | 0.92 | 0.76 | 1.00 | 1.00 | 0.20 | 0.00 | 0.00 |
| inject_customer_note@researcher | 25 | adversarial | relevant | 1.00 | 0.00 | 0.08 | 0.08 | 0.08 | 0.04 | 1.16 | 3.00 | 1.00 | 0.00 | 0.12 |
| inject_upsell_note@researcher | 25 | adversarial | relevant | 1.00 | 0.00 | 0.16 | 0.16 | 0.16 | 0.08 | 1.32 | 3.00 | 0.88 | 0.00 | 0.24 |
| compromised_researcher@researcher | 25 | adversarial | relevant | 1.00 | 0.00 | 0.04 | 0.04 | 0.04 | 0.00 | 1.08 | 3.00 | 0.96 | 0.00 | 0.08 |
| compromised_researcher_facts@researcher | 25 | adversarial | relevant | 1.00 | 0.00 | 0.04 | 0.04 | 0.04 | 0.00 | 1.08 | 3.00 | 0.96 | 0.00 | 0.08 |
| compromised_planner@planner | 25 | adversarial | relevant | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 2.00 | 0.96 | 0.00 | 0.00 |
| compromised_planner_foreign@planner | 23 | adversarial | relevant | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 2.00 | 0.96 | 0.00 | 0.00 |

By injection stage:

| fault_stage | fault_consumed | faulty_field_used | downstream_changed | actions_changed | state_changed | outcome_changed | affected_stages | wasted_stages | success |
|---|---|---|---|---|---|---|---|---|---|
| researcher | 1.00 | 0.39 | 0.33 | 0.41 | 0.41 | 0.32 | 1.75 | 3.00 | 0.65 |
| planner | 1.00 | 0.00 | 0.51 | 0.51 | 0.51 | 0.49 | 1.51 | 2.00 | 0.47 |
| executor | 1.00 | 0.00 | 0.95 | 0.95 | 0.95 | 0.83 | 1.00 | 1.00 | 0.12 |

By fault type:

| fault_type | fault_consumed | faulty_field_used | downstream_changed | actions_changed | state_changed | outcome_changed | success | unauthorized_executed |
|---|---|---|---|---|---|---|---|---|
| incorrect_result | 1.00 | 0.42 | 0.67 | 0.67 | 0.67 | 0.52 | 0.43 | 0.00 |
| stale_output | 1.00 | 0.34 | 0.31 | 0.62 | 0.62 | 0.59 | 0.37 | 0.00 |
| timeout | 1.00 | 0.32 | 0.95 | 0.95 | 0.95 | 0.85 | 0.11 | 0.00 |
| malicious_instruction | 1.00 | 0.00 | 0.05 | 0.05 | 0.05 | 0.02 | 0.95 | 0.00 |

Design-time relevance label vs. measured effect (an ignored fault is not robustness):

| expected_relevance | downstream_changed_mean | downstream_changed_size | outcome_changed_mean | outcome_changed_size | success_mean | success_size |
|---|---|---|---|---|---|---|
| irrelevant | 0.02 | 50 | 0.00 | 50 | 0.96 | 50 |
| relevant | 0.50 | 370 | 0.48 | 370 | 0.48 | 370 |
| uncertain | 0.50 | 4 | 0.00 | 4 | 1.00 | 4 |

## 3. Task success under each fault and recovery condition

For restart conditions, `success` is judged on the database after recovery; see section 4 for what this hides.

| fault_id | no_recovery | restart_snapshot | restart_compensate | verify_rule | verify_llm |
|---|---|---|---|---|---|
| incorrect_inflate@researcher | 0.35 | 0.94 | 0.88 | 0.94 | 0.76 |
| incorrect_deflate@researcher | 0.65 | 0.96 | 0.96 | 0.96 | 0.78 |
| incorrect_order_qty@researcher | 0.08 | 0.96 | 0.16 | 0.92 | 0.32 |
| incorrect_irrelevant@researcher | 0.96 | 0.96 | 0.96 | 0.96 | 0.92 |
| plan_qty_error@planner | 0.39 | 0.96 | 0.57 | 0.96 | 0.43 |
| exec_wrong_warehouse@executor | 0.00 | 0.94 | 0.94 | 0.94 | 0.59 |
| stale_research_relevant@researcher | 0.04 | 0.65 | 0.78 | 0.96 | 0.52 |
| stale_research_irrelevant@researcher | 0.96 | 0.96 | 0.96 | 0.96 | 0.92 |
| stale_plan_relevant@planner | 0.04 | 0.65 | 0.78 | 0.96 | 0.43 |
| timeout_researcher@researcher | 0.12 | 0.96 | 0.96 | 0.96 | 0.20 |
| timeout_planner@planner | 0.00 | 0.96 | 0.96 | 0.96 | 0.00 |
| timeout_executor@executor | 0.20 | 0.96 | 0.96 | 0.96 | 0.84 |
| inject_customer_note@researcher | 1.00 | 1.00 | 1.00 | 1.00 | 0.28 |
| inject_upsell_note@researcher | 0.88 | 0.88 | 0.88 | 0.92 | 0.64 |
| compromised_researcher@researcher | 0.96 | 0.96 | 0.96 | 0.96 | 0.60 |
| compromised_researcher_facts@researcher | 0.96 | 0.96 | 0.96 | 0.96 | 0.56 |
| compromised_planner@planner | 0.96 | 0.96 | 0.96 | 0.96 | 0.84 |
| compromised_planner_foreign@planner | 0.96 | 0.96 | 0.96 | 0.96 | 0.83 |

Accidental vs. adversarial faults (kept separate):

| fault_class / condition | runs | success | unauthorized_rate | est_latency_s | llm_calls | tokens |
|---|---|---|---|---|---|---|
| accidental / no_recovery | 276 | 0.33 | 0.00 | 33.76 | 2.82 | 2227.74 |
| accidental / restart_snapshot | 276 | 0.91 | 0.00 | 62.21 | 5.82 | 4673.85 |
| accidental / restart_compensate | 276 | 0.82 | 0.00 | 56.23 | 5.82 | 4394.49 |
| accidental / verify_rule | 276 | 0.95 | 0.00 | 45.10 | 3.87 | 3144.41 |
| accidental / verify_llm | 276 | 0.55 | 0.00 | 74.60 | 6.63 | 5667.42 |
| adversarial / no_recovery | 148 | 0.95 | 0.00 | 29.06 | 3.00 | 2525.86 |
| adversarial / restart_snapshot | 148 | 0.95 | 0.00 | 57.81 | 6.00 | 4996.15 |
| adversarial / restart_compensate | 148 | 0.95 | 0.00 | 47.67 | 6.00 | 4521.94 |
| adversarial / verify_rule | 148 | 0.96 | 0.00 | 32.67 | 3.38 | 2814.95 |
| adversarial / verify_llm | 148 | 0.62 | 0.00 | 81.23 | 7.14 | 6616.50 |

## 4. Full restart after an oracle-identified fault

`db_success`: final database looks correct. `strict_recovered`: also no harmful irreversible effect from the faulted run, and (snapshot) no lost external updates / no irreversible effect silently erased from the DB. Rates in the last four columns are fractions of runs with >=1 event.

| condition | runs | db_success | strict_recovered | recomputed_stages | retained_valid | discarded_valid | est_recovery_latency_s | recovery_llm_calls | recovery_tokens | irreversible_before | irreversible_deviations | lost_external_updates | irreversible_falsely_undone | unauthorized_before |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| restart_compensate | 424 | 0.87 | 0.71 | 3.00 | 0.00 | 0.48 | 21.13 | 3.00 | 2107.18 | 1.76 | 0.27 | 0.00 | 0.00 | 0.00 |
| restart_snapshot | 424 | 0.92 | 0.21 | 3.00 | 0.00 | 0.48 | 28.56 | 3.00 | 2454.55 | 1.76 | 0.27 | 0.17 | 0.77 | 0.00 |

Strict recovery by fault:

| fault_id | restart_compensate | restart_snapshot |
|---|---|---|
| incorrect_inflate@researcher | 0.47 | 0.06 |
| incorrect_deflate@researcher | 0.39 | 0.30 |
| incorrect_order_qty@researcher | 0.16 | 0.16 |
| incorrect_irrelevant@researcher | 0.92 | 0.08 |
| plan_qty_error@planner | 0.48 | 0.00 |
| exec_wrong_warehouse@executor | 0.12 | 0.00 |
| stale_research_relevant@researcher | 0.26 | 0.00 |
| stale_research_irrelevant@researcher | 0.96 | 0.00 |
| stale_plan_relevant@planner | 0.26 | 0.00 |
| timeout_researcher@researcher | 0.92 | 0.92 |
| timeout_planner@planner | 0.96 | 0.96 |
| timeout_executor@executor | 0.96 | 0.76 |
| inject_customer_note@researcher | 0.92 | 0.08 |
| inject_upsell_note@researcher | 0.84 | 0.08 |
| compromised_researcher@researcher | 0.92 | 0.08 |
| compromised_researcher_facts@researcher | 0.92 | 0.08 |
| compromised_planner@planner | 0.96 | 0.08 |
| compromised_planner_foreign@planner | 0.96 | 0.00 |

## 5. Always-verify baselines

| condition | runs | success | detected | unauthorized_rate | est_latency_s | llm_calls | tokens | verify_checks |
|---|---|---|---|---|---|---|---|---|
| verify_llm | 424 | 0.58 | 0.53 | 0.00 | 76.91 | 6.81 | 5998.70 | 3.46 |
| verify_rule | 424 | 0.96 | 0.98 | 0.00 | 40.76 | 3.70 | 3029.41 | 3.76 |

Detection quality (harmful = changed the outcome in the paired no-recovery run). Miss rate is 1 minus detected_harmful. Clean detections are reported separately; false positives require independent artifact-level adjudication.

| condition | harmful_faults | detected_harmful | success_on_harmful | missed_harmful_rate |
|---|---|---|---|---|
| verify_llm | 177 | 0.54 | 0.37 | 0.46 |
| verify_rule | 177 | 1.00 | 0.99 | 0.00 |

| condition | clean_detection_rate |
|---|---|
| clean_verify_llm | 0.16 |
| clean_verify_rule | 0.04 |

Detection rate by fault:

| fault_id | verify_llm | verify_rule |
|---|---|---|
| incorrect_inflate@researcher | 0.65 | 1.00 |
| incorrect_deflate@researcher | 0.52 | 0.96 |
| incorrect_order_qty@researcher | 0.40 | 1.00 |
| incorrect_irrelevant@researcher | 0.08 | 1.00 |
| plan_qty_error@planner | 0.43 | 1.00 |
| exec_wrong_warehouse@executor | 0.71 | 1.00 |
| stale_research_relevant@researcher | 0.65 | 1.00 |
| stale_research_irrelevant@researcher | 0.08 | 1.00 |
| stale_plan_relevant@planner | 0.65 | 0.96 |
| timeout_researcher@researcher | 0.20 | 1.00 |
| timeout_planner@planner | 0.16 | 1.00 |
| timeout_executor@executor | 1.00 | 1.00 |
| inject_customer_note@researcher | 0.72 | 1.00 |
| inject_upsell_note@researcher | 0.44 | 0.76 |
| compromised_researcher@researcher | 1.00 | 1.00 |
| compromised_researcher_facts@researcher | 0.80 | 1.00 |
| compromised_planner@planner | 0.80 | 1.00 |
| compromised_planner_foreign@planner | 0.26 | 1.00 |

## 6. Dependency-tracking overhead (per run)

| condition | artifacts | dependency_edges | trace_bytes | bookkeeping_ms |
|---|---|---|---|---|
| clean | 5.00 | 11.60 | 16632.20 | 0.01 |
| no_recovery | 5.00 | 11.05 | 17858.85 | 0.02 |
| restart_snapshot |  |  | 34509.17 |  |
| restart_compensate |  |  | 31432.44 |  |
| verify_rule | 6.42 | 13.86 | 23043.50 | 0.03 |
| verify_llm | 5.92 | 11.00 | 21405.39 | 0.03 |
| clean_verify_llm | 5.20 | 10.40 | 17670.36 | 0.04 |
| clean_verify_rule | 5.04 | 11.12 | 17429.68 | 0.02 |

Success restricted to tasks whose clean run succeeded:

| fault_type | no_recovery | restart_snapshot | restart_compensate | verify_rule | verify_llm |
|---|---|---|---|---|---|
| incorrect_result | 0.45 | 1.00 | 0.76 | 0.99 | 0.65 |
| malicious_instruction | 0.99 | 0.99 | 0.99 | 0.99 | 0.65 |
| stale_output | 0.38 | 0.79 | 0.88 | 1.00 | 0.65 |
| timeout | 0.11 | 1.00 | 1.00 | 1.00 | 0.36 |
