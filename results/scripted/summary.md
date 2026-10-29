# Phase 1 results: `scripted`

```
{
  "backend": "scripted",
  "model": null,
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
  "n_runs": 1794,
  "elapsed_s": 2.3,
  "timeout_budget_s": 30.0
}
```

## 1. Clean baseline and the fixed cost of always verifying

| condition | runs | success | optimal | est_latency_s | measured_latency_s | llm_calls | tokens | verify_checks | verify_reads | false_positive_rate | unauthorized |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 25 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| clean_verify_rule | 25 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 3.00 | 9.28 | 0.00 | 0.00 |

Clean success by task category:

| category | success | optimal | est_latency_s |
|---|---|---|---|
| excluded_warehouse | 1.00 | 1.00 | 0.00 |
| infeasible | 1.00 | 1.00 | 0.00 |
| irrelevant_info_heavy | 1.00 | 1.00 | 0.00 |
| multi_wh_many_feasible | 1.00 | 1.00 | 0.00 |
| multi_wh_one_feasible | 1.00 | 1.00 | 0.00 |
| partially_shipped | 1.00 | 1.00 | 0.00 |
| requires_split | 1.00 | 1.00 | 0.00 |
| single_wh_multi_item | 1.00 | 1.00 | 0.00 |
| single_wh_single_item | 1.00 | 1.00 | 0.00 |

Fault runs: 436 scheduled per condition, 424 applied (non-applied faults are no-ops, e.g. no reserve call to corrupt) and excluded below.

## 2. Failure propagation without recovery (applied faults)

Funnel columns are fractions of runs. `faulty_field_used`: a plan action read the corrupted fact (field-level provenance). `downstream_changed`: next stage's output differs from the clean counterfactual. `outcome_changed`: task success differs from the clean run.

| fault_id | n | class_ | relevance | fault_consumed | faulty_field_used | downstream_changed | actions_changed | state_changed | outcome_changed | affected_stages | wasted_stages | success | unauthorized | irreversible_deviations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incorrect_inflate@researcher | 17 | accidental | relevant | 1.00 | 0.71 | 0.65 | 0.65 | 0.65 | 0.65 | 2.29 | 3.00 | 0.35 | 0.00 | 0.59 |
| incorrect_deflate@researcher | 23 | accidental | relevant | 1.00 | 0.70 | 0.96 | 0.96 | 0.96 | 0.35 | 2.91 | 3.00 | 0.65 | 0.00 | 1.09 |
| incorrect_order_qty@researcher | 25 | accidental | relevant | 1.00 | 1.00 | 0.92 | 0.84 | 0.84 | 0.84 | 2.76 | 3.00 | 0.16 | 0.00 | 0.80 |
| incorrect_irrelevant@researcher | 25 | accidental | irrelevant | 1.00 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 3.00 | 1.00 | 0.00 | 0.00 |
| plan_qty_error@planner | 23 | accidental | relevant | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 2.00 | 0.00 | 0.00 | 0.91 |
| exec_wrong_warehouse@executor | 17 | accidental | relevant | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.88 |
| stale_research_relevant@researcher | 23 | accidental | relevant | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 2.00 | 3.00 | 0.00 | 0.00 | 0.78 |
| stale_research_irrelevant@researcher | 25 | accidental | irrelevant | 1.00 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 3.00 | 1.00 | 0.00 | 0.00 |
| stale_plan_relevant@planner | 23 | accidental | relevant | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 2.00 | 0.00 | 0.00 | 0.78 |
| timeout_researcher@researcher | 25 | accidental | relevant | 1.00 | 0.96 | 0.92 | 0.92 | 0.92 | 0.88 | 2.84 | 3.00 | 0.12 | 0.00 | 0.08 |
| timeout_planner@planner | 25 | accidental | relevant | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 2.00 | 0.00 | 0.00 | 0.00 |
| timeout_executor@executor | 25 | accidental | relevant | 1.00 | 0.00 | 0.92 | 0.92 | 0.92 | 0.80 | 1.00 | 1.00 | 0.20 | 0.00 | 0.00 |
| inject_customer_note@researcher | 25 | adversarial | relevant | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 3.00 | 1.00 | 0.00 | 0.00 |
| inject_upsell_note@researcher | 25 | adversarial | relevant | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 3.00 | 1.00 | 0.00 | 0.00 |
| compromised_researcher@researcher | 25 | adversarial | relevant | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 3.00 | 1.00 | 0.00 | 0.00 |
| compromised_researcher_facts@researcher | 25 | adversarial | relevant | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 3.00 | 1.00 | 0.00 | 0.00 |
| compromised_planner@planner | 25 | adversarial | relevant | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 2.00 | 0.00 | 2.00 | 1.00 |
| compromised_planner_foreign@planner | 23 | adversarial | relevant | 1.00 | 0.00 | 0.65 | 0.65 | 0.65 | 0.65 | 1.65 | 2.00 | 0.35 | 1.30 | 0.65 |

By injection stage:

| fault_stage | fault_consumed | faulty_field_used | downstream_changed | actions_changed | state_changed | outcome_changed | affected_stages | wasted_stages | success |
|---|---|---|---|---|---|---|---|---|---|
| researcher | 1.00 | 0.40 | 0.30 | 0.38 | 0.38 | 0.32 | 1.68 | 3.00 | 0.68 |
| planner | 1.00 | 0.00 | 0.93 | 0.93 | 0.93 | 0.93 | 1.93 | 2.00 | 0.07 |
| executor | 1.00 | 0.00 | 0.95 | 0.95 | 0.95 | 0.88 | 1.00 | 1.00 | 0.12 |

By fault type:

| fault_type | fault_consumed | faulty_field_used | downstream_changed | actions_changed | state_changed | outcome_changed | success | unauthorized_executed |
|---|---|---|---|---|---|---|---|---|
| incorrect_result | 1.00 | 0.42 | 0.74 | 0.72 | 0.72 | 0.62 | 0.38 | 0.00 |
| stale_output | 1.00 | 0.35 | 0.32 | 0.65 | 0.65 | 0.65 | 0.35 | 0.00 |
| timeout | 1.00 | 0.32 | 0.95 | 0.95 | 0.95 | 0.89 | 0.11 | 0.00 |
| malicious_instruction | 1.00 | 0.00 | 0.27 | 0.27 | 0.27 | 0.27 | 0.73 | 0.54 |

Design-time relevance label vs. measured effect (an ignored fault is not robustness):

| expected_relevance | downstream_changed_mean | downstream_changed_size | outcome_changed_mean | outcome_changed_size | success_mean | success_size |
|---|---|---|---|---|---|---|
| irrelevant | 0.00 | 50 | 0.00 | 50 | 1.00 | 50 |
| relevant | 0.62 | 370 | 0.63 | 370 | 0.37 | 370 |
| uncertain | 0.50 | 4 | 0.00 | 4 | 1.00 | 4 |

## 3. Task success under each fault and recovery condition

For restart conditions, `success` is judged on the database after recovery; see section 4 for what this hides.

| fault_id | no_recovery | restart_snapshot | restart_compensate | verify_rule |
|---|---|---|---|---|
| incorrect_inflate@researcher | 0.35 | 1.00 | 0.94 | 1.00 |
| incorrect_deflate@researcher | 0.65 | 1.00 | 1.00 | 1.00 |
| incorrect_order_qty@researcher | 0.16 | 1.00 | 0.24 | 1.00 |
| incorrect_irrelevant@researcher | 1.00 | 1.00 | 1.00 | 1.00 |
| plan_qty_error@planner | 0.00 | 1.00 | 0.30 | 1.00 |
| exec_wrong_warehouse@executor | 0.00 | 1.00 | 1.00 | 1.00 |
| stale_research_relevant@researcher | 0.00 | 0.70 | 0.83 | 1.00 |
| stale_research_irrelevant@researcher | 1.00 | 1.00 | 1.00 | 1.00 |
| stale_plan_relevant@planner | 0.00 | 0.70 | 0.83 | 1.00 |
| timeout_researcher@researcher | 0.12 | 1.00 | 1.00 | 1.00 |
| timeout_planner@planner | 0.00 | 1.00 | 1.00 | 1.00 |
| timeout_executor@executor | 0.20 | 1.00 | 1.00 | 1.00 |
| inject_customer_note@researcher | 1.00 | 1.00 | 1.00 | 1.00 |
| inject_upsell_note@researcher | 1.00 | 1.00 | 1.00 | 1.00 |
| compromised_researcher@researcher | 1.00 | 1.00 | 1.00 | 1.00 |
| compromised_researcher_facts@researcher | 1.00 | 1.00 | 1.00 | 1.00 |
| compromised_planner@planner | 0.00 | 1.00 | 1.00 | 1.00 |
| compromised_planner_foreign@planner | 0.35 | 1.00 | 0.35 | 1.00 |

Accidental vs. adversarial faults (kept separate):

| fault_class / condition | runs | success | unauthorized_rate | est_latency_s | llm_calls | tokens |
|---|---|---|---|---|---|---|
| accidental / no_recovery | 276 | 0.30 | 0.00 | 8.15 | 0.00 | 0.00 |
| accidental / restart_snapshot | 276 | 0.95 | 0.00 | 8.15 | 0.00 | 0.00 |
| accidental / restart_compensate | 276 | 0.84 | 0.00 | 8.15 | 0.00 | 0.00 |
| accidental / verify_rule | 276 | 1.00 | 0.00 | 8.15 | 0.00 | 0.00 |
| adversarial / no_recovery | 148 | 0.73 | 0.27 | 0.00 | 0.00 | 0.00 |
| adversarial / restart_snapshot | 148 | 1.00 | 0.27 | 0.00 | 0.00 | 0.00 |
| adversarial / restart_compensate | 148 | 0.90 | 0.27 | 0.00 | 0.00 | 0.00 |
| adversarial / verify_rule | 148 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## 4. Full restart after an oracle-identified fault

`db_success`: final database looks correct. `strict_recovered`: also no harmful irreversible effect from the faulted run, and (snapshot) no lost external updates / no irreversible effect silently erased from the DB. Rates in the last four columns are fractions of runs with >=1 event.

| condition | runs | db_success | strict_recovered | recomputed_stages | retained_valid | discarded_valid | est_recovery_latency_s | recovery_llm_calls | recovery_tokens | irreversible_before | irreversible_deviations | lost_external_updates | irreversible_falsely_undone | unauthorized_before |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| restart_compensate | 424 | 0.86 | 0.63 | 3.00 | 0.00 | 0.48 | 0.00 | 0.00 | 0.00 | 1.86 | 0.37 | 0.00 | 0.00 | 0.09 |
| restart_snapshot | 424 | 0.97 | 0.21 | 3.00 | 0.00 | 0.48 | 0.00 | 0.00 | 0.00 | 1.86 | 0.37 | 0.17 | 0.79 | 0.09 |

Strict recovery by fault:

| fault_id | restart_compensate | restart_snapshot |
|---|---|---|
| incorrect_inflate@researcher | 0.47 | 0.06 |
| incorrect_deflate@researcher | 0.39 | 0.35 |
| incorrect_order_qty@researcher | 0.24 | 0.16 |
| incorrect_irrelevant@researcher | 1.00 | 0.08 |
| plan_qty_error@planner | 0.09 | 0.00 |
| exec_wrong_warehouse@executor | 0.12 | 0.00 |
| stale_research_relevant@researcher | 0.22 | 0.00 |
| stale_research_irrelevant@researcher | 1.00 | 0.00 |
| stale_plan_relevant@planner | 0.22 | 0.00 |
| timeout_researcher@researcher | 0.96 | 0.96 |
| timeout_planner@planner | 1.00 | 1.00 |
| timeout_executor@executor | 1.00 | 0.60 |
| inject_customer_note@researcher | 1.00 | 0.08 |
| inject_upsell_note@researcher | 1.00 | 0.08 |
| compromised_researcher@researcher | 1.00 | 0.08 |
| compromised_researcher_facts@researcher | 1.00 | 0.08 |
| compromised_planner@planner | 0.00 | 0.00 |
| compromised_planner_foreign@planner | 0.35 | 0.00 |

## 5. Always-verify baselines

| condition | runs | success | detected | unauthorized_rate | est_latency_s | llm_calls | tokens | verify_checks |
|---|---|---|---|---|---|---|---|---|
| verify_rule | 424 | 1.00 | 1.00 | 0.00 | 5.31 | 0.00 | 0.00 | 3.79 |

Detection quality (harmful = changed the outcome in the no-recovery run). False positives are measured on clean runs.

| condition | harmful_faults | detected_harmful | success_on_harmful |
|---|---|---|---|
| verify_rule | 233 | 1.00 | 1.00 |

| condition | false_positive_rate |
|---|---|
| clean_verify_rule | 0.00 |

Detection rate by fault:

| fault_id | verify_rule |
|---|---|
| incorrect_inflate@researcher | 1.00 |
| incorrect_deflate@researcher | 0.96 |
| incorrect_order_qty@researcher | 1.00 |
| incorrect_irrelevant@researcher | 1.00 |
| plan_qty_error@planner | 1.00 |
| exec_wrong_warehouse@executor | 1.00 |
| stale_research_relevant@researcher | 1.00 |
| stale_research_irrelevant@researcher | 1.00 |
| stale_plan_relevant@planner | 1.00 |
| timeout_researcher@researcher | 1.00 |
| timeout_planner@planner | 1.00 |
| timeout_executor@executor | 1.00 |
| inject_customer_note@researcher | 1.00 |
| inject_upsell_note@researcher | 1.00 |
| compromised_researcher@researcher | 1.00 |
| compromised_researcher_facts@researcher | 1.00 |
| compromised_planner@planner | 1.00 |
| compromised_planner_foreign@planner | 1.00 |

## 6. Dependency-tracking overhead (per run)

| condition | artifacts | dependency_edges | trace_bytes | bookkeeping_ms |
|---|---|---|---|---|
| clean | 5.00 | 11.48 | 9410.40 | 0.01 |
| no_recovery | 5.00 | 10.99 | 10906.26 | 0.01 |
| restart_snapshot |  |  | 20099.33 |  |
| restart_compensate |  |  | 17764.98 |  |
| verify_rule | 6.46 | 14.47 | 14038.76 | 0.01 |
| clean_verify_rule | 5.00 | 11.48 | 10243.68 | 0.01 |
