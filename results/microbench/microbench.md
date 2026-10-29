# Microbenchmarks

Median of repeated runs on one laptop core. The workflow in the macrobenchmark has 5-12 artifacts, 3-10 inventory keys per warehouse and 2-8 executed actions per run.

## M1. Dependency tracking

Branch = the middle of the chain, or the first child of the root in the fan-out tree (a fault in one of three parallel sub-agents).

| topology | artifacts | edges | record_us_per_artifact | descendants_root_ms | invalidated_from_root | descendants_branch_ms | invalidated_from_branch |
|---|---|---|---|---|---|---|---|
| chain | 10 | 9 | 2.17 | 0.002 | 9 | 0.001 | 4 |
| chain | 100 | 99 | 2.02 | 0.019 | 99 | 0.012 | 49 |
| chain | 1000 | 999 | 2.24 | 0.201 | 999 | 0.148 | 499 |
| chain | 10000 | 9999 | 2.80 | 2.403 | 9999 | 1.777 | 4999 |
| fanout3 | 10 | 9 | 2.18 | 0.002 | 9 | 0.001 | 3 |
| fanout3 | 100 | 99 | 2.23 | 0.020 | 99 | 0.012 | 39 |
| fanout3 | 1000 | 999 | 2.28 | 0.207 | 999 | 0.124 | 363 |
| fanout3 | 10000 | 9999 | 2.41 | 2.195 | 9999 | 1.265 | 3438 |

## M2. Field-level provenance

106 plan actions over 25 tasks, 187 field dependencies. Deriving dependencies: 0.28 us/action. Matching against a corrupted fact: 0.100 us/dependency.

## M3. State restoration

| warehouses | inventory_keys | snapshot_ms | restore_ms |
|---|---|---|---|
| 3 | 60 | 0.029 | 0.030 |
| 10 | 200 | 0.083 | 0.083 |
| 100 | 2000 | 0.766 | 0.772 |
| 1000 | 20000 | 7.716 | 7.744 |

| executed_actions | compensate_ms | compensate_us_per_action | snapshot_restore_ms_same_world |
|---|---|---|---|
| 1 | 0.003 | 3.04 | 0.083 |
| 10 | 0.018 | 1.82 | 0.083 |
| 100 | 0.270 | 2.70 | 0.083 |
| 1000 | 12.019 | 12.02 | 0.083 |

## M4. Rule verifier (per check, 25 tasks)

| check | median_us | max_us | env_reads |
|---|---|---|---|
| research | 17.4 | 27.8 | 4.6 |
| plan | 17.2 | 23.0 | 4.6 |
| calls | 8.9 | 13.0 | 0.0 |

## M5. LLM cost per stage and per LLM-verifier check (7B traces, token-based latency)

| stage | calls | prompt_tokens | completion_tokens | est_latency_s |
|---|---|---|---|---|
| researcher | 25 | 524 | 225 | 10.1 |
| planner | 25 | 749 | 225 | 10.4 |
| executor | 25 | 533 | 157 | 7.3 |

LLM verifier, per check (21 clean runs without retries): 677 prompt + 281 completion tokens, about 12.7 s.
