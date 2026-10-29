# Existing benchmark and framework experiments

## tau-bench retail (replay of ground-truth actions, scripted customer)

Clean: 78/115 test tasks solved (pass^1 = 0.68); 0.0 LLM calls, 0.0k prompt tokens, ~0 s per task (token-based); errors: 0.

Fault runs on 108 test tasks that have a ground-truth write (2592 runs).

### Propagation (no recovery)

| fault                   |   tasks |   applied |   success |   actions_changed |   state_changed |   outcome_changed |   irreversible_dev |   attack_success |
|:------------------------|--------:|----------:|----------:|------------------:|----------------:|------------------:|-------------------:|-----------------:|
| corrupt_read            |  108.00 |      0.56 |      0.69 |              0.00 |            0.00 |              0.00 |               0.00 |             0.00 |
| corrupt_read_irrelevant |  108.00 |      0.53 |      0.69 |              0.00 |            0.00 |              0.00 |               0.00 |             0.00 |
| injection               |  108.00 |      1.00 |      0.69 |              0.00 |            1.00 |              0.00 |               0.00 |             0.00 |
| lost_response           |  108.00 |      1.00 |      0.69 |              0.00 |            0.00 |              0.00 |               0.00 |             0.00 |
| transient_error         |  108.00 |      1.00 |      0.69 |              0.00 |            0.00 |              0.00 |               0.00 |             0.00 |
| wrong_args              |  108.00 |      0.96 |      0.14 |              0.83 |            0.82 |              0.56 |               0.67 |             0.00 |

### Success by fault and condition (applied faults)

| fault                   |   no_recovery |   restart_compensate |   restart_snapshot |   verify_rule |
|:------------------------|--------------:|---------------------:|-------------------:|--------------:|
| corrupt_read            |          0.58 |                 0.55 |               0.58 |          0.58 |
| corrupt_read_irrelevant |          0.58 |                 0.54 |               0.58 |          0.58 |
| injection               |          0.69 |                 0.58 |               0.69 |          0.69 |
| lost_response           |          0.69 |                 0.58 |               0.69 |          0.69 |
| transient_error         |          0.69 |                 0.58 |               0.69 |          0.69 |
| wrong_args              |          0.12 |                 0.14 |               0.70 |          0.70 |

### Full restart

| condition          |   success |   strict |   irreversible_before |   falsely_undone |   uncompensable |   total_llm_calls |
|:-------------------|----------:|---------:|----------------------:|-----------------:|----------------:|------------------:|
| restart_compensate |      0.49 |     0.49 |                  1.31 |             0.00 |            1.31 |              0.00 |
| restart_snapshot   |      0.67 |     0.05 |                  1.31 |             1.31 |            0.00 |              0.00 |

### Rule verifier

Detected 1.00 of outcome-changing faults (60); success 0.67; 0.0 LLM calls vs 0.0 without verification.

### Cost per run (restart rows: rerun only)

| condition          |   llm_calls |   est_latency_s |
|:-------------------|------------:|----------------:|
| no_recovery        |         0.0 |             0.0 |
| restart_compensate |         0.0 |             0.0 |
| restart_snapshot   |         0.0 |             0.0 |
| verify_rule        |         0.0 |             0.0 |

## LangGraph implementation of the workflow (same agents, tasks, faults)

Clean success 1.00 (25 tasks); 5.0 checkpoints per run.

### Success by faulty stage

| fault_stage   |   ('success', 'no_recovery') |   ('success', 'restart_checkpoint') |   ('success', 'restart_checkpoint_compensate') |
|:--------------|-----------------------------:|------------------------------------:|-----------------------------------------------:|
| executor      |                         0.12 |                                0.43 |                                           0.81 |
| planner       |                         0.07 |                                0.31 |                                           0.52 |
| researcher    |                         0.68 |                                0.79 |                                           0.91 |

### Checkpoint rewind vs. full restart

| condition                     |   success |   strict |   recomputed_stages |   retained_stages |   over_shipped |   dangling |   recovery_llm_calls |
|:------------------------------|----------:|---------:|--------------------:|------------------:|---------------:|-----------:|---------------------:|
| restart_checkpoint            |      0.62 |     0.52 |                2.52 |              0.48 |           0.16 |       0.02 |                 0.00 |
| restart_checkpoint_compensate |      0.79 |     0.62 |                2.52 |              0.48 |           0.14 |       0.00 |                 0.00 |
| native_restart_compensate     |      0.86 |     0.63 |                3.00 |              0.00 |         nan    |     nan    |                 0.00 |
| native_restart_snapshot       |      0.97 |     0.21 |                3.00 |              0.00 |         nan    |     nan    |                 0.00 |

