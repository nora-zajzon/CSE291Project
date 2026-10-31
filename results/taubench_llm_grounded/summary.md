# Existing benchmark and framework experiments

## tau-bench retail (tau-bench's ToolCallingAgent, scripted customer)

Grounded variant: the harness prefetches the customer's user record and all their orders through the fault/verification hooks. No gold action or expected answer is provided. This changes the observation protocol, so it is not an unmodified tau-bench score.

Latency estimates use token throughput. `wall_s` records actual elapsed run time (cache hits make this shorter); `llm_latency_s` sums original measured model call times, including cached calls. Restart costs must include both the faulted attempt and the rerun.

Clean: 3/3 test tasks solved (pass^1 = 1.00); 6.0 LLM calls, 38.3k prompt tokens, ~82 s per task (token-based); errors: 0.

Fault runs on 3 test tasks that have a ground-truth write (36 runs).

### Propagation (no recovery)

| fault        |   tasks |   applied |   success |   actions_changed |   state_changed |   outcome_changed |   irreversible_dev |   attack_success |
|:-------------|--------:|----------:|----------:|------------------:|----------------:|------------------:|-------------------:|-----------------:|
| corrupt_read |    3.00 |      1.00 |      1.00 |              0.33 |            0.00 |              0.00 |               0.00 |             0.00 |
| injection    |    3.00 |      1.00 |      0.67 |              0.67 |            1.00 |              0.33 |               0.00 |             0.33 |
| wrong_args   |    3.00 |      1.00 |      0.33 |              0.67 |            0.67 |              0.67 |               0.67 |             0.00 |

### Observation exposure and downstream changes

|                                |   tasks |   exposed_to_llm |   changed_actions_after_exposure |
|:-------------------------------|--------:|-----------------:|---------------------------------:|
| ('accidental', 'corrupt_read') |       3 |                3 |                                1 |
| ('accidental', 'wrong_args')   |       3 |                0 |                                0 |
| ('adversarial', 'injection')   |       3 |                3 |                                2 |

Exposure means the faulty tool observation appeared in a later model prompt. Changed actions after exposure support propagation, but do not by themselves establish a general causal guarantee. Action-argument faults occur after inference and are assessed through the action/state comparisons above.

Agent-caused state changes (the injected note itself is excluded):

| fault        |   state_changed_excluding_injection |
|:-------------|------------------------------------:|
| corrupt_read |                                0.00 |
| injection    |                                0.33 |
| wrong_args   |                                0.67 |

### Paired clean-success stratum

|                                     |   tasks |   success |   outcome_changed |
|:------------------------------------|--------:|----------:|------------------:|
| ('accidental', 'corrupt_read', 1.0) |    3.00 |      1.00 |              0.00 |
| ('accidental', 'wrong_args', 1.0)   |    3.00 |      0.33 |              0.67 |
| ('adversarial', 'injection', 1.0)   |    3.00 |      0.67 |              0.33 |

### Timing definitions kept separate

| condition          |   observed_wall_s |   original_model_service_s |   estimated_inference_s |
|:-------------------|------------------:|---------------------------:|------------------------:|
| no_recovery        |             18.93 |                      43.25 |                   86.25 |
| restart_compensate |             29.72 |                      51.53 |                  110.62 |
| restart_snapshot   |              0.20 |                      42.72 |                   83.35 |
| verify_rule        |             18.72 |                      42.95 |                   70.42 |

Total attempt costs (restart includes the faulted attempt and rerun; restore overhead is excluded):

| condition          |   attempts_wall_s |   attempts_llm_latency_s |   attempts_est_latency_s |
|:-------------------|------------------:|-------------------------:|-------------------------:|
| no_recovery        |             18.93 |                    43.25 |                    86.25 |
| restart_compensate |             48.66 |                    94.78 |                   196.88 |
| restart_snapshot   |             19.13 |                    85.97 |                   169.60 |
| verify_rule        |             18.72 |                    42.95 |                    70.42 |

### Success by fault and condition (applied faults)

| fault        |   no_recovery |   restart_compensate |   restart_snapshot |   verify_rule |
|:-------------|--------------:|---------------------:|-------------------:|--------------:|
| corrupt_read |          1.00 |                 1.00 |               1.00 |          1.00 |
| injection    |          0.67 |                 0.33 |               0.67 |          1.00 |
| wrong_args   |          0.33 |                 0.67 |               1.00 |          1.00 |

### Full restart

| condition          |   success |   strict |   irreversible_before |   falsely_undone |   uncompensable |   total_llm_calls |
|:-------------------|----------:|---------:|----------------------:|-----------------:|----------------:|------------------:|
| restart_compensate |      0.67 |     0.67 |                  0.78 |             0.00 |            0.78 |             14.00 |
| restart_snapshot   |      0.89 |     0.22 |                  0.78 |             0.78 |            0.00 |             12.22 |

### Rule verifier

Detected 1.00 of outcome-changing faults (3); success 1.00; 5.3 LLM calls vs 6.2 without verification.

### Cost per run (restart rows: rerun only)

| condition          |   llm_calls |   est_latency_s |
|:-------------------|------------:|----------------:|
| no_recovery        |         6.2 |            86.3 |
| restart_compensate |         7.8 |           110.6 |
| restart_snapshot   |         6.0 |            83.4 |
| verify_rule        |         5.3 |            70.4 |

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

