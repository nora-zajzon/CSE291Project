# Existing benchmark and framework experiments

## tau-bench retail (tau-bench's ToolCallingAgent, scripted customer)

Latency estimates use token throughput. `wall_s` records actual elapsed run time (cache hits make this shorter); `llm_latency_s` sums original measured model call times, including cached calls. Restart costs must include both the faulted attempt and the rerun.

Clean: 0/3 test tasks solved (pass^1 = 0.00); 9.7 LLM calls, 43.1k prompt tokens, ~99 s per task (token-based); errors: 0.

(fault runs not available yet)

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

