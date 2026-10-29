# Failure Propagation and Recovery in Multi-Agent Workflows (CSE 291A project)

A researcher → planner → executor workflow over a deterministic simulated order-fulfillment
environment, with a fault-injection harness, artifact-level and field-level dependency tracking,
full-restart recovery baselines, and always-verify baselines. Phase 1 (this code) measures clean
behavior, fault propagation, and the cost of conservative recovery. Phase 2 will add
dependency-aware selective invalidation on top of the same traces.

```
World (inventory, orders, reservations, shipments, emails, refunds)
  │ get_order / list_warehouses / get_inventory      (read-set + per-key versions recorded)
  ▼
Researcher (LLM) ── fact sheet artifact ──► Planner (LLM) ── plan artifact ──► Executor (LLM)
                                                                                 │ tool calls
                                                                                 ▼
                      reserve / ship(irreversible) / notify(irreversible) / mark_infeasible
                      privileged: adjust_inventory / cancel_order / issue_refund(irreversible)
```

The agents reason with an LLM; the world itself is deterministic, so correctness is checked
programmatically (no LLM judge).

## Layout

| Path | What it is |
|---|---|
| `masrec/world.py` | Simulated shared state, per-key versions, snapshots |
| `masrec/tools.py` | Tools, authorization policy, reversibility + compensating actions |
| `masrec/tasks.py` | Task generator (25 tasks, 9 categories) and exhaustive ground-truth solver |
| `masrec/agents.py` | Researcher / Planner / Executor (LLM) and scripted reference agents |
| `masrec/artifacts.py` | Artifacts with `depends_on`, read versions, field-level provenance |
| `masrec/faults.py` | 15 fault types, per-task fault schedule, injector |
| `masrec/workflow.py` | Orchestrator with fault hooks and optional verification |
| `masrec/verifier.py` | Always-verify baselines: deterministic rule verifier and LLM verifier |
| `masrec/recovery.py` | Full restart: DB-snapshot restore vs. compensate-and-rerun |
| `masrec/evaluator.py` | Correctness, unauthorized actions, propagation funnel vs. clean counterfactual |
| `masrec/experiment.py` | Runs all conditions under identical fault schedules |
| `run_experiment.py` | CLI |
| `analyze_results.py` | Tables (`summary.md`) and figures (macrobenchmark) |
| `microbench.py` | Per-mechanism microbenchmarks: tracking, provenance, restore vs. compensation, verifier checks, per-stage LLM cost (`results/microbench/`) |
| `masrec/taubench.py` | Fault injection and recovery on Sierra's **tau-bench** retail env and `ToolCallingAgent` |
| `run_taubench.py` | CLI for tau-bench clean + fault conditions (`--agent llm` or `replay`) |
| `masrec/langgraph_workflow.py` | Same workflow on **LangGraph**, using `MemorySaver` checkpoints as a recovery baseline |
| `run_langgraph.py` | CLI for LangGraph clean / no_recovery / checkpoint rewind |
| `analyze_external.py` | Tables for tau-bench and LangGraph (`results/external_summary.md`) |
| `third_party/tau-bench` | Sierra tau-bench (git submodule, commit `59a200c6`) |
| `data/tasks.json` | Generated task dataset with ground truth |
| `results/<run>/` | `runs.csv` (one row per run), `fault_schedule.jsonl`, `traces/*.jsonl`, `summary.md`, `figures/` |
| `report/phase1_report.md` | Phase 1 write-up and timeline |

## Running

```bash
git clone --recurse-submodules https://github.com/nora-zajzon/CSE291Project.git
cd CSE291Project
python3 -m venv .venv && .venv/bin/pip install matplotlib pandas numpy requests pytest langgraph tabulate
.venv/bin/python -m pytest -q tests                       # harness tests (scripted agents)

# deterministic reference pipeline (no LLM; perfect reasoning, trusts all upstream artifacts)
.venv/bin/python run_experiment.py --backend scripted --out results/scripted

# LLM agents via Ollama on a laptop / small CPU instance
ollama pull qwen2.5:7b-instruct
.venv/bin/python run_experiment.py --backend ollama --model qwen2.5:7b-instruct --workers 4 --out results/qwen7b
# any OpenAI-compatible server (vLLM, llama.cpp server)
.venv/bin/python run_experiment.py --backend openai --base-url http://host:8000/v1 --model <name>

.venv/bin/python analyze_results.py results/qwen7b
.venv/bin/python microbench.py --runs results/qwen7b     # no LLM needed; ~5 s

# existing benchmark (tau-bench retail). --agent llm uses tau-bench's ToolCallingAgent;
# --agent replay replays ground-truth actions (no LLM) for controlled recovery metrics.
.venv/bin/pip install -e third_party/tau-bench
.venv/bin/python run_taubench.py --phase clean --agent llm --out results/taubench_qwen7b
.venv/bin/python run_taubench.py --phase faults --agent llm --out results/taubench_qwen7b
.venv/bin/python run_taubench.py --phase clean --agent replay --out results/taubench_replay
.venv/bin/python run_taubench.py --phase faults --agent replay --out results/taubench_replay

# existing framework (LangGraph checkpoint rewind on the same 25 tasks)
.venv/bin/python run_langgraph.py --backend scripted --out results/langgraph_scripted
.venv/bin/python run_langgraph.py --backend ollama --out results/langgraph_qwen7b
.venv/bin/python analyze_external.py
```

Useful flags: `--task-ids T01,T12`, `--faults inflate,stale`, `--conditions clean,no_recovery`,
`--seeds 0,1,2 --temperature 0.7`, `--no-cache`.

LLM responses are cached by (model, temperature, seed, messages) in `cache/`. A cache hit reports
the original call's latency and token counts, so cost metrics are unchanged; the cache only
saves wall-clock time when, for example, the researcher prompt is identical across faults injected
later in the pipeline.

## Conditions

| Condition | Meaning |
|---|---|
| `clean` | No fault. Also the counterfactual reference for every faulted run of that task. |
| `clean_verify_rule`, `clean_verify_llm` | No fault, with verification (fixed overhead, false positives). |
| `no_recovery` | Fault injected; workflow runs to completion. Measures propagation. |
| `restart_snapshot` | After the (oracle-identified) fault: restore the DB snapshot, rerun all stages. |
| `restart_compensate` | Undo our own reversible actions, keep external changes, rerun all stages. |
| `verify_rule`, `verify_llm` | Each artifact is verified just before the next stage consumes it; on failure the producing stage (and, if stale, the researcher) is rerun. |

## Faults

Accidental: `incorrect_inflate`, `incorrect_deflate`, `incorrect_order_qty`, `incorrect_irrelevant`
(researcher facts), `plan_qty_error` (planner), `exec_wrong_warehouse` (executor), `stale_research_relevant`,
`stale_research_irrelevant`, `stale_plan_relevant` (a concurrent external process changes stock after a stage
read it), `timeout_researcher` (partial fact sheet forwarded), `timeout_planner`, `timeout_executor`
(half the tool calls ran).

Adversarial: `inject_customer_note` (persistent indirect prompt injection in the order record),
`compromised_researcher` (malicious instructions in the fact sheet), `compromised_planner`
(privileged actions appended to the plan).

Faults are transient (fire on the first attempt of a stage only) except the persistent injected note.
Each fault carries a design-time relevance label, but relevance is also *measured* against the clean
counterfactual (`downstream_changed`, `actions_changed`, `state_changed`, `outcome_changed`), so an
ignored fault is never counted as robust recovery.
