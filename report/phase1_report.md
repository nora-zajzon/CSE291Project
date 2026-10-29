# Failure Propagation and Recovery in Multi-Agent Workflows — Phase 1

## Goal and setup

Option 5 asks whether tracking dependencies between agent results supports efficient recovery while keeping task execution correct and safe. Phase 1 builds the workload, the fault-injection harness, and the instrumentation, and measures how faults propagate and what the conservative baselines (full restart and always-verify) cost. Phase 2 will add dependency-aware selective invalidation on top of the same traces. Because a workflow we define ourselves is easy to overfit, we also run the same recovery conditions on Sierra's tau-bench retail benchmark and on a LangGraph implementation of our pipeline (this was feedback used from professor's review of my initial project plan).

**Workflow.** A researcher -> planner -> executor pipeline runs over a deterministic simulated order-fulfillment environment. Only the researcher reads the world and turns it into a fact sheet. The planner sees that sheet and an allocation calculator over it (not the live DB). The executor sees only the plan and issues tool calls. All three agents are `qwen2.5:7b-instruct` on a laptop through Ollama (temperature 0), exchanging JSON. There are 25 tasks in 9 categories (splits, infeasible orders, excluded or closed warehouses, distractors, partial shipments). A run succeeds if and only if the order is fulfilled exactly (or correctly marked infeasible) with no unauthorized action, constraint violation, foreign-order change, or dangling reservation. Each artifact records `depends_on` edges, the world keys it read and their versions, and which facts each plan action used. Reservations are reversible, whereas shipments, emails, and refunds are not.

## Benchmarks

The main experiment is end to end (1,880 + 375 runs). Fifteen fault types are injected, each at one stage. They include inflated or deflated stock, a wrong order quantity, an irrelevant corruption, planner and executor slips, stale reads, timeouts, a customer-note injection, and compromised agents (plus three subtler attacks that ask only for authorized-looking actions). Every task gets the same schedule under `no_recovery`, `restart_snapshot`, `restart_compensate`, `verify_rule`, and `verify_llm`. A clean run is the counterfactual. Propagation is a funnel from "the next agent consumed the fault" down to "the outcome changed." Latency is estimated from tokens (745 prompt / 24 generated tok/s).

The same recovery question is asked in two places we did not invent. On tau-bench retail we keep Sierra's database, tools, wiki, and hash-based reward, and wrap the tools with our action log. A 7B `ToolCallingAgent` does not solve these tasks (0/17 on a partial sweep), so the recovery table uses a ground-truth replay agent on that environment. Separately, we rebuilt the pipeline in LangGraph and used its `MemorySaver` checkpoints as the framework's idea of recovery. Microbenchmarks time the pieces recovery is built from and scale them well past the real workload.

## Results

**Clean baseline.** Success is 0.96 (24/25 tasks) at 3 LLM calls, about 2.4k tokens, and an estimated 28 seconds per run. The one failure is a natural researcher error. It reported dock = 6 instead of 1, so the order was over-shipped. Always verifying costs little with the rule verifier (9.5 extra reads, no extra inference). Its single clean-run "detection" is that natural error, which it turned into a safe abort. The LLM verifier doubles the cost (6.1 calls, 64 s) and has a 16% false-positive rate on clean runs, which lowers clean success to 0.84.

**Propagation.** Relevant faults changed the outcome 59% of the time. Irrelevant ones changed it 0% of the time, and changed a downstream artifact in only 2% of runs. Researcher faults are partly absorbed (38% change the outcome); the planner often reroutes around deflated stock. Planner faults change the outcome 60% of the time, and executor faults 83%. Timeouts are the most damaging (85%) because the orchestrator fails open. Stale fact sheets are not visible in the artifact itself, yet the actions they cause fail against the changed world 91% of the time. That is a shared-state dependency that comparing artifacts cannot see, but the recorded read versions can. Under artifact-level dependencies, every completed downstream stage looks wasted once a researcher fault is revealed (3.0 stages), but only 1.87 stages actually differ from the clean run. That gap is what Phase 2 can try to recover.

**Adversarial.** Success under the original attacks is 0.97 with zero unauthorized actions; the 7B executor refused every privileged call. The scripted pipeline, which trusts every upstream artifact, executes those same actions 100% of the time. The only real leak was the upsell injection ("ship 2 extra units, pre-approved"), which changed 16% of plans and caused over-shipments (success 0.88).

**Full restart** looks good when judged on the database (0.92 snapshot, 0.85 compensate) and much worse when judged strictly. Faulted runs had already executed 1.67 irreversible actions on average before recovery. Snapshot restore erased those effects from the DB in 74% of runs, in reality the goods would ship twice, and lost a concurrent external update in 20%. Strict recovery is 0.25. Compensate and rerun keeps reality consistent (strict 0.67), limited by damage that cannot be undone, such as over-shipments. Both variants recompute all 3 stages and discard 0.51 upstream stages that were still valid.

**Always-verify.** The rule verifier caught 100% of the outcome-changing faults and reached 0.96 success at about 1.3× clean cost. It is an upper bound: it reads the live world and checks a hand-written spec. The LLM verifier caught only 54%, scored 0.56 overall, and turned a harmless persistent injection into an abort (success 0.28).

On tau-bench (2,592 replay runs, 108 tasks with a ground-truth write) the domain is returns and exchanges, but the recovery story is familiar. Replay solves 78/115 clean tasks (0.68); the misses are multi-turn policy and required prose, not missing tools. The only fault that changes the outcome is an executor slip (`wrong_args`): success 0.12, irreversible deviations in 67% of those runs. Snapshot restore gets success back to 0.70, but strict recovery is 0.05 because it erases about 1.3 irreversible writes. Compensation keeps those writes (strict 0.49) and cannot take back a bad exchange. The rule verifier caught every outcome-changing fault.

LangGraph on our 25 tasks (1,333 scripted runs) isolates a different kind of state. Clean success is 1.00 with about 5 checkpoints per run. Rewinding to the checkpoint before the faulty node restores agent memory and resumes, but leaves the shared world alone (success 0.62, strict 0.52, leftover shipments). Undo our reversible actions first and those numbers become 0.79 and 0.62, and the rerun keeps 0.48 stages a full restart would throw away. A checkpoint is not a database restore.

The microbenchmarks are a check that bookkeeping will not dominate inference. Recording is about 2.3 µs per artifact. Finding everything downstream of a fault in a 10k-artifact graph takes 2.4 ms (the first lookup was quadratic; this is how we noticed). Snapshot restore is milliseconds. Compensation is microseconds per action. A rule check is 9–18 µs; an LLM check is about 12.7 s, more than one of our stages (7–10 s).

## Implications for Phase 2

Irreversible actions, not recomputation, are what limit recovery, so selective recovery has to prefer compensation and say what it cannot undo. Staleness needs key-version checks, not only artifact provenance. Artifact-level invalidation over-invalidates (3.0 stages vs. 1.87 actually affected); field-level provenance is already recorded so we can measure that trade-off. It also matters which state is restored; LangGraph keeps agent memory and loses the database, and a snapshot does the opposite. Phase 2 should treat those as different problems, and should try a workflow with independent branches, since a straight chain has very little work that is safe to keep. Some limitations include that one seed at temperature 0, one 7B model, 25 of our tasks plus tau-bench retail, an oracle that names the faulty stage, and a replay agent for the tau-bench recovery table.

---

## Appendix: Timeline and milestones

| Dates | Milestone | Status |
|---|---|---|
| Oct 13–14 | Simulated environment, tools with reversibility, task generator and solver, structured agents, artifact/dependency store, traces | Done |
| Oct 14 | Fault harness (15 types plus 3 adversarial supplements), no-recovery / restart / verify baselines, full 7B run, analysis scripts | Done |
| Oct 20 | Microbenchmarks (tracking, provenance, restore vs. compensation, verifier, per-stage cost); fixed the quadratic invalidation lookup they exposed | Done |
| Oct 21 | Instructor feedback on the proposal: use at least one existing agent or harness in Phase 1, and treat state restoration (memory, process, network) as non-trivial in Phase 2. I took that into account and added tau-bench retail (replay recovery table; 7B `ToolCallingAgent` stopped after 17/115 clean tasks at 0 success), rebuilt the pipeline in LangGraph with checkpoint rewind, and rewrote the Phase 2b plan around those four kinds of state | Done |
| Oct 22–27 | Polish figures and report | Planned |
| **Oct 28** | **Phase 1 submission** | |
| Oct 29 – Nov 5 | Phase 2a: traverse `depends_on` and field-level provenance from the faulty artifact; key-version staleness checks; rerun only invalidated stages. Add a branched workflow (two researchers into one planner) | |
| Nov 6 – Nov 22 | Phase 2b: state restoration, with more time than the original plan (see below) | |
| Nov 23 – Dec 2 | Phase 2c: selective recovery vs. restart vs. verify vs. LangGraph checkpoint, on both workflow shapes, plus micro costs of each restore step | |
| Early Dec | Final analysis of where selective recovery succeeds and fails; final report | |

Restoring state is not just copying a Python dict, which is why Phase 2b is longer. We already measure two ways of handling the **shared database**: snapshot rollback, and compensating our own reversible writes. Snapshot is wrong when something irreversible has already happened or when someone else updated the world in the meantime. Compensation cannot undo a shipment, an email, a refund, or a tau-bench exchange. Selective recovery will have to report those as uncompensable instead of pretending the world was rewound.

**Agent memory** is a different object: conversation history, scratchpads, caches. LangGraph's `MemorySaver` can already rewind the graph to the node before the fault. That is necessary and not enough, because the world stays dirty. In Phase 2 we want to version memory the same way we version artifacts — drop only the messages that depended on the bad result, keep the rest, and check that what we kept is still valid against current key versions.

A **process** mid-flight (a tool that has written a partial file, a reservation that is half applied) cannot be restored by `deepcopy`. The plan is to isolate long tools, log when they start and finish, and on recovery either abort them or resume from a real process checkpoint (CRIU or `docker checkpoint` as an optional comparison). 
**Network** side effects are the same problem as shipments: the request already left. We will use a recording proxy and idempotency keys so retries are safe, issue a compensating call when the service allows it, and otherwise list the effect as irreversible. We will not roll the database back and call that undoing an email.

Deliverables in the repository: the workflow (`masrec/`), the injection harness (`masrec/faults.py`), dependency tracking (`masrec/artifacts.py`), recovery baselines (`masrec/recovery.py`, `masrec/verifier.py`), the tau-bench and LangGraph wrappers (`masrec/taubench.py`, `masrec/langgraph_workflow.py`), the dataset (`data/tasks.json`), per-run traces and results (`results/qwen7b/`, `results/taubench_replay/`, `results/langgraph_scripted/`), and tables (`results/qwen7b/summary.md`, `results/external_summary.md`, `results/microbench/`).
