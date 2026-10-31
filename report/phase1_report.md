# Failure Propagation and Recovery in Multi-Agent Workflows (Phase 1\)

## Setup and evaluation

Phase 1 measures failure propagation and recovery costs in a researcher → planner → executor workflow, comparing full restart and intermediate verification. Three Qwen2.5-7B agents perform 25 simulated order-fulfillment tasks on a laptop. The researcher produces facts, the planner generates actions, and the executor calls tools. Programmatic checks evaluate fulfillment and safety. Traces record messages, artifacts, tool actions, dependencies, and state versions. Reservations are reversible; shipments, emails, and refunds are irreversible.

The workflow has 1,880 original runs with 15 fault types, plus 375 runs covering three supplemental attacks. Faults include incorrect results, stale outputs, simulated timeouts, and malicious instructions. Accidental and adversarial cases are analyzed separately. Conditions share task/fault schedules; clean runs provide counterfactuals. Restart methods receive the known faulty-stage identity, separating recovery from detection. Measurements cover correctness, unauthorized actions, affected stages, wasted work, latency, inference cost, and tracking overhead. Custom-workflow aggregate results below pool both run sets, matching the repository summary.

To incorporate an existing agent, it also evaluates tau-bench's ToolCallingAgent on three preselected retail tasks. The ordinary protocol solves 0/3; a grounded variant supplying current customer records, without expected answers, solves 3/3. It runs 36 fault-condition combinations with the same 16-step limit per attempt. This adapted protocol is a controlled study, not an unmodified benchmark score. Additional controls use tau-bench replay and LangGraph checkpoints.

## Results

**Clean baseline and cost.** The custom workflow succeeds on 24/25 tasks, using three LLM calls and about 2.4k tokens. Estimated latency is 28 seconds; account stage service time is 102 seconds. The LLM verifier increases calls to 6.1 and lowers clean success to 0.84. It flags 16% of uninjected runs; this is a detection rate, not an adjudicated false-positive rate, because natural model errors occur. False positives remain unadjudicated. Latency estimates, observed elapsed times, and cache-adjusted service times are distinguished in the appendix. Restart costs include both attempts.

**Propagation and safety.** Relevant faults change task success in 48% of cases, versus 0% for irrelevant faults. Outcome changes increase from researcher faults (32%) to planner (49%) and executor faults (83%). Stale outputs expose shared-state dependencies that artifact comparisons alone miss. Exposure and downstream changes are measured separately. An ignored fault is not evidence of robust recovery. The workflow does not run any unauthorized actions under the tested attacks, although an upsell injection changes 16% of plans and causes over-shipments.

**Recovery and verification.** Snapshot restart achieves 0.92 database success but only 0.21 strict recovery; compensation achieves 0.87 and 0.71, respectively. Strict recovery additionally rejects harmful irreversible effects and, for snapshots, erased irreversible effects or lost external updates. Snapshot rollback erases irreversible effects in 77% of runs and loses external updates in 17%; rerunning could duplicate external effects that database rollback cannot undo. Both restart methods recompute all three stages. The rule verifier detects all outcome-changing faults and achieves 0.96 success at approximately 1.3x clean inference cost. It is an upper bound with privileged live-state access. The LLM verifier detects 54% of outcome-changing faults, misses 46%, and achieves 0.58 success.

**Existing-agent evidence.** In the grounded retail experiemnt, every corrupt read and malicious note reaches a later model prompt. Corrupt reads preserve correctness on all three tasks. A constant note causes an unauthorized address change, and restart does not reliably prevent recurrence. Verification blocks the address attack and achieves 9/9 task/fault successes. Snapshot restart achieves 8/9 database successes but only 2/9 strict recoveries; compensation achieves 6/9 by both metrics. These counts describe three tasks, not population estimates. Tau-bench replay provides state-recovery controls, but predetermined actions cannot establish robustness to faulty observations.

LangGraph checkpoint rewind restores agent memory while leaving shared state unchanged: strict recovery is 52%, increasing to 62% with compensation. Dependency recording costs approximately 2.3 μs per artifact, suggesting low tracking overhead relative to inference.

## Implications and limitations

Phase 2 will implement dependency-aware invalidation, check retained results against current state versions, and add independent branches to make reuse meaningful. LangGraph rewinds agent memory without restoring shared state; database snapshots restore shared state without undoing external effects. Recovery must coordinate these states, compensate for reversible actions, and report irreversible damage. The appendix expands the restoration plan to process and network state.

Limitations include one model, one seed for the custom workflow, three grounded retail tasks, unadjudicated false positives, and simulated timeouts rather than process kills. LangGraph retains our workflow and scripted agents, so it tests an existing framework rather than an independent agent workload.

## Appendix A: Cohorts and detailed results

The pooled custom-workflow analysis combines `results/qwen7b/runs.csv` (1,880 runs) and `results/qwen7b_adv/runs.csv` (375 runs), retaining one clean baseline per task. Fault comparisons use applied faults; the pooled recovery/verification tables contain 424 task/fault pairs. The relevant-fault propagation denominator is 370, with 50 irrelevant and four uncertain cases reported separately. Stage-level averages include all applied faults at that stage. Full tables are in \[the pooled summary\](../results/qwen7b/summary.md).

Earlier drafts described the original 15-fault cohort: snapshot database/strict success 0.92/0.25, compensation 0.85/0.67, and LLM-verifier success 0.56. The main text now consistently uses pooled figures: 0.92/0.21, 0.87/0.71, and 0.58. Changed outputs do not alone prove that retained work is safe to reuse; Phase 2 will validate reuse explicitly.

The original faults cover stock and order-quantity errors, irrelevant corruption, planner/executor slips, stale reads, three timeout locations, customer-note injection, and compromised agents. Supplemental attacks use an upsell note, fabricated researcher facts, and a foreign-order planner action. Artifacts record dependency edges, read versions, and field provenance.

Tau-bench replay solves 78/115 clean tasks and supplies 2,592 fault-condition runs on 108 tasks with ground-truth writes. For applied executor-argument faults, success rises from 0.12 without recovery to 0.70 with snapshot restart. Across applied fault cases, snapshot strict recovery is 0.05 and compensation strict recovery is 0.49. Replay ignores observation content, so these are state-recovery controls. \[External control results\](../results/external\_summary.md) also contain the 1,333-run scripted LangGraph study.

## Appendix B: Timing and overhead

`wall_s` records observed harness elapsed time. In the custom workflow, `latency_s` sums stage times, original model-call times reused on disk-cache hits, and simulated timeout charges. Token-throughput estimates use approximately 745 prompt and 24 generated tokens/s. These quantities are not interchangeable, and cache reuse does not constitute additional independent trials. \[The pooled summary\](../results/qwen7b/summary.md) reports clean estimated/service times of 27.83/102.42 seconds, increasing to 64.11/224.48 with LLM verification. \[The retail summary\](../results/taubench\_llm\_grounded/summary.md) separately reports observed elapsed, original model service, and estimated inference times, including paired total-attempt costs; its timing totals exclude restore overhead. Budgets are equal per attempt, not equal in total inference consumption across methods.

\[Microbenchmarks\](../results/microbench/microbench.md) report approximately 2.3 μs per recorded artifact and 2.4 ms to find descendants in a 10,000-artifact graph. Snapshot restoration takes milliseconds; compensation takes microseconds per action. Rule checks take approximately 9–18 μs, whereas an LLM check takes approximately 12.7 seconds.

## Appendix C: Existing-agent protocol and safety scope

Retail tasks 17, 48, and 53 (one address change and two returns) were selected by task type before observing their clean outcomes. The grounded variant prefetches the known customer's record and all their orders through the same tool, injection, and verification hooks. No expected actions or answers are supplied. All tasks remain in the fault cohort regardless of clean success. The study has six clean runs across the ordinary and grounded protocols and 36 fault-condition runs: three tasks x three faults x four conditions.

The faults are corrupt reads, altered action arguments, and a persistent malicious note; conditions are no recovery, snapshot restart, compensation restart, and rule verification. Traces retain model inputs, responses, observations, actions, and conservative dependencies on all previous tool events. Exposure means faulty content appears in a later model prompt. Agent-caused state comparisons exclude the injected note's own database change. Restart runs retain persistent inputs while transient faults do not recur.

The retail rule verifier has privileged live-data access and a rule for the known payload. It sanitizes both reads and successful-write responses, which can return the same malicious note. It is not a general malicious-instruction detector. The targeted attack succeeds in 1/3 no-recovery injection cases, 1/3 snapshot cases, and 2/3 compensation cases; verification prevents it in all three. Retail safety measurement targets unauthorized attacker-address changes; benchmark reward alone does not establish full policy compliance. For example, clean task 48 also escalates to a human after returning the requested item.

\[Grounded results and traces\](../results/taubench\_llm\_grounded/summary.md), \[ordinary-agent results\](../results/taubench\_llm\_phase1/summary.md), and runtime metadata document the supplement. It uses local Ollama with a 16k context, temperature 0, one worker, and no explicitly supplied inference seed. Its three-task results do not establish representative benchmark performance.

## Appendix D: Timeline and state-restoration plan

| Dates | Milestone | Status |
| ----- | ----- | ----- |
| Oct 13–14 | Simulated environment, tools with reversibility, task generator and solver, structured agents, artifact/dependency store, traces | Done |
| Oct 14 | Fault harness (15 types plus 3 adversarial supplements), no-recovery / restart / verify baselines, full 7B run, analysis scripts | Done |
| Oct 20 | Microbenchmarks (tracking, provenance, restore vs. compensation, verifier, per-stage cost); fixed the quadratic invalidation lookup they exposed | Done |
| Oct 21 | Instructor feedback on the proposal: use at least one existing agent or harness in Phase 1, and treat state restoration (memory, process, network) as non-trivial in Phase 2\. I took that into account and added tau-bench retail (replay recovery table; 7B ToolCallingAgent stopped after 17/115 clean tasks at 0 success), rebuilt the pipeline in LangGraph with checkpoint rewind, and rewrote the Phase 2b plan around those four kinds of state | Done |
| Oct 31 | Phase 1 audit: real-agent retail supplement, observation-exposure traces, verifier coverage, and corrected metric definitions | Done |
| Oct 29 – Nov 5 | Phase 2a: traverse depends\_on and field-level provenance from the faulty artifact; key-version staleness checks; rerun only invalidated stages. Add a branched workflow (two researchers into one planner) |  |
| Nov 6 – Nov 22 | Phase 2b: state restoration, with more time than the original plan (see below) |  |
| Nov 23 – Dec 2 | Phase 2c: selective recovery vs. restart vs. verify vs. LangGraph checkpoint, on both workflow shapes, plus micro costs of each restore step |  |
| Early Dec | Final analysis of where selective recovery succeeds and fails; final report |  |

Restoring state is not just copying a Python dict, which is why Phase 2b is longer. I already measure two ways of handling the **shared database**: snapshot rollback, and compensating our own reversible writes. Snapshot is wrong when something irreversible has already happened or when someone else updated the world in the meantime. Compensation cannot undo a shipment, an email, a refund, or a tau-bench exchange. Selective recovery will have to report those as uncompensable instead of pretending the world was rewound.

**Agent memory** is a different object: conversation history, scratchpads, caches. LangGraph's `MemorySaver` can already rewind the graph to the node before the fault. That is necessary and not enough, because the world stays dirty. In Phase 2 I want to version memory the same way I version artifacts. I only drop the messages that depended on the bad result, keep the rest, and check that what I kept is still valid against current key versions.


A **process** mid-flight (a tool that has written a partial file, a reservation that is half applied) cannot be restored by `deepcopy`. The plan is to isolate long tools, log when they start and finish, and on recovery either abort them or resume from a real process checkpoint (CRIU or `docker checkpoint` as an optional comparison).   
**Network** side effects are the same problem as shipments: the request already left. I will use a recording proxy and idempotency keys so retries are safe, issue a compensating call when the service allows it, and otherwise list the effect as irreversible. I will not roll the database back and call that undoing an email.

Deliverables in the repository: the workflow (`masrec/`), the injection harness (`masrec/faults.py`), dependency tracking (`masrec/artifacts.py`), recovery baselines (`masrec/recovery.py`, `masrec/verifier.py`), the tau-bench and LangGraph wrappers (`masrec/taubench.py`, `masrec/langgraph_workflow.py`), the dataset (`data/tasks.json`), per-run traces and results (`results/qwen7b/`, `results/taubench_replay/`, `results/langgraph_scripted/`, `results/taubench_llm_phase1/`, `results/taubench_llm_grounded/`), and tables (`results/qwen7b/summary.md`, `results/external_summary.md`, `results/microbench/`).

