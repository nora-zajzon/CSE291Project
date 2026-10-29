"""Full-restart recovery baselines, run after the faulty artifact has been identified (oracle).

Two variants make the "the environment moved on" problem measurable:

* ``snapshot``   - restore the whole simulated database to its pre-run snapshot (VM/container style),
                   then rerun all stages. Concurrent external updates are lost, and irreversible
                   effects (shipments, emails, refunds) are erased from the DB but not from reality.
* ``compensate`` - undo only our own reversible actions with compensating calls, keep external
                   updates and irreversible effects, then rerun all stages against the current world.
"""
from __future__ import annotations

import time

from .faults import Injector
from .tools import ToolBox
from .workflow import Workflow


def full_restart(task, world, spec, agents, store, trace, snapshot: dict, mode: str,
                 timeout_budget_s: float = 30.0) -> dict:
    t = time.perf_counter()
    main_acts = [e for e in world.action_log if e["actor"] == "executor" and e.get("phase") == "main" and e["ok"]]
    irreversible = [e for e in main_acts if e["irreversible_effect"]]
    external = [e for e in world.action_log if e["actor"] == "external"]
    compensated = 0
    uncompensable = 0
    trace.log("recovery_start", mode=mode, faulty_stage=spec.stage if spec else None)
    if mode == "snapshot":
        world.restore(snapshot)
        lost_external = len(external)
        falsely_undone = len(irreversible)
    elif mode == "compensate":
        tb = ToolBox(world, "recovery")
        for e in reversed(main_acts):
            if e["reversible"] and e.get("compensation"):
                r = tb.compensate(e["compensation"])
                compensated += int(r.ok)
                uncompensable += int(not r.ok)
                trace.log("compensation", of_seq=e["seq"], tool=e["compensation"]["tool"], ok=r.ok, error=r.error)
            elif not e["reversible"]:
                uncompensable += 1
        lost_external = 0
        falsely_undone = 0
    else:
        raise ValueError(mode)
    restore_s = time.perf_counter() - t
    # transient faults do not recur; persistent ones are already part of the world state
    wf = Workflow(agents, store, trace, verifier=None, timeout_budget_s=timeout_budget_s, phase="recovery")
    rr = wf.run(task, world, Injector(spec, enabled=False))
    return {
        "result": rr,
        "restore_s": restore_s,
        "recovery_latency_s": restore_s + rr.latency_s,
        "recovery_llm_calls": rr.llm_calls,
        "recovery_tokens": rr.tokens,
        "recomputed_stages": sum(1 for s in rr.stages.values() if s.attempts > 0),
        "retained_valid_stages": 0,
        "compensations": compensated,
        "uncompensable_actions": uncompensable,
        "irreversible_before_recovery": len(irreversible),
        "lost_external_updates": lost_external,
        "irreversible_falsely_undone": falsely_undone,
    }
