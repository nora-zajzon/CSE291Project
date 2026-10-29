"""Experiment harness: identical fault schedules across all conditions, one CSV row per run.

Conditions
  clean               no fault, no verification
  clean_verify_rule   no fault, rule verifier (overhead + false positives)
  clean_verify_llm    no fault, LLM verifier
  no_recovery         fault injected, workflow runs to completion
  restart_snapshot    no_recovery, then (oracle-identified fault) DB snapshot restore + full rerun
  restart_compensate  no_recovery, then undo own reversible actions + full rerun on current world
  verify_rule         fault injected, rule verifier checks each artifact before it is consumed
  verify_llm          fault injected, LLM verifier checks each artifact before it is consumed
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass

from .artifacts import ArtifactStore
from .evaluator import (agent_fingerprint, evaluate_outcome, executed_sig, irreversible_deviations, irreversible_sig,
                        plan_sig, propagation, research_sig)
from .faults import FaultSpec, Injector, build_schedule
from .recovery import full_restart
from .trace import Trace
from .workflow import Workflow
from .world import World

ALL_CONDITIONS = ["clean", "clean_verify_rule", "clean_verify_llm", "no_recovery", "restart_snapshot",
                  "restart_compensate", "verify_rule", "verify_llm"]


@dataclass
class Ctx:
    agents: tuple
    verifiers: dict
    trace_dir: str
    timeout_budget_s: float = 30.0
    max_retries: int = 2
    seed: int = 0
    backend: str = "scripted"


def _base_row(task, spec: FaultSpec | None, condition: str, ctx: Ctx) -> dict:
    return {
        "task_id": task["task_id"], "category": task["category"], "condition": condition, "seed": ctx.seed,
        "backend": ctx.backend,
        "fault_id": spec.fault_id if spec else "none",
        "fault_name": spec.fault_id.split("@")[0] if spec else "none",
        "fault_type": spec.fault_type if spec else "none",
        "fault_class": spec.fault_class if spec else "none",
        "fault_stage": spec.stage if spec else "none",
        "expected_relevance": spec.expected_relevance if spec else "none",
        "persistent": int(spec.persistent) if spec else 0,
    }


def _stage_cols(res, prefix="") -> dict:
    d = {}
    for s, st in res.stages.items():
        d[f"{prefix}{s}_latency_s"] = round(st.latency_s, 4)
        d[f"{prefix}{s}_status"] = st.status
        d[f"{prefix}{s}_attempts"] = st.attempts
    d[f"{prefix}latency_s"] = round(res.latency_s, 4)
    d[f"{prefix}timed_out"] = int(res.timed_out_stage is not None)
    d[f"{prefix}llm_calls"] = res.llm_calls
    d[f"{prefix}tokens"] = res.tokens
    d[f"{prefix}prompt_tokens"] = sum(s.prompt_tokens for s in res.stages.values()) + res.verify.prompt_tokens
    d[f"{prefix}completion_tokens"] = (sum(s.completion_tokens for s in res.stages.values())
                                       + res.verify.completion_tokens)
    return d


def run_once(task: dict, spec: FaultSpec | None, verifier_name: str | None, ctx: Ctx, run_id: str):
    world = World(task["initial_state"])
    inj = Injector(spec)
    inj.prepare_world(world, task)
    snapshot = world.snapshot()
    store = ArtifactStore()
    trace = Trace(run_id)
    trace.log("run_start", task_id=task["task_id"], fault=spec.to_dict() if spec else None,
              verifier=verifier_name, initial_world_version=world.version)
    t = time.perf_counter()
    wf = Workflow(ctx.agents, store, trace, verifier=ctx.verifiers.get(verifier_name),
                  max_retries=ctx.max_retries, timeout_budget_s=ctx.timeout_budget_s, phase="main")
    res = wf.run(task, world, inj)
    wall = time.perf_counter() - t
    for ev in inj.events:
        trace.log("fault_injected", **ev)
    return world, inj, snapshot, store, trace, res, wall


def clean_reference(world, res, outcome) -> dict:
    return {"research_sig": research_sig(res.research), "plan_sig": plan_sig(res.plan),
            "exec_sig": executed_sig(world, "main"), "fingerprint": agent_fingerprint(world),
            "success": outcome["success"], "irrev": irreversible_sig(world, "main")}


def _finish_row(row, world, inj, store, trace, res, wall, spec, clean, ctx):
    out = evaluate_outcome(_task_of(row), world, {"main"})
    row.update(out)
    row.update(_stage_cols(res))
    row["aborted"] = res.aborted or ""
    row["wall_s"] = round(wall, 3)
    v = res.verify
    row.update({"verify_checks": v.checks, "verify_failures": v.failures, "verify_latency_s": round(v.latency_s, 4),
                "verify_reads": v.reads, "verify_llm_calls": v.llm_calls,
                "detected": int(bool(v.detected_stages)), "detected_stages": ",".join(v.detected_stages),
                "verify_issues": " | ".join(v.issues)[:500]})
    if clean is not None and spec is not None:
        row.update(propagation(spec, inj, store, res, world, clean, out))
        row["irreversible_deviations"] = len(irreversible_deviations(world, clean["irrev"]))
    row["artifacts"] = len(store.artifacts)
    row["dependency_edges"] = store.n_edges()
    row["bookkeeping_s"] = round(store.bookkeeping_s + trace.bookkeeping_s, 6)
    trace.log("run_end", metrics={k: v for k, v in row.items() if not k.startswith("_")})
    row["trace_bytes"] = trace.write(ctx.trace_dir)
    row["run_id"] = trace.run_id
    return row


_TASKS: dict[str, dict] = {}


def _task_of(row) -> dict:
    return _TASKS[row["task_id"]]


def run_task(task: dict, ctx: Ctx, conditions: list[str], fault_filter: list[str] | None = None) -> list[dict]:
    _TASKS[task["task_id"]] = task
    rows = []
    tid = task["task_id"]
    sfx = f"s{ctx.seed}"

    # ---- clean reference (always run; needed for counterfactual comparisons)
    w, inj, snap, store, tr, res, wall = run_once(task, None, None, ctx, f"{tid}__clean__{sfx}")
    out = evaluate_outcome(task, w, {"main"})
    clean = clean_reference(w, res, out)
    clean["latency_s"] = res.latency_s
    if "clean" in conditions:
        rows.append(_finish_row(_base_row(task, None, "clean", ctx), w, inj, store, tr, res, wall, None, None, ctx))
    for vname in ("rule", "llm"):
        cond = f"clean_verify_{vname}"
        if cond in conditions and vname in ctx.verifiers:
            r = run_once(task, None, vname, ctx, f"{tid}__{cond}__{sfx}")
            row = _finish_row(_base_row(task, None, cond, ctx), *r[:2], *r[3:], None, None, ctx)
            row["false_positive"] = int(row["detected"])
            rows.append(row)

    specs = build_schedule(task)
    if fault_filter:
        specs = [s for s in specs if any(f in s.fault_id for f in fault_filter)]
    for spec in specs:
        fid = spec.fault_id.replace("@", "_at_")
        need_nr = any(c in conditions for c in ("no_recovery", "restart_snapshot", "restart_compensate"))
        if need_nr:
            w, inj, snap, store, tr, res, wall = run_once(task, spec, None, ctx, f"{tid}__{fid}__no_recovery__{sfx}")
            base_world, base_store, base_trace = w.copy(), copy.deepcopy(store), copy.deepcopy(tr)
            nr_row = _finish_row(_base_row(task, spec, "no_recovery", ctx), w, inj, store, tr, res, wall, spec,
                                 clean, ctx)
            if "no_recovery" in conditions:
                rows.append(nr_row)
            for mode in ("snapshot", "compensate"):
                cond = f"restart_{mode}"
                if cond not in conditions:
                    continue
                w2, st2, tr2 = base_world.copy(), copy.deepcopy(base_store), copy.deepcopy(base_trace)
                tr2.run_id = f"{tid}__{fid}__{cond}__{sfx}"
                rec = full_restart(task, w2, spec, ctx.agents, st2, tr2, snap, mode, ctx.timeout_budget_s)
                row = _base_row(task, spec, cond, ctx)
                faulted = {k: nr_row[k] for k in nr_row if k in (
                    "success", "unauthorized_executed", "unauthorized_attempted", "irreversible_deviations",
                    "irreversible_actions", "latency_s", "llm_calls", "tokens", "fault_applied", "fault_consumed",
                    "downstream_changed", "actions_changed", "state_changed", "outcome_changed", "affected_stages",
                    "wasted_stages", "discarded_valid_stages_on_restart", "faulty_artifact")}
                row.update({f"faulted_{k}": v for k, v in faulted.items()})
                rout = evaluate_outcome(task, w2, {"recovery"})
                rr = rec["result"]
                strict = rout["success"] and nr_row.get("irreversible_deviations", 0) == 0
                if mode == "snapshot":
                    strict = strict and rec["lost_external_updates"] == 0 and rec["irreversible_falsely_undone"] == 0
                row.update({f"recovered_{k}": v for k, v in rout.items()})
                row.update({
                    "success": rout["success"],
                    "strict_recovered": int(bool(strict)),
                    "recomputed_stages": rec["recomputed_stages"],
                    "retained_valid_stages": rec["retained_valid_stages"],
                    "recovery_latency_s": round(rec["recovery_latency_s"], 4),
                    "recovery_llm_calls": rec["recovery_llm_calls"],
                    "recovery_tokens": rec["recovery_tokens"],
                    "compensations": rec["compensations"],
                    "uncompensable_actions": rec["uncompensable_actions"],
                    "irreversible_before_recovery": rec["irreversible_before_recovery"],
                    "lost_external_updates": rec["lost_external_updates"],
                    "irreversible_falsely_undone": rec["irreversible_falsely_undone"],
                    "latency_s": round(nr_row["latency_s"] + rec["recovery_latency_s"], 4),
                    "llm_calls": nr_row["llm_calls"] + rec["recovery_llm_calls"],
                    "tokens": nr_row["tokens"] + rec["recovery_tokens"],
                    "prompt_tokens": nr_row["prompt_tokens"] + sum(s.prompt_tokens for s in rr.stages.values()),
                    "completion_tokens": nr_row["completion_tokens"] + sum(s.completion_tokens
                                                                           for s in rr.stages.values()),
                    "timed_out": nr_row["timed_out"],
                    "recovery_prompt_tokens": sum(s.prompt_tokens for s in rr.stages.values()),
                    "recovery_completion_tokens": sum(s.completion_tokens for s in rr.stages.values()),
                    "unauthorized_executed": nr_row["unauthorized_executed"] + rout["unauthorized_executed"],
                    "aborted": rr.aborted or "",
                })
                tr2.log("run_end", metrics=row)
                row["trace_bytes"] = tr2.write(ctx.trace_dir)
                row["run_id"] = tr2.run_id
                rows.append(row)
        for vname in ("rule", "llm"):
            cond = f"verify_{vname}"
            if cond in conditions and vname in ctx.verifiers:
                r = run_once(task, spec, vname, ctx, f"{tid}__{fid}__{cond}__{sfx}")
                w, inj, snap, store, tr, res, wall = r
                row = _finish_row(_base_row(task, spec, cond, ctx), w, inj, store, tr, res, wall, spec, clean, ctx)
                row["extra_latency_vs_clean_s"] = round(row["latency_s"] - clean["latency_s"], 4)
                rows.append(row)
    return rows


def dump_schedule(tasks) -> list[dict]:
    out = []
    for t in tasks:
        for s in build_schedule(t):
            out.append({"task_id": t["task_id"], "category": t["category"], **s.to_dict()})
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
