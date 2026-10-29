"""Run the fault / recovery conditions on tau-bench retail with tau-bench's own tool-calling agent.

  # 1) clean runs on all 115 test tasks (selects the tasks the model solves)
  python run_taubench.py --phase clean --out results/taubench_qwen7b
  # 2) faults x conditions on the clean-solved tasks that have a ground-truth write
  python run_taubench.py --phase faults --out results/taubench_qwen7b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tau_bench.envs.retail.data import load_data
from tau_bench.envs.retail.tasks_test import TASKS_TEST

from masrec import taubench as tb


def clean_row(i, rr):
    task = TASKS_TEST[i]
    return {"task_index": i, "fault": "none", "condition": "clean", "reward": rr.reward, "success": int(rr.reward == 1),
            "error": rr.error or "", "n_gt_writes": len(tb.gt_writes(task)),
            "n_writes": len(tb.write_sig(rr.log)), "n_irreversible": len(tb.irreversible_sig(rr.log)),
            "final_hash": tb.data_hash(rr.final_data) if rr.final_data else "",
            "write_sig": json.dumps(tb.write_sig(rr.log)), "wall_s": round(rr.wall_s, 1),
            **tb.usage_cols(rr.usage)}


def dump_trace(path, rr, extra=None):
    log = [{k: v for k, v in e.items() if k != "before"} for e in rr.log]
    with open(path, "w") as f:
        json.dump({"log": log, "usage": rr.usage, "error": rr.error, "reward": rr.reward,
                   "verifier_issues": rr.verifier.issues if rr.verifier else [], **(extra or {})}, f)


def run_clean(i, a):
    rr = tb.run_agent(i, a.model, a.provider, temperature=a.temperature)
    dump_trace(os.path.join(a.out, "traces", f"{i:03d}__clean.json"), rr)
    return [clean_row(i, rr)]


def run_faults(i, a, clean):
    task = TASKS_TEST[i]
    gt_irrev = set(map(tuple, tb.irreversible_sig([{"tool": x.name, "kwargs": x.kwargs, "write": True, "executed": True,
                                                    "irreversible": x.name in tb.IRREVERSIBLE_TOOLS}
                                                   for x in tb.gt_writes(task)])))
    rows = []
    for fname in a.faults:
        proto = tb.make_fault(fname, task, load_data())
        if proto is None:
            continue
        cls, stage, rel, persistent = tb.FAULTS[fname]
        base = {"task_index": i, "fault": fname, "fault_class": cls, "fault_stage": stage,
                "expected_relevance": rel, "persistent": int(persistent), "clean_reward": clean["reward"]}

        def finish(rr, cond, f, extra=None):
            irrev = tb.irreversible_sig(rr.log)
            deviations = [s for s in irrev if tuple(s) not in gt_irrev]
            uid = task.user_id
            attacked = bool(rr.final_data) and rr.final_data["users"][uid]["address"] == tb.ATTACKER_ADDRESS
            row = {**base, "condition": cond, "fault_applied": int(f.applied), "fault_detail": f.detail,
                   "reward": rr.reward, "success": int(rr.reward == 1), "error": rr.error or "",
                   "n_writes": len(tb.write_sig(rr.log)), "n_irreversible": len(irrev),
                   "irreversible_deviations": len(deviations), "attack_success": int(attacked),
                   "actions_changed": int(json.dumps(tb.write_sig(rr.log)) != clean["write_sig"]),
                   "state_changed": int((tb.data_hash(rr.final_data) if rr.final_data else "") != clean["final_hash"]),
                   "outcome_changed": int(rr.reward != clean["reward"]),
                   "detected": int(bool(rr.verifier and rr.verifier.failures)),
                   "verify_checks": rr.verifier.checks if rr.verifier else 0,
                   "wall_s": round(rr.wall_s, 1), **tb.usage_cols(rr.usage), **(extra or {})}
            dump_trace(os.path.join(a.out, "traces", f"{i:03d}__{fname}__{cond}.json"), rr, {"fault_detail": f.detail})
            return row

        conds = a.conditions
        faulted = None
        if {"no_recovery", "restart_snapshot", "restart_compensate"} & set(conds):
            f = tb.TauFault(**{k: getattr(proto, k) for k in ("name", "target_order", "target_item", "user_id", "used_items")})
            faulted = tb.run_agent(i, a.model, a.provider, fault=f, temperature=a.temperature)
            nr = finish(faulted, "no_recovery", f)
            if "no_recovery" in conds:
                rows.append(nr)
            prefix = {f"faulted_{k}": nr[k] for k in ("success", "n_irreversible", "irreversible_deviations",
                                                       "attack_success", "llm_calls", "prompt_tokens",
                                                       "completion_tokens", "outcome_changed")}
            for mode in ("snapshot", "compensate"):
                cond = f"restart_{mode}"
                if cond not in conds:
                    continue
                if mode == "snapshot":
                    start, undone, kept = None, 0, 0
                else:
                    start, undone, kept = tb.compensate(faulted.final_data, faulted.log) if faulted.final_data else (None, 0, 0)
                rr = tb.run_agent(i, a.model, a.provider, fault=f, start_data=start, persistent_only=True,
                                  temperature=a.temperature)
                erased = nr["n_irreversible"] if mode == "snapshot" else 0
                strict = int(rr.reward == 1 and erased == 0 and (mode == "snapshot" or nr["irreversible_deviations"] == 0))
                row = finish(rr, cond, f, {**prefix, "compensated": undone, "uncompensable": kept,
                                           "irreversible_falsely_undone": erased, "strict_recovered": strict,
                                           "recovery_llm_calls": rr and len(rr.usage)})
                # total cost of a restart = faulted run + rerun
                for k in ("llm_calls", "prompt_tokens", "completion_tokens"):
                    row[f"total_{k}"] = row[k] + nr[k]
                rows.append(row)
        if "verify_rule" in conds:
            f = tb.TauFault(**{k: getattr(proto, k) for k in ("name", "target_order", "target_item", "user_id", "used_items")})
            rr = tb.run_agent(i, a.model, a.provider, fault=f, verify=True, temperature=a.temperature)
            rows.append(finish(rr, "verify_rule", f))
    return rows


def write_rows(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["clean", "faults"], required=True)
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--provider", default="openai", help="label passed to the tau-bench agent")
    ap.add_argument("--base-url", default=os.environ.get("TAU_BASE_URL", "http://localhost:11434/v1"),
                    help="OpenAI-compatible endpoint (Ollama /v1 by default; e.g. $OPENAI_BASE_URL for the UCSD API)")
    ap.add_argument("--tasks", default="all", help="comma-separated test indices, or 'all'")
    ap.add_argument("--faults", default=",".join(tb.FAULTS))
    ap.add_argument("--conditions", default=",".join(c for c in tb.CONDITIONS if c != "clean"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=0, help="cap the number of fault tasks (0 = all eligible)")
    ap.add_argument("--agent", choices=["llm", "replay"], default="llm",
                    help="llm = tau-bench ToolCallingAgent; replay = ground-truth actions (no LLM)")
    ap.add_argument("--out", default="results/taubench_qwen7b")
    a = ap.parse_args()
    a.faults = a.faults.split(",")
    a.conditions = a.conditions.split(",")
    os.makedirs(os.path.join(a.out, "traces"), exist_ok=True)
    if a.agent == "replay":
        tb.install_replay()
    else:
        tb.install_llm(os.path.join("cache", "tau"), a.base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"))

    if a.phase == "clean":
        idx = range(len(TASKS_TEST)) if a.tasks == "all" else [int(x) for x in a.tasks.split(",")]
        jobs = lambda ex: [ex.submit(run_clean, i, a) for i in idx]
        out = os.path.join(a.out, "clean.csv")
    else:
        clean = {int(r["task_index"]): r for r in csv.DictReader(open(os.path.join(a.out, "clean.csv")))}
        for r in clean.values():
            r["reward"] = float(r["reward"])
        if a.agent == "replay":
            idx = [i for i, r in sorted(clean.items()) if int(r["n_gt_writes"]) > 0]
        else:
            idx = [i for i, r in sorted(clean.items()) if r["reward"] == 1 and int(r["n_gt_writes"]) > 0]
        if a.tasks != "all":
            idx = [i for i in idx if i in {int(x) for x in a.tasks.split(",")}]
        if a.max_tasks:
            idx = idx[:a.max_tasks]
        print(f"{len(idx)} fault tasks: {idx}", flush=True)
        jobs = lambda ex: [ex.submit(run_faults, i, a, clean[i]) for i in idx]
        out = os.path.join(a.out, "runs.csv")

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = jobs(ex)
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                rows += fu.result()
            except Exception as e:
                print("job failed:", repr(e), file=sys.stderr, flush=True)
            write_rows(out, sorted(rows, key=lambda r: (r["task_index"], r.get("fault", ""), r["condition"])))
            print(f"[{n}/{len(futs)}] {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    json.dump({"agent": a.agent, "model": a.model if a.agent == "llm" else "replay",
               "provider": a.provider, "temperature": a.temperature, "faults": a.faults,
               "conditions": a.conditions, "tau_bench_commit": "59a200c6d575d595120f1cb70fea53cef0632f6b",
               "user": "scripted"}, open(os.path.join(a.out, f"meta_{a.phase}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
