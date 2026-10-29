"""Microbenchmarks: cost of each recovery mechanism in isolation (no LLM needed).

The end-to-end experiment (run_experiment.py) is the macrobenchmark. This script measures the components
that selective recovery will be built from, and how their cost scales:

  M1  dependency tracking: recording an artifact, and finding everything downstream of a faulty one,
      as the dependency graph grows (chain vs. fan-out topologies)
  M2  field-level provenance: deriving per-action field dependencies and matching them to a corrupted fact
  M3  state restoration: snapshot/restore vs. compensating our own actions, as the world and the number of
      executed actions grow
  M4  rule-verifier checks (per artifact kind) on the 25 tasks
  M5  per-stage and per-check LLM cost, taken from the 7B macrobenchmark traces

Usage:  python microbench.py [--runs results/qwen7b] [--out results/microbench]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import time
from collections import defaultdict

from masrec.agents import ScriptedExecutor, ScriptedPlanner, ScriptedResearcher
from masrec.artifacts import Artifact, ArtifactStore, fact_matches, plan_field_deps
from masrec.tasks import load_tasks
from masrec.tools import ToolBox
from masrec.verifier import RuleVerifier
from masrec.world import World

REPS = 5


def timed(fn, reps=REPS):
    """Median wall time of ``fn()`` over ``reps`` repetitions, in seconds."""
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


# ---------------------------------------------------------------------------------------------- M1
def build_store(n: int, topology: str) -> ArtifactStore:
    s = ArtifactStore()
    for k in range(n):
        if k == 0:
            deps = []
        elif topology == "chain":
            deps = [f"a{k - 1}"]
        else:  # fan-out tree with branching factor 3 (parallel sub-agents)
            deps = [f"a{(k - 1) // 3}"]
        s.add(Artifact(f"a{k}", "research", "researcher", {"k": k}, depends_on=deps))
    return s


def m1_tracking(sizes):
    rows = []
    for topo in ("chain", "fanout3"):
        for n in sizes:
            add_s = timed(lambda: build_store(n, topo), reps=3)
            store = build_store(n, topo)
            branch = f"a{n // 2}" if topo == "chain" else "a1"
            root_s = timed(lambda: store.descendants("a0"))
            branch_s = timed(lambda: store.descendants(branch))
            rows.append({"topology": topo, "artifacts": n, "edges": store.n_edges(),
                         "record_us_per_artifact": 1e6 * add_s / n,
                         "descendants_root_ms": 1e3 * root_s, "invalidated_from_root": len(store.descendants("a0")),
                         "descendants_branch_ms": 1e3 * branch_s,
                         "invalidated_from_branch": len(store.descendants(branch))})
    return rows


# ---------------------------------------------------------------------------------------------- M2
def m2_provenance(tasks):
    plans = []
    for t in tasks:
        tb = ToolBox(World(t["initial_state"]), "researcher")
        facts = ScriptedResearcher().run(t, tb).content
        plans.append(ScriptedPlanner().run(t, facts).content["actions"])
    n_actions = sum(len(p) for p in plans)
    derive_s = timed(lambda: [plan_field_deps(p) for p in plans], reps=50)
    deps = [plan_field_deps(p) for p in plans]
    n_deps = sum(len(v) for d in deps for v in d.values())
    probe = "warehouses.warehouse_A.inventory.laptop"
    match_s = timed(lambda: [fact_matches(x, probe) for d in deps for v in d.values() for x in v], reps=50)
    return {"plans": len(plans), "actions": n_actions, "field_deps": n_deps,
            "derive_us_per_action": 1e6 * derive_s / n_actions, "match_us_per_dep": 1e6 * match_s / n_deps}


# ---------------------------------------------------------------------------------------------- M3
def synthetic_world(n_wh: int, n_items: int = 20) -> dict:
    items = [f"item_{i}" for i in range(n_items)]
    whs = {f"wh_{w}": {"region": "r", "status": "open", "inventory": {i: 10_000 for i in items}} for w in range(n_wh)}
    return {"warehouses": whs, "orders": {"order_1": {"customer": "c", "items": {items[0]: 10 ** 9}, "shipped": {},
                                                      "status": "pending", "note": ""}}}


def m3_restoration(wh_sizes, action_counts):
    snap_rows = []
    for n_wh in wh_sizes:
        w = World(synthetic_world(n_wh))
        snap = w.snapshot()
        snap_rows.append({"warehouses": n_wh, "inventory_keys": n_wh * 20,
                          "snapshot_ms": 1e3 * timed(w.snapshot), "restore_ms": 1e3 * timed(lambda: w.restore(snap))})
    comp_rows = []
    for k in action_counts:
        def run():
            w = World(synthetic_world(10))
            tb = ToolBox(w, "executor")
            for j in range(k):
                tb.call("reserve_inventory", {"order_id": "order_1", "warehouse": f"wh_{j % 10}",
                                              "item": "item_0", "quantity": 1})
            t = time.perf_counter()
            rtb = ToolBox(w, "recovery")
            for e in reversed([e for e in w.action_log if e["actor"] == "executor"]):
                rtb.compensate(e["compensation"])
            return time.perf_counter() - t
        comp_s = statistics.median(run() for _ in range(REPS))
        w = World(synthetic_world(10))
        s = w.snapshot()
        comp_rows.append({"executed_actions": k, "compensate_ms": 1e3 * comp_s,
                          "compensate_us_per_action": 1e6 * comp_s / k,
                          "snapshot_restore_ms_same_world": 1e3 * timed(lambda: w.restore(s))})
    return snap_rows, comp_rows


# ---------------------------------------------------------------------------------------------- M4
def m4_rule_verifier(tasks):
    v = RuleVerifier()
    acc = defaultdict(list)
    reads = defaultdict(list)
    for t in tasks:
        world = World(t["initial_state"])
        research = ScriptedResearcher().run(t, ToolBox(world, "researcher")).content
        plan = ScriptedPlanner().run(t, research).content
        calls = ScriptedExecutor().run(t, plan).content
        for kind, fn in (("research", lambda: v.check_research(t, research, world)),
                         ("plan", lambda: v.check_plan(t, plan, world)),
                         ("calls", lambda: v.check_calls(t, plan, calls, world))):
            acc[kind].append(timed(fn, reps=20))
            reads[kind].append(fn().reads)
    return [{"check": k, "median_us": 1e6 * statistics.median(x), "max_us": 1e6 * max(x),
             "env_reads": statistics.mean(reads[k])} for k, x in acc.items()]


# ---------------------------------------------------------------------------------------------- M5
def est_latency(pt, ct, cal):
    return pt / cal["pp_tok_per_s"] + ct / cal["tg_tok_per_s"]


def m5_llm_costs(runs_dir, cal):
    per_agent = defaultdict(lambda: [[], []])
    for f in glob.glob(os.path.join(runs_dir, "traces", "*__clean__s*.jsonl")):
        for line in open(f):
            e = json.loads(line)
            if e["type"] == "llm_call" and e.get("phase") == "main":
                per_agent[e["agent"]][0].append(e["prompt_tokens"])
                per_agent[e["agent"]][1].append(e["completion_tokens"])
    stages = []
    for a in ("researcher", "planner", "executor"):
        pt, ct = per_agent[a]
        if pt:
            mp, mc = statistics.mean(pt), statistics.mean(ct)
            stages.append({"stage": a, "calls": len(pt), "prompt_tokens": mp, "completion_tokens": mc,
                           "est_latency_s": est_latency(mp, mc, cal)})
    # LLM verifier per check = (verified clean run - clean run) / verifier calls, on runs with no retries
    rows = list(csv.DictReader(open(os.path.join(runs_dir, "runs.csv"))))
    clean = {r["task_id"]: r for r in rows if r["condition"] == "clean"}
    pts, cts = [], []
    for r in rows:
        if r["condition"] != "clean_verify_llm" or r["task_id"] not in clean or float(r["verify_failures"] or 0):
            continue
        n = int(float(r["verify_llm_calls"] or 0))
        if n:
            c = clean[r["task_id"]]
            pts.append((float(r["prompt_tokens"]) - float(c["prompt_tokens"])) / n)
            cts.append((float(r["completion_tokens"]) - float(c["completion_tokens"])) / n)
    check = None
    if pts:
        mp, mc = statistics.mean(pts), statistics.mean(cts)
        check = {"runs": len(pts), "prompt_tokens": mp, "completion_tokens": mc, "est_latency_s": est_latency(mp, mc, cal)}
    return stages, check


# ---------------------------------------------------------------------------------------------- report
def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.json")
    ap.add_argument("--runs", default="results/qwen7b")
    ap.add_argument("--out", default="results/microbench")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = load_tasks(a.tasks)
    cal = json.load(open("data/latency_calibration.json"))

    m1 = m1_tracking([10, 100, 1_000, 10_000])
    m2 = m2_provenance(tasks)
    m3s, m3c = m3_restoration([3, 10, 100, 1_000], [1, 10, 100, 1_000])
    m4 = m4_rule_verifier(tasks)
    m5s, m5c = m5_llm_costs(a.runs, cal)
    json.dump({"m1_tracking": m1, "m2_provenance": m2, "m3_snapshot": m3s, "m3_compensate": m3c,
               "m4_rule_verifier": m4, "m5_stage_llm": m5s, "m5_llm_verifier_check": m5c},
              open(os.path.join(a.out, "microbench.json"), "w"), indent=2)

    f2, f3 = "{:.2f}", "{:.3f}"
    parts = ["# Microbenchmarks", "",
             "Median of repeated runs on one laptop core. The workflow in the macrobenchmark has 5-12 artifacts, "
             "3-10 inventory keys per warehouse and 2-8 executed actions per run.", "",
             "## M1. Dependency tracking", "",
             "Branch = the middle of the chain, or the first child of the root in the fan-out tree "
             "(a fault in one of three parallel sub-agents).", "",
             md_table(m1, ["topology", "artifacts", "edges", "record_us_per_artifact", "descendants_root_ms",
                           "invalidated_from_root", "descendants_branch_ms", "invalidated_from_branch"],
                      {"record_us_per_artifact": f2, "descendants_root_ms": f3, "descendants_branch_ms": f3}), "",
             "## M2. Field-level provenance", "",
             f"{m2['actions']} plan actions over {m2['plans']} tasks, {m2['field_deps']} field dependencies. "
             f"Deriving dependencies: {m2['derive_us_per_action']:.2f} us/action. "
             f"Matching against a corrupted fact: {m2['match_us_per_dep']:.3f} us/dependency.", "",
             "## M3. State restoration", "",
             md_table(m3s, ["warehouses", "inventory_keys", "snapshot_ms", "restore_ms"],
                      {"snapshot_ms": f3, "restore_ms": f3}), "",
             md_table(m3c, ["executed_actions", "compensate_ms", "compensate_us_per_action",
                            "snapshot_restore_ms_same_world"],
                      {"compensate_ms": f3, "compensate_us_per_action": f2, "snapshot_restore_ms_same_world": f3}), "",
             "## M4. Rule verifier (per check, 25 tasks)", "",
             md_table(m4, ["check", "median_us", "max_us", "env_reads"],
                      {"median_us": "{:.1f}", "max_us": "{:.1f}", "env_reads": "{:.1f}"}), "",
             "## M5. LLM cost per stage and per LLM-verifier check (7B traces, token-based latency)", "",
             md_table(m5s, ["stage", "calls", "prompt_tokens", "completion_tokens", "est_latency_s"],
                      {"prompt_tokens": "{:.0f}", "completion_tokens": "{:.0f}", "est_latency_s": "{:.1f}"})]
    if m5c:
        parts += ["", f"LLM verifier, per check ({m5c['runs']} clean runs without retries): "
                      f"{m5c['prompt_tokens']:.0f} prompt + {m5c['completion_tokens']:.0f} completion tokens, "
                      f"about {m5c['est_latency_s']:.1f} s."]
    open(os.path.join(a.out, "microbench.md"), "w").write("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
