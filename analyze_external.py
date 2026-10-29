"""Tables for the existing-harness experiments: tau-bench retail and the LangGraph workflow.

  python analyze_external.py [--tau results/taubench_qwen7b] [--lg results/langgraph_qwen7b]
                             [--native results/qwen7b] [--out results/external_summary.md]
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

PP, TG = 745.2, 23.9          # measured single-stream throughput (data/latency_calibration.json)


def est_latency(df, prefix=""):
    return df[f"{prefix}prompt_tokens"] / PP + df[f"{prefix}completion_tokens"] / TG


def md(df, floatfmt=".2f"):
    return df.to_markdown(floatfmt=floatfmt)


def tau_section(d):
    meta = {}
    mp = os.path.join(d, "meta_clean.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    agent = meta.get("agent", "llm")
    label = "replay of ground-truth actions" if agent == "replay" else "tau-bench's ToolCallingAgent"
    out = [f"## tau-bench retail ({label}, scripted customer)", ""]
    c = pd.read_csv(os.path.join(d, "clean.csv"))
    c["est_latency_s"] = est_latency(c)
    out += [f"Clean: {int(c.success.sum())}/{len(c)} test tasks solved (pass^1 = {c.success.mean():.2f}); "
            f"{c.llm_calls.mean():.1f} LLM calls, {c.prompt_tokens.mean() / 1e3:.1f}k prompt tokens, "
            f"~{c.est_latency_s.mean():.0f} s per task (token-based); errors: {int((c.error.fillna('') != '').sum())}.", ""]
    path = os.path.join(d, "runs.csv")
    if not os.path.exists(path):
        return out + ["(fault runs not available yet)", ""]
    r = pd.read_csv(path)
    r["est_latency_s"] = est_latency(r)
    n_tasks = r.task_index.nunique()
    out += [f"Fault runs on {n_tasks} test tasks that have a ground-truth write ({len(r)} runs).", ""]
    nr = r[r.condition == "no_recovery"]
    t = nr.groupby("fault").agg(tasks=("task_index", "count"), applied=("fault_applied", "mean"),
                                success=("success", "mean"), actions_changed=("actions_changed", "mean"),
                                state_changed=("state_changed", "mean"), outcome_changed=("outcome_changed", "mean"),
                                irreversible_dev=("irreversible_deviations", "mean"),
                                attack_success=("attack_success", "mean"))
    out += ["### Propagation (no recovery)", "", md(t), ""]
    app = r[r.fault_applied == 1]
    s = app.groupby(["fault", "condition"]).success.mean().unstack()
    out += ["### Success by fault and condition (applied faults)", "", md(s), ""]
    rs = app[app.condition.str.startswith("restart")]
    if len(rs):
        k = rs.groupby("condition").agg(success=("success", "mean"), strict=("strict_recovered", "mean"),
                                        irreversible_before=("faulted_n_irreversible", "mean"),
                                        falsely_undone=("irreversible_falsely_undone", "mean"),
                                        uncompensable=("uncompensable", "mean"),
                                        total_llm_calls=("total_llm_calls", "mean"))
        out += ["### Full restart", "", md(k), ""]
    v = app[app.condition == "verify_rule"]
    if len(v):
        harmful = nr[(nr.fault_applied == 1) & (nr.outcome_changed == 1)][["task_index", "fault"]]
        vh = v.merge(harmful, on=["task_index", "fault"])
        out += [f"### Rule verifier", "",
                f"Detected {vh.detected.mean():.2f} of outcome-changing faults ({len(vh)}); "
                f"success {v.success.mean():.2f}; {v.llm_calls.mean():.1f} LLM calls vs "
                f"{nr.llm_calls.mean():.1f} without verification.", ""]
    cost = app.groupby("condition").agg(llm_calls=("llm_calls", "mean"), est_latency_s=("est_latency_s", "mean"))
    out += ["### Cost per run (restart rows: rerun only)", "", md(cost, ".1f"), ""]
    return out


def lg_section(d, native):
    out = ["## LangGraph implementation of the workflow (same agents, tasks, faults)", ""]
    path = os.path.join(d, "runs.csv")
    if not os.path.exists(path):
        return out + ["(not available yet)", ""]
    r = pd.read_csv(path)
    c = r[r.condition == "clean"]
    out += [f"Clean success {c.success.mean():.2f} ({len(c)} tasks); {c.checkpoints.mean():.1f} checkpoints per run.", ""]
    a = r[(r.condition != "clean") & (r.fault_applied == 1)]
    t = a.groupby(["fault_stage", "condition"]).agg(success=("success", "mean")).unstack()
    out += ["### Success by faulty stage", "", md(t), ""]
    rc = a[a.condition.str.startswith("restart")]
    k = rc.groupby("condition").agg(success=("success", "mean"), strict=("strict_recovered", "mean"),
                                    recomputed_stages=("recomputed_stages", "mean"),
                                    retained_stages=("retained_stages", "mean"),
                                    over_shipped=("over_shipped", "mean"),
                                    dangling=("dangling_reservations", "mean"),
                                    recovery_llm_calls=("recovery_llm_calls", "mean"))
    if native and os.path.exists(os.path.join(native, "runs.csv")):
        n = pd.read_csv(os.path.join(native, "runs.csv"), low_memory=False)
        n = n[n.condition.isin(["restart_snapshot", "restart_compensate"]) & (n.faulted_fault_applied == 1)]
        shared = set(n.fault_id) & set(rc.fault_id)
        n = n[n.fault_id.isin(shared)]
        k = rc[rc.fault_id.isin(shared)].groupby("condition").agg(
            success=("success", "mean"), strict=("strict_recovered", "mean"),
            recomputed_stages=("recomputed_stages", "mean"), retained_stages=("retained_stages", "mean"),
            over_shipped=("over_shipped", "mean"), dangling=("dangling_reservations", "mean"),
            recovery_llm_calls=("recovery_llm_calls", "mean"))
        nk = n.groupby("condition").agg(success=("success", "mean"), strict=("strict_recovered", "mean"),
                                        recomputed_stages=("recomputed_stages", "mean"),
                                        retained_stages=("retained_valid_stages", "mean"),
                                        recovery_llm_calls=("recovery_llm_calls", "mean"))
        k = pd.concat([k, nk.rename(index=lambda x: f"native_{x}")])
    out += ["### Checkpoint rewind vs. full restart", "", md(k), ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="results/taubench_qwen7b")
    ap.add_argument("--lg", default="results/langgraph_qwen7b")
    ap.add_argument("--native", default="results/qwen7b")
    ap.add_argument("--out", default="results/external_summary.md")
    a = ap.parse_args()
    lines = ["# Existing benchmark and framework experiments", ""]
    lines += tau_section(a.tau) if os.path.exists(a.tau) else []
    lines += lg_section(a.lg, a.native)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
