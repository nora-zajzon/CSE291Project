"""Aggregate runs.csv into tables (summary.md) and figures.

  python analyze_results.py results/qwen7b
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAULT_ORDER = [
    "incorrect_inflate@researcher", "incorrect_deflate@researcher", "incorrect_order_qty@researcher",
    "incorrect_irrelevant@researcher", "plan_qty_error@planner", "exec_wrong_warehouse@executor",
    "stale_research_relevant@researcher", "stale_research_irrelevant@researcher", "stale_plan_relevant@planner",
    "timeout_researcher@researcher", "timeout_planner@planner", "timeout_executor@executor",
    "inject_customer_note@researcher", "inject_upsell_note@researcher", "compromised_researcher@researcher",
    "compromised_researcher_facts@researcher", "compromised_planner@planner", "compromised_planner_foreign@planner",
]
COND_ORDER = ["no_recovery", "restart_snapshot", "restart_compensate", "verify_rule", "verify_llm"]
TYPE_ORDER = ["incorrect_result", "stale_output", "timeout", "malicious_instruction"]


def meta_path(out: Path) -> Path:
    return out / "meta.json"


def md(df: pd.DataFrame, floatfmt=".2f") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    cols = [str(c) for c in df.columns]
    idx = df.index.names if any(df.index.names) else [""]
    head = "| " + " | ".join([" / ".join(str(i) for i in idx if i)] + cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for k, r in df.iterrows():
        k = " / ".join(map(str, k)) if isinstance(k, tuple) else str(k)
        rows.append("| " + " | ".join([k] + [str(v) for v in r.values]) + " |")
    return "\n".join([head, sep] + rows)


def _ordered(index, order):
    return [x for x in order if x in index] + [x for x in index if x not in order]


def main(path: str, extra: list[str] = ()):
    out = Path(path)
    d = pd.concat([pd.read_csv(out / "runs.csv")] + [pd.read_csv(Path(p) / "runs.csv") for p in extra],
                  ignore_index=True)
    d = d.drop_duplicates(subset=["task_id", "fault_id", "condition", "seed"], keep="last")
    d = d[d.condition != "harness_error"].copy()
    cal_p = Path("data/latency_calibration.json")
    cal = json.loads(cal_p.read_text()) if cal_p.exists() else {"pp_tok_per_s": 745.0, "tg_tok_per_s": 24.0}
    budget = 30.0
    if meta_path(out).exists():
        budget = json.loads(meta_path(out).read_text()).get("timeout_budget_s", 30.0)

    def est(pt, ct, to):
        return pt / cal["pp_tok_per_s"] + ct / cal["tg_tok_per_s"] + budget * to

    rule_verify_s = d.get("verify_latency_s", pd.Series(0, index=d.index)).fillna(0).where(
        d.get("verify_llm_calls", pd.Series(0, index=d.index)).fillna(0) == 0, 0)
    d["est_latency_s"] = est(d.prompt_tokens.fillna(0), d.completion_tokens.fillna(0), d.timed_out.fillna(0)) \
        + rule_verify_s
    if "recovery_prompt_tokens" in d:
        d["est_recovery_latency_s"] = est(d.recovery_prompt_tokens.fillna(0), d.recovery_completion_tokens.fillna(0), 0)
    if d.prompt_tokens.fillna(0).sum() == 0:  # scripted backend: no inference, use measured
        d["est_latency_s"] = d.latency_s
        d["est_recovery_latency_s"] = d.get("recovery_latency_s")
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    S: list[str] = [f"# Phase 1 results: `{out.name}`\n"]
    meta_p = out / "meta.json"
    if meta_p.exists():
        S.append("```\n" + meta_p.read_text() + "\n```\n")

    # ------------------------------------------------------------------ clean baseline
    clean = d[d.condition.str.startswith("clean")]
    t1 = clean.groupby("condition").agg(
        runs=("success", "size"), success=("success", "mean"), optimal=("optimal", "mean"),
        est_latency_s=("est_latency_s", "mean"), measured_latency_s=("latency_s", "mean"),
        llm_calls=("llm_calls", "mean"), tokens=("tokens", "mean"),
        verify_checks=("verify_checks", "mean"), verify_reads=("verify_reads", "mean"),
        clean_detection_rate=("detected", "mean"), unauthorized=("unauthorized_executed", "mean"))
    t1 = t1.loc[_ordered(t1.index, ["clean", "clean_verify_rule", "clean_verify_llm"])]
    S += ["## 1. Clean baseline and the fixed cost of always verifying\n", md(t1), "",
          "Clean detection rate is not a false-positive rate: uninjected runs can contain natural model errors. "
          "Verifier decisions have not all been independently adjudicated.\n",
          "`measured_latency_s` is serial service-time accounting (original call times on cache hits), "
          "including simulated timeout charges. `wall_s` in runs.csv is observed harness elapsed time. "
          "`est_latency_s` is a token-throughput estimate, not a measured end-to-end time.\n"]
    cl = d[d.condition == "clean"]
    S += ["Clean success by task category:\n",
          md(cl.groupby("category")[["success", "optimal", "est_latency_s"]].mean()), ""]
    failed = cl[cl.success == 0]
    if len(failed):
        S.append(f"Clean failures: {', '.join(failed.task_id)}\n")
    clean_ok = set(cl[cl.success == 1].task_id)

    f = d[~d.condition.str.startswith("clean")].copy()
    nr = f[f.condition == "no_recovery"]
    applied_ids = set(zip(nr[nr.fault_applied == 1].task_id, nr[nr.fault_applied == 1].fault_id))
    f["applied"] = [(a, b) in applied_ids for a, b in zip(f.task_id, f.fault_id)]
    fa = f[f.applied]
    nra = fa[fa.condition == "no_recovery"]
    S.append(f"Fault runs: {len(nr)} scheduled per condition, {len(nra)} applied "
             f"(non-applied faults are no-ops, e.g. no reserve call to corrupt) and excluded below.\n")

    # ------------------------------------------------------------------ propagation funnel
    funnel_cols = ["fault_consumed", "faulty_field_used", "downstream_changed", "actions_changed", "state_changed",
                   "outcome_changed"]
    t3 = nra.groupby("fault_id").agg(
        n=("success", "size"), class_=("fault_class", "first"), relevance=("expected_relevance", "first"),
        **{c: (c, "mean") for c in funnel_cols},
        affected_stages=("affected_stages", "mean"), wasted_stages=("wasted_stages", "mean"),
        success=("success", "mean"), unauthorized=("unauthorized_executed", "mean"),
        irreversible_deviations=("irreversible_deviations", "mean"))
    t3 = t3.loc[_ordered(t3.index, FAULT_ORDER)]
    S += ["## 2. Failure propagation without recovery (applied faults)\n",
          "Funnel columns are fractions of runs. `faulty_field_used`: a plan action read the corrupted fact "
          "(field-level provenance). `downstream_changed`: next stage's output differs from the clean "
          "counterfactual. `outcome_changed`: task success differs from the clean run.\n", md(t3), ""]

    t3b = nra.groupby("fault_stage")[funnel_cols + ["affected_stages", "wasted_stages", "success"]].mean()
    t3b = t3b.loc[_ordered(t3b.index, ["researcher", "planner", "executor"])]
    S += ["By injection stage:\n", md(t3b), ""]
    t3c = nra.groupby("fault_type")[funnel_cols + ["success", "unauthorized_executed"]].mean()
    S += ["By fault type:\n", md(t3c.loc[_ordered(t3c.index, TYPE_ORDER)]), ""]
    t3d = nra.groupby("expected_relevance")[["downstream_changed", "outcome_changed", "success"]].agg(["mean", "size"])
    t3d.columns = [f"{a}_{b}" for a, b in t3d.columns]
    S += ["Design-time relevance label vs. measured effect (an ignored fault is not robustness):\n", md(t3d), ""]

    # ------------------------------------------------------------------ success by condition
    t2 = fa.pivot_table(index="fault_id", columns="condition", values="success", aggfunc="mean")
    t2 = t2.loc[_ordered(t2.index, FAULT_ORDER), _ordered(t2.columns, COND_ORDER)]
    S += ["## 3. Task success under each fault and recovery condition\n",
          "For restart conditions, `success` is judged on the database after recovery; see section 4 for "
          "what this hides.\n", md(t2), ""]

    agg_spec = dict(
        runs=("success", "size"), success=("success", "mean"),
        unauthorized_rate=("unauthorized_executed", lambda s: (s > 0).mean()),
        est_latency_s=("est_latency_s", "mean"), llm_calls=("llm_calls", "mean"), tokens=("tokens", "mean"))
    t4 = fa.groupby(["fault_class", "condition"]).agg(**agg_spec)
    t4 = t4.reindex([(c, k) for c in ["accidental", "adversarial"] for k in COND_ORDER if (c, k) in t4.index])
    S += ["Accidental vs. adversarial faults (kept separate):\n", md(t4), ""]

    # ------------------------------------------------------------------ recovery
    rs = fa[fa.condition.str.startswith("restart")]
    if len(rs):
        t5 = rs.groupby("condition").agg(
            runs=("success", "size"), db_success=("success", "mean"), strict_recovered=("strict_recovered", "mean"),
            recomputed_stages=("recomputed_stages", "mean"), retained_valid=("retained_valid_stages", "mean"),
            discarded_valid=("faulted_discarded_valid_stages_on_restart", "mean"),
            est_recovery_latency_s=("est_recovery_latency_s", "mean"), recovery_llm_calls=("recovery_llm_calls", "mean"),
            recovery_tokens=("recovery_tokens", "mean"),
            irreversible_before=("irreversible_before_recovery", "mean"),
            irreversible_deviations=("faulted_irreversible_deviations", lambda s: (s > 0).mean()),
            lost_external_updates=("lost_external_updates", lambda s: (s > 0).mean()),
            irreversible_falsely_undone=("irreversible_falsely_undone", lambda s: (s > 0).mean()),
            unauthorized_before=("faulted_unauthorized_executed", lambda s: (s > 0).mean()))
        S += ["## 4. Full restart after an oracle-identified fault\n",
              "`db_success`: final database looks correct. `strict_recovered`: also no harmful irreversible "
              "effect from the faulted run, and (snapshot) no lost external updates / no irreversible effect "
              "silently erased from the DB. Rates in the last four columns are fractions of runs with >=1 event.\n",
              md(t5), ""]
        t5b = rs.pivot_table(index="fault_id", columns="condition", values="strict_recovered", aggfunc="mean")
        t5b = t5b.loc[_ordered(t5b.index, FAULT_ORDER)]
        S += ["Strict recovery by fault:\n", md(t5b), ""]

    # ------------------------------------------------------------------ verification
    vf = fa[fa.condition.str.startswith("verify")]
    if len(vf):
        harmful = set(nra.loc[nra.outcome_changed == 1, ["task_id", "fault_id", "seed"]].itertuples(index=False, name=None))
        vf = vf.assign(harmful=[key in harmful for key in vf[["task_id", "fault_id", "seed"]].itertuples(index=False, name=None)])
        t6 = vf.groupby("condition").agg(
            runs=("success", "size"), success=("success", "mean"), detected=("detected", "mean"),
            unauthorized_rate=("unauthorized_executed", lambda s: (s > 0).mean()),
            est_latency_s=("est_latency_s", "mean"),
            llm_calls=("llm_calls", "mean"), tokens=("tokens", "mean"), verify_checks=("verify_checks", "mean"))
        miss = vf[vf.harmful].groupby("condition").agg(
            harmful_faults=("detected", "size"), detected_harmful=("detected", "mean"),
            success_on_harmful=("success", "mean"))
        miss["missed_harmful_rate"] = 1 - miss.detected_harmful
        fp = clean[clean.condition.str.startswith("clean_verify")].groupby("condition")["detected"].mean()
        S += ["## 5. Always-verify baselines\n", md(t6), "",
              "Detection quality (harmful = changed the outcome in the paired no-recovery run). "
              "Miss rate is 1 minus detected_harmful. Clean detections are reported separately; "
              "false positives require independent artifact-level adjudication.\n", md(miss), "",
              md(fp.to_frame("clean_detection_rate")), ""]
        t6b = vf.pivot_table(index="fault_id", columns="condition", values="detected", aggfunc="mean")
        S += ["Detection rate by fault:\n", md(t6b.loc[_ordered(t6b.index, FAULT_ORDER)]), ""]

    # ------------------------------------------------------------------ overhead
    t7 = d.groupby("condition")[["bookkeeping_s", "artifacts", "dependency_edges", "trace_bytes"]].mean()
    t7["bookkeeping_ms"] = t7.pop("bookkeeping_s") * 1000
    S += ["## 6. Dependency-tracking overhead (per run)\n", md(t7.loc[_ordered(t7.index, ["clean"] + COND_ORDER)]), ""]

    # ------------------------------------------------------------------ clean-success-only robustness check
    if clean_ok and len(clean_ok) < cl.task_id.nunique():
        sub = fa[fa.task_id.isin(clean_ok)].pivot_table(index="fault_type", columns="condition", values="success")
        S += ["Success restricted to tasks whose clean run succeeded:\n",
              md(sub[_ordered(sub.columns, COND_ORDER)]), ""]

    (out / "summary.md").write_text("\n".join(S))
    figures(d, fa, nra, t1, fig_dir)
    print(f"wrote {out / 'summary.md'} and figures in {fig_dir}")


def figures(d, fa, nra, t1, fig_dir):
    plt.rcParams.update({"font.size": 9})
    # Fig 1: success by fault type and condition
    p = fa.pivot_table(index="fault_type", columns="condition", values="success")
    p = p.loc[_ordered(p.index, TYPE_ORDER), _ordered(p.columns, COND_ORDER)]
    clean_s = d[d.condition == "clean"].success.mean()
    ax = p.plot.bar(figsize=(7.5, 3.8), width=0.8, rot=0)
    ax.axhline(clean_s, color="k", ls="--", lw=1, label=f"clean ({clean_s:.2f})")
    ax.set_ylabel("task success (DB-level for restarts)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.set_title("Task success by fault type and condition")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_success_by_fault_type.png", dpi=160)
    plt.close()

    # Fig 2: propagation by injection stage
    cols = ["fault_consumed", "downstream_changed", "actions_changed", "state_changed", "outcome_changed"]
    p = nra.groupby("fault_stage")[cols].mean()
    p = p.loc[_ordered(p.index, ["researcher", "planner", "executor"])]
    ax = p.plot.bar(figsize=(7.5, 3.2), width=0.8, rot=0)
    ax.set_ylabel("fraction of applied faults")
    ax.set_title("Propagation by injection stage (no recovery)")
    ax.set_xlabel("")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_propagation_by_stage.png", dpi=160)
    plt.close()

    # Fig 3: funnel per fault
    cols = ["fault_consumed", "faulty_field_used", "downstream_changed", "actions_changed", "state_changed",
            "outcome_changed"]
    p = nra.groupby("fault_id")[cols].mean()
    p = p.loc[_ordered(p.index, FAULT_ORDER)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(p.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols])
    ax.set_yticks(range(len(p)), p.index)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Fault outcome funnel (no recovery)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_funnel.png", dpi=160)
    plt.close()

    # Fig 4: cost of clean vs always verify
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.6))
    for ax, col in zip(axes, ["success", "est_latency_s", "llm_calls"]):
        ax.bar(range(len(t1)), t1[col].values, color=["#4c72b0", "#55a868", "#c44e52"][: len(t1)])
        ax.set_xticks(range(len(t1)), [c.replace("clean_", "").replace("clean", "none") for c in t1.index])
        ax.set_title(col)
    fig.suptitle("Clean runs: cost of always verifying")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_verify_cost.png", dpi=160)
    plt.close()

    # Fig 5: recovery tradeoff (strict correctness vs LLM calls)
    rows = []
    for c in COND_ORDER:
        x = fa[fa.condition == c]
        if not len(x):
            continue
        succ = x.strict_recovered.mean() if c.startswith("restart") else x.success.mean()
        rows.append((c, succ, x.llm_calls.mean(), x.est_latency_s.mean()))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for c, s, calls, lat in rows:
        ax.scatter(calls, s, s=40 + 4 * lat)
        ax.annotate(c, (calls, s), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("mean LLM calls per faulted run (incl. recovery)")
    ax.set_ylabel("success (strict for restarts)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Correctness vs. cost under faults")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_tradeoff.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/run", sys.argv[2:])
