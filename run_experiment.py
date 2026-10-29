"""Run the Phase 1 experiments.

Examples
  python run_experiment.py --backend scripted --out results/scripted
  python run_experiment.py --backend ollama --model qwen2.5:7b-instruct --out results/qwen7b --workers 4
  python run_experiment.py --backend ollama --tasks-limit 3 --conditions clean --out results/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from masrec.agents import make_agents
from masrec.experiment import ALL_CONDITIONS, Ctx, dump_schedule, run_task, write_jsonl
from masrec.llm import OllamaClient, OpenAICompatClient
from masrec.tasks import generate_tasks, load_tasks
from masrec.verifier import LLMVerifier, RuleVerifier


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["scripted", "ollama", "openai"], default="scripted")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                    help="for --backend openai (defaults to $OPENAI_BASE_URL)")
    ap.add_argument("--tasks", default="data/tasks.json")
    ap.add_argument("--tasks-limit", type=int, default=0)
    ap.add_argument("--task-ids", default="", help="comma-separated subset")
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    ap.add_argument("--faults", default="", help="comma-separated substrings of fault ids to keep")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout-budget", type=float, default=30.0, help="simulated seconds charged per timeout")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="results/run")
    a = ap.parse_args(argv)

    if not Path(a.tasks).exists():
        Path(a.tasks).parent.mkdir(parents=True, exist_ok=True)
        Path(a.tasks).write_text(json.dumps(generate_tasks(), indent=2))
    tasks = load_tasks(a.tasks)
    if a.task_ids:
        keep = set(a.task_ids.split(","))
        tasks = [t for t in tasks if t["task_id"] in keep]
    if a.tasks_limit:
        tasks = tasks[: a.tasks_limit]
    conditions = [c for c in a.conditions.split(",") if c]
    faults = [f for f in a.faults.split(",") if f] or None
    out = Path(a.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "fault_schedule.jsonl", dump_schedule(tasks))

    all_rows = []
    t0 = time.time()
    for seed in [int(s) for s in a.seeds.split(",")]:
        llm = None
        if a.backend != "scripted":
            cache = None if a.no_cache else f"cache/llm_{a.model.replace(':', '_').replace('/', '_')}.jsonl"
            kw = dict(temperature=a.temperature, seed=seed, cache_path=cache)
            llm = (OllamaClient(a.model, **kw) if a.backend == "ollama"
                   else OpenAICompatClient(a.model, a.base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"),
                                           **kw))
        agents = make_agents(a.backend, llm)
        verifiers = {"rule": RuleVerifier()}
        if llm is not None:
            verifiers["llm"] = LLMVerifier(llm)
        ctx = Ctx(agents=agents, verifiers=verifiers, trace_dir=str(out / "traces"),
                  timeout_budget_s=a.timeout_budget, seed=seed,
                  backend=a.backend if a.backend == "scripted" else a.model)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(run_task, t, ctx, conditions, faults): t["task_id"] for t in tasks}
            for k, f in enumerate(as_completed(futs), 1):
                tid = futs[f]
                try:
                    rows = f.result()
                except Exception as e:  # keep going; record the failure
                    import traceback
                    traceback.print_exc()
                    rows = [{"task_id": tid, "condition": "harness_error", "error": repr(e), "seed": seed}]
                all_rows += rows
                print(f"[seed {seed}] {k}/{len(tasks)} {tid}: {len(rows)} runs  ({time.time() - t0:.0f}s)",
                      flush=True)
                pd.DataFrame(all_rows).to_csv(out / "runs.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "runs.csv", index=False)
    meta = {"backend": a.backend, "model": a.model if a.backend != "scripted" else None, "seeds": a.seeds,
            "temperature": a.temperature, "n_tasks": len(tasks), "conditions": conditions,
            "n_runs": len(df), "elapsed_s": round(time.time() - t0, 1), "timeout_budget_s": a.timeout_budget}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
