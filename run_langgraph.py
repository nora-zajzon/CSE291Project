"""Same tasks and fault schedule on the LangGraph implementation of the workflow.

Conditions
  clean                        LangGraph run, no fault
  no_recovery                  fault injected
  restart_checkpoint           rewind LangGraph to the checkpoint before the faulty node and resume
                               (agent state restored, world state left as is)
  restart_checkpoint_compensate  same, after undoing our own reversible actions in the world

  python run_langgraph.py --backend ollama --out results/langgraph_qwen7b
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from masrec.agents import make_agents
from masrec.evaluator import evaluate_outcome, irreversible_deviations
from masrec.experiment import clean_reference
from masrec.faults import Injector, build_schedule
from masrec.langgraph_workflow import LangGraphWorkflow, stages_before
from masrec.llm import OllamaClient, OpenAICompatClient
from masrec.tasks import load_tasks
from masrec.tools import ToolBox
from masrec.trace import Trace
from masrec.world import World

CONDITIONS = ["clean", "no_recovery", "restart_checkpoint", "restart_checkpoint_compensate"]


def usage(res):
    return {"llm_calls": res.llm_calls, "tokens": res.tokens,
            "prompt_tokens": sum(s.prompt_tokens for s in res.stages.values()),
            "completion_tokens": sum(s.completion_tokens for s in res.stages.values())}


def compensate_main(world) -> tuple[int, int]:
    tb = ToolBox(world, "recovery")
    done = kept = 0
    for e in reversed([e for e in world.action_log if e["actor"] == "executor" and e.get("phase") == "main" and e["ok"]]):
        if e["reversible"] and e.get("compensation"):
            done += int(tb.compensate(e["compensation"]).ok)
        elif not e["reversible"]:
            kept += 1
    return done, kept


def run_task(task, agents, faults):
    rows = []
    tid = task["task_id"]
    base = {"task_id": tid, "category": task["category"], "framework": "langgraph"}

    lg = LangGraphWorkflow(agents, Trace(f"{tid}__lg_clean"))
    world = World(task["initial_state"])
    inj = Injector(None)
    t = time.perf_counter()
    res = lg.run(task, world, inj, f"{tid}-clean")
    wall = time.perf_counter() - t
    out = evaluate_outcome(task, world, {"main"})
    clean = clean_reference(world, res, out)
    rows.append({**base, "fault_id": "none", "condition": "clean", "success": int(out["success"]),
                 "checkpoints": lg.n_checkpoints(f"{tid}-clean"), "wall_s": round(wall, 3), **usage(res)})

    for spec in build_schedule(task):
        if faults and not any(f in spec.fault_id for f in faults):
            continue
        fb = {**base, "fault_id": spec.fault_id, "fault_type": spec.fault_type, "fault_class": spec.fault_class,
              "fault_stage": spec.stage, "expected_relevance": spec.expected_relevance}
        lg = LangGraphWorkflow(agents, Trace(f"{tid}__{spec.fault_id}"))
        world = World(task["initial_state"])
        inj = Injector(spec)
        inj.prepare_world(world, task)
        thread = f"{tid}-{spec.fault_id}"
        res = lg.run(task, world, inj, thread)
        out = evaluate_outcome(task, world, {"main"})
        devs = len(irreversible_deviations(world, clean["irrev"]))
        n_irrev = sum(1 for e in world.action_log
                      if e["actor"] == "executor" and e.get("phase") == "main" and e["irreversible_effect"])
        nr = {**fb, "condition": "no_recovery", "fault_applied": int(inj.applied), "success": int(out["success"]),
              "irreversible_deviations": devs, "irreversible_actions": n_irrev, **usage(res)}
        rows.append(nr)
        cfg = lg.checkpoint_before(thread, spec.stage)
        for mode in ("plain", "compensate"):
            cond = "restart_checkpoint" + ("_compensate" if mode == "compensate" else "")
            w2 = world.copy()
            done = kept = 0
            if mode == "compensate":
                done, kept = compensate_main(w2)
            t = time.perf_counter()
            rec = lg.resume(task, w2, Injector(spec, enabled=False), cfg)
            rwall = time.perf_counter() - t
            rout = evaluate_outcome(task, w2, {"recovery"})
            strict = rout["success"] and devs == 0
            rows.append({**fb, "condition": cond, "fault_applied": int(inj.applied), "success": int(rout["success"]),
                         "strict_recovered": int(bool(strict)), "faulted_success": nr["success"],
                         "irreversible_before_recovery": n_irrev, "irreversible_deviations": devs,
                         "recomputed_stages": len(lg.nodes_run), "retained_stages": stages_before(spec.stage),
                         "compensations": done, "uncompensable_actions": kept,
                         "over_shipped": int(rout.get("over_shipped", 0)),
                         "dangling_reservations": int(rout.get("dangling_reservations", 0)),
                         "recovery_wall_s": round(rwall, 3),
                         **{f"recovery_{k}": v for k, v in usage(rec).items()},
                         **{k: nr[k] + v for k, v in usage(rec).items()}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["scripted", "ollama", "openai"], default="scripted")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--tasks", default="data/tasks.json")
    ap.add_argument("--faults", default="", help="comma-separated substrings of fault ids to keep")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/langgraph_scripted")
    a = ap.parse_args()
    llm = None
    if a.backend != "scripted":
        cache = f"cache/llm_{a.model.replace(':', '_').replace('/', '_')}.jsonl"
        kw = dict(temperature=0.0, seed=0, cache_path=cache)
        llm = (OllamaClient(a.model, **kw) if a.backend == "ollama"
               else OpenAICompatClient(a.model, a.base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"), **kw))
    agents = make_agents(a.backend, llm)
    tasks = load_tasks(a.tasks)
    faults = [f for f in a.faults.split(",") if f] or None
    os.makedirs(a.out, exist_ok=True)
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_task, t, agents, faults) for t in tasks]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                rows += f.result()
            except Exception as e:
                print("task failed:", repr(e), file=sys.stderr, flush=True)
            keys = []
            for r in rows:
                keys += [k for k in r if k not in keys]
            with open(os.path.join(a.out, "runs.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, keys)
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: (r["task_id"], r["fault_id"], r["condition"])))
            print(f"[{n}/{len(futs)}] {time.time() - t0:.0f}s rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
