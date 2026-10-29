"""Measure single-stream prompt-processing and generation throughput of the local model.

Measured stage latencies are inflated when several runs share one GPU, so the analysis also
reports a token-based estimate: prompt_tokens / pp_rate + completion_tokens / tg_rate.

  python calibrate_latency.py qwen2.5:7b-instruct
"""
import json
import sys

import requests

from masrec.agents import RESEARCHER_SYSTEM, read_records
from masrec.tasks import load_tasks
from masrec.tools import ToolBox
from masrec.world import World

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b-instruct"
pp, tg = [], []
for t in load_tasks("data/tasks.json")[:6]:
    records, *_ = read_records(t, ToolBox(World(t["initial_state"]), "x"))
    r = requests.post("http://localhost:11434/api/chat", timeout=300, json={
        "model": model, "stream": False, "format": "json",
        "messages": [{"role": "system", "content": RESEARCHER_SYSTEM}, {"role": "user", "content": records}],
        "options": {"temperature": 0.7, "seed": 123}}).json()
    pp.append(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9))
    tg.append(r["eval_count"] / (r["eval_duration"] / 1e9))
out = {"model": model, "pp_tok_per_s": round(sorted(pp)[len(pp) // 2], 1), "tg_tok_per_s": round(sorted(tg)[len(tg) // 2], 1)}
print(json.dumps(out))
with open("data/latency_calibration.json", "w") as f:
    json.dump(out, f, indent=2)
