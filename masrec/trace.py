"""Per-run event trace written as JSON Lines."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: list[dict] = []
        self.t0 = time.time()
        self.bookkeeping_s = 0.0

    def log(self, etype: str, **payload: Any) -> None:
        t = time.perf_counter()
        self.events.append({"run_id": self.run_id, "seq": len(self.events), "t": round(time.time() - self.t0, 4),
                            "type": etype, **payload})
        self.bookkeeping_s += time.perf_counter() - t

    def write(self, directory: str | Path) -> int:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        data = "\n".join(json.dumps(e, default=str) for e in self.events) + "\n"
        (p / f"{self.run_id}.jsonl").write_text(data)
        return len(data.encode())
