"""Intermediate results tagged with their sources.

Two levels of dependencies are recorded:
* artifact-level ``depends_on`` edges (research -> plan -> execution), and
* field-level provenance: which world keys (with versions) an artifact read, and for each plan
  action / tool call, which upstream fields it relied on.

Phase 1 only *records* these; Phase 2 traverses them for selective invalidation.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .world import digest


@dataclass
class Artifact:
    artifact_id: str
    kind: str                  # env_read | research | plan | tool_calls | execution | verification
    producer: str              # researcher | planner | executor | verifier | harness
    content: Any
    depends_on: list[str] = field(default_factory=list)
    read_versions: dict[str, int] = field(default_factory=dict)
    field_deps: dict[str, list[str]] = field(default_factory=dict)
    attempt: int = 0
    status: str = "ok"         # ok | injected | partial | timeout | rejected
    created_at: float = field(default_factory=time.time)
    world_version: int = 0
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = digest(self.content)


class ArtifactStore:
    def __init__(self):
        self.artifacts: dict[str, Artifact] = {}
        self.order: list[str] = []
        self.bookkeeping_s = 0.0
        self._n = 0

    def new_id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}_{self._n:03d}"

    def add(self, art: Artifact) -> Artifact:
        t = time.perf_counter()
        self.artifacts[art.artifact_id] = art
        self.order.append(art.artifact_id)
        self.bookkeeping_s += time.perf_counter() - t
        return art

    def get(self, aid: str) -> Artifact:
        return self.artifacts[aid]

    def descendants(self, aid: str) -> list[str]:
        """All artifacts that transitively depend on ``aid`` (excluding itself)."""
        t = time.perf_counter()
        children: dict[str, list[str]] = {}
        for a in self.artifacts.values():
            for p in a.depends_on:
                children.setdefault(p, []).append(a.artifact_id)
        seen, out, stack = set(), [], [aid]
        while stack:
            for c in children.get(stack.pop(), []):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
                    stack.append(c)
        self.bookkeeping_s += time.perf_counter() - t
        return out

    def latest(self, kind: str) -> Artifact | None:
        for aid in reversed(self.order):
            if self.artifacts[aid].kind == kind:
                return self.artifacts[aid]
        return None

    def n_edges(self) -> int:
        return sum(len(a.depends_on) for a in self.artifacts.values()) + sum(
            len(v) for a in self.artifacts.values() for v in a.field_deps.values())

    def to_json(self) -> list[dict]:
        return [asdict(self.artifacts[a]) for a in self.order]


def plan_field_deps(actions: list[dict]) -> dict[str, list[str]]:
    """Which research facts each plan action relies on (derived from the action itself)."""
    deps = {}
    for idx, a in enumerate(actions):
        t, args = a.get("type"), a.get("args", {})
        w, i = args.get("warehouse"), args.get("item")
        if t == "reserve_inventory":
            d = [f"warehouses.{w}.inventory.{i}", f"warehouses.{w}.status", f"required_items.{i}"]
        elif t == "ship_order":
            d = [f"warehouses.{w}.status"]
        elif t == "mark_infeasible":
            d = ["warehouses.*", "required_items.*"]
        else:
            d = []
        deps[f"action[{idx}]"] = d
    return deps


def fact_matches(dep: str, fault_key: str) -> bool:
    """Does a field dependency (possibly wildcard) cover a corrupted fact key?"""
    if dep == fault_key:
        return True
    if dep.endswith(".*"):
        return fault_key.startswith(dep[:-1])
    return False
