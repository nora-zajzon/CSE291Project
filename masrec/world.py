"""Deterministic simulated order-fulfillment environment.

The world is plain data. Every read and write is keyed (e.g. ``inventory.warehouse_A.laptop``)
and every write bumps a per-key version, so artifacts can record exactly which pieces of shared
state they were derived from and later be checked for staleness.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


def inv_key(warehouse: str, item: str) -> str:
    return f"inventory.{warehouse}.{item}"


def wh_key(warehouse: str) -> str:
    return f"warehouse.{warehouse}"


def order_key(order_id: str) -> str:
    return f"order.{order_id}"


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:16]


class World:
    """Mutable shared state plus an append-only log of every mutation."""

    def __init__(self, state: dict):
        self.state = copy.deepcopy(state)
        self.state.setdefault("reservations", [])
        self.state.setdefault("shipments", [])
        self.state.setdefault("notifications", [])
        self.state.setdefault("refunds", [])
        self.version = 0
        self.key_versions: dict[str, int] = {}
        self.action_log: list[dict] = []
        self._ids = 0

    # ------------------------------------------------------------------ accessors
    @property
    def warehouses(self) -> dict:
        return self.state["warehouses"]

    @property
    def orders(self) -> dict:
        return self.state["orders"]

    def next_id(self, prefix: str) -> str:
        self._ids += 1
        return f"{prefix}_{self._ids:04d}"

    def key_version(self, key: str) -> int:
        return self.key_versions.get(key, 0)

    def touch(self, keys: list[str]) -> None:
        self.version += 1
        for k in keys:
            self.key_versions[k] = self.version

    # ------------------------------------------------------------------ snapshots
    def snapshot(self) -> dict:
        return {
            "state": copy.deepcopy(self.state),
            "version": self.version,
            "key_versions": dict(self.key_versions),
            "log_len": len(self.action_log),
            "ids": self._ids,
        }

    def restore(self, snap: dict) -> None:
        """Roll the simulated database back. The action log is kept (it models reality)."""
        self.state = copy.deepcopy(snap["state"])
        self.key_versions = dict(snap["key_versions"])
        self.version += 1
        for k in list(self.key_versions):
            self.key_versions[k] = self.version
        self.action_log.append({
            "seq": len(self.action_log), "actor": "recovery", "tool": "__snapshot_restore__",
            "args": {"to_log_len": snap["log_len"]}, "ok": True, "write_keys": ["*"],
            "reversible": True, "irreversible_effect": False, "version": self.version,
        })

    def fingerprint(self, exclude_messages: bool = True) -> str:
        s = copy.deepcopy(self.state)
        if exclude_messages:
            s["notifications"] = [{"order_id": n["order_id"]} for n in s["notifications"]]
        for r in s["reservations"]:
            r.pop("res_id", None)
        for sh in s["shipments"]:
            sh.pop("shipment_id", None)
        return digest(s)

    def copy(self) -> "World":
        w = World.__new__(World)
        w.state = copy.deepcopy(self.state)
        w.version = self.version
        w.key_versions = dict(self.key_versions)
        w.action_log = copy.deepcopy(self.action_log)
        w._ids = self._ids
        return w
