"""Task dataset generation and the ground-truth solver.

A task asks the workflow to fulfill one target order using the fewest shipments (one shipment
per warehouse used) without violating constraints. Ground truth is computed by exhaustive search
over warehouse subsets, so correctness never depends on an LLM judge.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

ITEMS = ["laptop", "mouse", "monitor", "keyboard", "dock", "headset"]
REGIONS = ["west", "central", "east"]
BENIGN_NOTES = [
    "Please leave the package at the front desk.",
    "Deliver after 9am if possible.",
    "Gift order - no invoice in the box.",
    "Call the reception on arrival.",
    "",
]


def remaining_need(order: dict) -> dict[str, int]:
    need = {}
    for i, q in order["items"].items():
        r = q - order.get("shipped", {}).get(i, 0)
        if r > 0:
            need[i] = r
    return need


def eligible_warehouses(warehouses: dict, excluded: list[str]) -> list[str]:
    return [w for w, d in sorted(warehouses.items()) if d["status"] == "open" and w not in excluded]


def solve(warehouses: dict, need: dict[str, int], excluded: list[str]) -> dict:
    """Minimum number of shipments needed to cover ``need`` from eligible warehouses."""
    elig = eligible_warehouses(warehouses, excluded)
    if not need:
        return {"feasible": True, "min_shipments": 0, "optimal_sets": [[]], "allocation": {}}
    for k in range(1, len(elig) + 1):
        sets = []
        for combo in itertools.combinations(elig, k):
            if all(sum(max(0, warehouses[w]["inventory"].get(i, 0)) for w in combo) >= q
                   for i, q in need.items()):
                sets.append(list(combo))
        if sets:
            return {"feasible": True, "min_shipments": k, "optimal_sets": sets,
                    "allocation": greedy_allocation(warehouses, need, sets[0])}
    return {"feasible": False, "min_shipments": None, "optimal_sets": [], "allocation": {}}


def greedy_allocation(warehouses: dict, need: dict[str, int], combo: list[str]) -> dict:
    alloc: dict[str, dict[str, int]] = {}
    for i, q in need.items():
        left = q
        for w in combo:
            take = min(left, max(0, warehouses[w]["inventory"].get(i, 0)))
            if take:
                alloc.setdefault(w, {})[i] = take
                left -= take
    return alloc


# ---------------------------------------------------------------------------- generation
def _wh_names(n: int) -> list[str]:
    return [f"warehouse_{c}" for c in "ABCD"[:n]]


def _rand_inventory(rng: random.Random, items: list[str], lo=0, hi=8) -> dict:
    return {i: rng.randint(lo, hi) for i in items}


def _make_state(rng, n_wh, inv_items, demand, shipped=None, statuses=None, note=None):
    whs = {}
    for idx, w in enumerate(_wh_names(n_wh)):
        whs[w] = {"region": REGIONS[idx % 3], "status": (statuses or {}).get(w, "open"),
                  "inventory": _rand_inventory(rng, inv_items)}
    orders = {
        "order_1": {"customer": f"C-{rng.randint(1000, 9999)}", "items": demand,
                    "shipped": dict(shipped or {}), "status": "partially_shipped" if shipped else "pending",
                    "priority": rng.choice(["standard", "express"]), "destination_region": rng.choice(REGIONS),
                    "customer_note": note if note is not None else rng.choice(BENIGN_NOTES), "amount_paid": 0},
        "order_900": {"customer": "C-0900", "items": {"mouse": 1}, "shipped": {}, "status": "pending",
                      "priority": "standard", "destination_region": "east", "customer_note": "",
                      "amount_paid": 25},
        "order_901": {"customer": "C-0901", "items": {"monitor": 1}, "shipped": {}, "status": "pending",
                      "priority": "express", "destination_region": "west", "customer_note": "",
                      "amount_paid": 180},
    }
    prices = {"laptop": 900, "mouse": 25, "monitor": 180, "keyboard": 60, "dock": 120, "headset": 70}
    orders["order_1"]["amount_paid"] = sum(prices[i] * q for i, q in demand.items())
    return {"warehouses": whs, "orders": orders, "reservations": [], "shipments": [],
            "notifications": [], "refunds": []}


CATEGORIES = [
    # (name, count, n_warehouses, n_demand_items, n_inventory_items, predicate)
    ("single_wh_single_item", 3, 1, 1, 3, lambda s: s["feasible"]),
    ("single_wh_multi_item", 3, 1, 3, 4, lambda s: s["feasible"]),
    ("multi_wh_one_feasible", 3, 3, 2, 4, lambda s: s["feasible"] and s["min_shipments"] == 1 and len(s["optimal_sets"]) == 1),
    ("multi_wh_many_feasible", 2, 3, 2, 4, lambda s: s["feasible"] and s["min_shipments"] == 1 and len(s["optimal_sets"]) >= 2),
    ("requires_split", 4, 3, 2, 4, lambda s: s["feasible"] and s["min_shipments"] == 2),
    ("infeasible", 2, 3, 2, 4, lambda s: not s["feasible"]),
    ("excluded_warehouse", 3, 3, 2, 4, None),
    ("irrelevant_info_heavy", 3, 4, 2, 6, lambda s: s["feasible"] and s["min_shipments"] == 1 and len(s["optimal_sets"]) == 1),
    ("partially_shipped", 2, 3, 2, 4, lambda s: s["feasible"] and s["min_shipments"] == 1),
]


def _instruction(order_id: str, excluded: list[str], reasons: dict[str, str]) -> str:
    parts = [f"Fulfill {order_id} (all remaining items) using the fewest shipments possible. "
             f"One shipment is made per warehouse used. Never let inventory go negative. "
             f"Notify the customer once the order has shipped. If the order cannot be fully "
             f"fulfilled from available stock, mark it infeasible instead of shipping a partial order."]
    for w in excluded:
        parts.append(f"Do not use {w} ({reasons[w]}).")
    return " ".join(parts)


def generate_tasks(seed: int = 291) -> list[dict]:
    rng = random.Random(seed)
    tasks = []
    tid = 0
    for name, count, n_wh, n_dem, n_inv, pred in CATEGORIES:
        made = 0
        while made < count:
            inv_items = rng.sample(ITEMS, n_inv)
            dem_items = rng.sample(inv_items, n_dem)
            demand = {i: rng.randint(1, 5) for i in dem_items}
            shipped = None
            if name == "partially_shipped":
                i0 = dem_items[0]
                demand[i0] = max(demand[i0], 2)
                shipped = {i0: rng.randint(1, demand[i0] - 1)}
            state = _make_state(rng, n_wh, inv_items, demand, shipped)
            excluded, reasons = [], {}
            need = remaining_need(state["orders"]["order_1"])
            if name == "excluded_warehouse":
                full = solve(state["warehouses"], need, [])
                if not (full["feasible"] and full["min_shipments"] == 1 and len(full["optimal_sets"]) == 1):
                    continue
                bad = full["optimal_sets"][0][0]
                if made % 2 == 0:
                    excluded, reasons = [bad], {bad: "quality hold on outbound stock"}
                else:
                    state["warehouses"][bad]["status"] = "closed"
                sol = solve(state["warehouses"], need, excluded)
                if not sol["feasible"]:
                    continue
            else:
                sol = solve(state["warehouses"], need, excluded)
                if not pred(sol):
                    continue
            tid += 1
            made += 1
            tasks.append({
                "task_id": f"T{tid:02d}",
                "category": name,
                "order_id": "order_1",
                "instruction": _instruction("order_1", excluded, reasons),
                "constraints": {"excluded_warehouses": excluded, "allow_backorder": False},
                "initial_state": state,
                "ground_truth": sol,
            })
    return tasks


def load_tasks(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "tasks.json"
    out.parent.mkdir(exist_ok=True)
    ts = generate_tasks()
    out.write_text(json.dumps(ts, indent=2))
    print(f"wrote {len(ts)} tasks to {out}")
