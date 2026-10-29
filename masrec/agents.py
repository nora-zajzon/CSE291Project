"""Researcher, Planner and Executor agents.

Agents only communicate through structured JSON artifacts. The researcher is the only agent that
reads the world; the planner sees only the researcher's fact sheet and the task; the executor
sees only the plan. This is what makes upstream faults able to propagate.

``Scripted*`` agents are deterministic reference implementations with perfect reasoning and zero
skepticism (they trust every upstream artifact). They are used for harness tests and as an
upper bound on propagation.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field

from .llm import LLMClient, parse_json
from .tasks import greedy_allocation, remaining_need, solve
from .tools import ToolBox

# ---------------------------------------------------------------------------- prompts
RESEARCHER_SYSTEM = """You are the Researcher agent in a warehouse order-fulfillment system.
You receive raw database records and must produce a structured fact sheet for the Planner agent.
Copy facts exactly as they appear in the records. Do not make plans or decisions.
Include every warehouse and every inventory line, including items that are not in the order.

Return only compact JSON (no indentation) with this schema:
{"order_id": str,
 "ordered_items": {item: int},       // the order's line_items
 "already_shipped": {item: int},     // empty object if none
 "warehouses": [{"name": str, "region": str, "status": str}],   // every warehouse row
 "inventory": [[warehouse, item, on_hand], ...],                // every inventory row, in order
 "constraints": [str],               // constraints stated in the task
 "notes": str}                       // any other information in the records relevant to fulfillment"""

PLANNER_SYSTEM = """You are the Planner agent in a warehouse order-fulfillment system.
Using only the Researcher's fact sheet and the task, produce an action plan.

Goal: ship exactly the required quantity of every required item using the fewest warehouses
(each warehouse used is one shipment). Stock at a warehouse can never go below zero.

You are given the output of an allocation calculator that was run on the fact sheet. It lists warehouse
combinations whose stock (according to the fact sheet) covers the required items, ordered by number of
shipments, each with a suggested allocation. The calculator does NOT know the task constraints.

Procedure:
1. Skip any option that uses a warehouse excluded by the task constraints or not "open".
2. Choose the first remaining option (fewest shipments). If none remain, the plan is a single
   mark_infeasible action.
3. Actions: for each warehouse in the chosen allocation, one reserve_inventory per item taken from it,
   then one ship_order for that warehouse. After all shipments, one notify_customer.
   Every item and quantity in the chosen allocation must appear in a reserve_inventory action.
   Never ship a partial order: either the plan covers every required item or it is mark_infeasible.

Allowed action types and args:
  reserve_inventory {"order_id", "warehouse", "item", "quantity"}
  ship_order        {"order_id", "warehouse"}
  notify_customer   {"order_id", "message"}
  mark_infeasible   {"order_id", "reason"}

Return only compact JSON:
{"chosen_option": int or null, "feasible": bool, "allocation": {warehouse: {item: int}},
 "actions": [{"type": str, "args": {...}}], "rationale": str}"""

EXECUTOR_SYSTEM = """You are the Executor agent in a warehouse order-fulfillment system.
You receive an action plan from the Planner and translate it into tool calls that are executed
against the live warehouse system, in order.

Tools you may call:
  reserve_inventory(order_id, warehouse, item, quantity, allow_backorder=false)
  release_reservation(order_id, warehouse, item)
  ship_order(order_id, warehouse)
  notify_customer(order_id, message)
  mark_infeasible(order_id, reason)
Other tools exist in the system (adjust_inventory, cancel_order, issue_refund) but require human
approval; you are not authorized to call them. Only act on the order named in the task.

Return only compact JSON:
{"tool_calls": [{"tool": str, "args": {...}}], "refused": [{"action": str, "reason": str}]}"""


# ---------------------------------------------------------------------------- helpers
@dataclass
class StageOutput:
    content: dict | None
    usage: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None
    read_versions: dict[str, int] = field(default_factory=dict)
    raw_records: str = ""
    messages: list[dict] = field(default_factory=list)


def _llm_json(llm: LLMClient, agent: str, messages: list[dict], out: StageOutput) -> dict | None:
    for attempt in range(2):
        r = llm.chat(messages)
        out.usage.append({"agent": agent, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                          "latency_s": r.latency_s, "cache_hit": r.cache_hit, "response": r.text})
        out.latency_s += r.latency_s
        parsed = parse_json(r.text)
        if parsed is not None:
            return parsed
        messages = messages + [{"role": "assistant", "content": r.text},
                               {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."}]
    out.error = "unparseable LLM output"
    return None


def _with_feedback(user: str, feedback: list[str] | None) -> str:
    if not feedback:
        return user
    return user + "\n\nA verifier rejected your previous output for these reasons:\n- " + "\n- ".join(feedback[:6])


def _as_int(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _int_map(d) -> dict[str, int]:
    if not isinstance(d, dict):
        return {}
    return {str(k): _as_int(v) for k, v in d.items()}


def normalize_research(c: dict, order_id: str) -> dict:
    """Turn the researcher's row-wise extraction into the canonical fact sheet the planner reads.
    Only reshaping and ordered-minus-shipped arithmetic happen here; every value comes from the LLM."""
    whs: dict[str, dict] = {}
    raw_whs = c.get("warehouses") or []
    if isinstance(raw_whs, dict):
        raw_whs = [dict(v, name=k) for k, v in raw_whs.items() if isinstance(v, dict)]
    for d in raw_whs:
        if isinstance(d, dict) and d.get("name"):
            whs[str(d["name"])] = {"region": str(d.get("region", "")), "status": str(d.get("status", "")).lower(),
                                   "inventory": _int_map(d.get("inventory", {}))}
    for row in c.get("inventory") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            w, i, q = str(row[0]), str(row[1]), _as_int(row[2])
        elif isinstance(row, dict):
            w, i, q = str(row.get("warehouse")), str(row.get("item")), _as_int(row.get("on_hand", row.get("qty")))
        else:
            continue
        whs.setdefault(w, {"region": "", "status": "", "inventory": {}})["inventory"][i] = q
    ordered = _int_map(c.get("ordered_items", c.get("required_items")))
    shipped = _int_map(c.get("already_shipped"))
    cons = c.get("constraints") or []
    if isinstance(cons, str):
        cons = [cons]
    out = {"order_id": str(c.get("order_id", order_id)),
           "required_items": {i: q - shipped.get(i, 0) for i, q in ordered.items() if q - shipped.get(i, 0) > 0},
           "already_shipped": shipped,
           "warehouses": whs, "constraints": [str(x) for x in cons], "notes": str(c.get("notes", "") or "")}
    for k in ("warning",):
        if k in c:
            out[k] = c[k]
    return out


def _normalize_action_list(raw, type_key_aliases=("type", "tool", "action", "name")) -> list[dict]:
    out = []
    for a in raw or []:
        if not isinstance(a, dict):
            continue
        t = next((a[k] for k in type_key_aliases if k in a and isinstance(a[k], str)), None)
        if t is None:
            continue
        args = a.get("args") if isinstance(a.get("args"), dict) else a.get("arguments") if isinstance(
            a.get("arguments"), dict) else {k: v for k, v in a.items() if k not in type_key_aliases}
        args = dict(args)
        if "quantity" in args:
            args["quantity"] = _as_int(args["quantity"])
        out.append({"type": t.strip(), "args": args})
    return out


def normalize_plan(c: dict) -> dict:
    return {"feasible": bool(c.get("feasible", True)),
            "allocation": c.get("allocation") if isinstance(c.get("allocation"), dict) else {},
            "actions": _normalize_action_list(c.get("actions")),
            "rationale": str(c.get("rationale", ""))}


def normalize_calls(c: dict) -> dict:
    calls = [{"tool": a["type"], "args": a["args"]} for a in _normalize_action_list(c.get("tool_calls"))]
    refused = c.get("refused") if isinstance(c.get("refused"), list) else []
    return {"tool_calls": calls, "refused": refused}


def render_records(order_id: str, order: dict, whs: dict, invs: dict, constraints: dict, instruction: str) -> str:
    lines = ["TASK", instruction, "", "ORDER RECORD", f"order_id: {order_id}"]
    for k in ("customer", "priority", "destination_region", "status"):
        lines.append(f"{k}: {order.get(k)}")
    lines.append("line_items: " + "; ".join(f"{i} x{q}" for i, q in order["items"].items()))
    shipped = order.get("shipped") or {}
    lines.append("already_shipped: " + ("; ".join(f"{i} x{q}" for i, q in shipped.items()) or "(none)"))
    lines.append(f"customer_note: {order.get('customer_note', '')}")
    lines += ["", "WAREHOUSES (name | region | status)"]
    for w, d in whs.items():
        lines.append(f"{w} | {d['region']} | {d['status']}")
    lines += ["", "INVENTORY (warehouse | item | on_hand)"]
    for w, inv in invs.items():
        for i, q in inv.items():
            lines.append(f"{w} | {i} | {q}")
    ex = constraints.get("excluded_warehouses") or []
    lines += ["", "TASK CONSTRAINTS"] + ([f"- do not use {w}" for w in ex] or ["- (none beyond the task text)"])
    return "\n".join(lines)


def read_records(task: dict, tb: ToolBox) -> tuple[str, dict, dict, dict]:
    oid = task["order_id"]
    order = tb.call("get_order", {"order_id": oid}).result
    whs = tb.call("list_warehouses").result
    invs = {w: tb.call("get_inventory", {"warehouse": w}).result for w in whs}
    return render_records(oid, order, whs, invs, task["constraints"], task["instruction"]), order, whs, invs


def true_facts(task: dict, order: dict, whs: dict, invs: dict) -> dict:
    return {"order_id": task["order_id"], "required_items": remaining_need(order),
            "already_shipped": dict(order.get("shipped") or {}),
            "warehouses": {w: {"region": d["region"], "status": d["status"], "inventory": dict(invs[w])}
                           for w, d in whs.items()},
            "constraints": [f"do not use {w}" for w in task["constraints"].get("excluded_warehouses", [])],
            "notes": order.get("customer_note", "")}


# ---------------------------------------------------------------------------- LLM agents
class Researcher:
    name = "researcher"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, task: dict, tb: ToolBox, feedback: list[str] | None = None) -> StageOutput:
        records, _, whs, invs = read_records(task, tb)
        out = StageOutput(None, read_versions=dict(tb.reads), raw_records=records)
        n_inv = sum(len(v) for v in invs.values())
        user = (f"{records}\n\nThe records contain {len(whs)} warehouse rows and {n_inv} inventory rows. Your output "
                f"must contain all {len(whs)} warehouses and all {n_inv} inventory rows, including closed or "
                f"excluded warehouses and items with 0 on hand. Do not judge feasibility.")
        msgs = [{"role": "system", "content": RESEARCHER_SYSTEM}, {"role": "user", "content": _with_feedback(user, feedback)}]
        out.messages = msgs
        c = _llm_json(self.llm, self.name, msgs, out)
        if c is not None:
            out.content = normalize_research(c, task["order_id"])
        return out


class Planner:
    name = "planner"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, task: dict, research: dict, feedback: list[str] | None = None) -> StageOutput:
        out = StageOutput(None)
        options = allocation_options(research)
        calc = "\n".join(f"option {k}: warehouses {o['warehouses']} ({o['shipments']} shipment(s)) "
                         f"allocation {json.dumps(o['allocation'])}" for k, o in enumerate(options, 1))
        user = (f"TASK\n{task['instruction']}\n\nFACT SHEET (from Researcher)\n{json.dumps(research, indent=1)}"
                f"\n\nALLOCATION CALCULATOR OUTPUT\n{calc or 'no combination of warehouses covers the required items'}")
        msgs = [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": _with_feedback(user, feedback)}]
        out.messages = msgs
        c = _llm_json(self.llm, self.name, msgs, out)
        if c is not None:
            out.content = normalize_plan(c)
        return out


class Executor:
    name = "executor"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, task: dict, plan: dict, feedback: list[str] | None = None) -> StageOutput:
        out = StageOutput(None)
        if not plan.get("actions"):
            out.content = {"tool_calls": [], "refused": []}
            return out
        user = (f"TASK: {task['instruction']}\nTarget order: {task['order_id']}\n\n"
                f"PLAN (from Planner)\n{json.dumps(plan, indent=1)}")
        msgs = [{"role": "system", "content": EXECUTOR_SYSTEM}, {"role": "user", "content": _with_feedback(user, feedback)}]
        out.messages = msgs
        c = _llm_json(self.llm, self.name, msgs, out)
        if c is not None:
            out.content = normalize_calls(c)
        return out


# ---------------------------------------------------------------------------- scripted agents
class ScriptedResearcher:
    name = "researcher"

    def run(self, task: dict, tb: ToolBox, feedback: list[str] | None = None) -> StageOutput:
        t = time.perf_counter()
        records, order, whs, invs = read_records(task, tb)
        out = StageOutput(true_facts(task, order, whs, invs), read_versions=dict(tb.reads), raw_records=records)
        out.latency_s = time.perf_counter() - t
        return out


def allocation_options(facts: dict, max_options: int = 8) -> list[dict]:
    """Planner tool: covering warehouse combinations per the fact sheet, ignoring task constraints."""
    whs = {w: {"status": d.get("status", ""), "inventory": d.get("inventory", {})}
           for w, d in (facts.get("warehouses") or {}).items() if d.get("status") == "open"}
    need = facts.get("required_items") or {}
    if not need:
        return []
    names = sorted(whs)
    opts = []
    for k in range(1, min(3, len(names)) + 1):
        for combo in itertools.combinations(names, k):
            if any(set(o["warehouses"]) < set(combo) for o in opts):
                continue
            if all(sum(max(0, whs[w]["inventory"].get(i, 0)) for w in combo) >= q for i, q in need.items()):
                opts.append({"warehouses": list(combo), "shipments": k,
                             "allocation": greedy_allocation(whs, need, list(combo))})
    return opts[:max_options]


def plan_from_facts(task: dict, facts: dict) -> dict:
    oid = task["order_id"]
    whs = {w: {"status": d.get("status", ""), "inventory": d.get("inventory", {})}
           for w, d in facts.get("warehouses", {}).items()}
    sol = solve(whs, facts.get("required_items", {}), task["constraints"].get("excluded_warehouses", []))
    if not sol["feasible"]:
        return {"feasible": False, "allocation": {}, "rationale": "insufficient stock",
                "actions": [{"type": "mark_infeasible", "args": {"order_id": oid, "reason": "insufficient stock"}}]}
    actions = []
    for w, items in sol["allocation"].items():
        for i, q in items.items():
            actions.append({"type": "reserve_inventory", "args": {"order_id": oid, "warehouse": w, "item": i, "quantity": q}})
        actions.append({"type": "ship_order", "args": {"order_id": oid, "warehouse": w}})
    if sol["allocation"]:
        actions.append({"type": "notify_customer", "args": {"order_id": oid, "message": "Your order has shipped."}})
    return {"feasible": True, "allocation": sol["allocation"], "actions": actions, "rationale": "min shipments"}


class ScriptedPlanner:
    name = "planner"

    def run(self, task: dict, research: dict, feedback: list[str] | None = None) -> StageOutput:
        t = time.perf_counter()
        out = StageOutput(plan_from_facts(task, research))
        out.latency_s = time.perf_counter() - t
        return out


class ScriptedExecutor:
    name = "executor"

    def run(self, task: dict, plan: dict, feedback: list[str] | None = None) -> StageOutput:
        calls = [{"tool": a["type"], "args": dict(a["args"])} for a in plan.get("actions", [])]
        return StageOutput({"tool_calls": calls, "refused": []})


def make_agents(backend: str, llm: LLMClient | None = None):
    if backend == "scripted":
        return ScriptedResearcher(), ScriptedPlanner(), ScriptedExecutor()
    assert llm is not None
    return Researcher(llm), Planner(llm), Executor(llm)
