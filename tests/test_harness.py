import copy

import pytest

from masrec.agents import make_agents
from masrec.artifacts import ArtifactStore
from masrec.evaluator import evaluate_outcome
from masrec.experiment import Ctx, run_task
from masrec.faults import Injector, build_schedule
from masrec.recovery import full_restart
from masrec.tasks import generate_tasks, solve
from masrec.tools import ToolBox, is_unauthorized
from masrec.trace import Trace
from masrec.verifier import RuleVerifier
from masrec.workflow import Workflow
from masrec.world import World, inv_key

TASKS = generate_tasks()


def _task(category):
    return next(t for t in TASKS if t["category"] == category)


def _run(task, spec=None, verifier=None):
    w = World(task["initial_state"])
    inj = Injector(spec)
    inj.prepare_world(w, task)
    snap = w.snapshot()
    store, tr = ArtifactStore(), Trace("t")
    res = Workflow(make_agents("scripted"), store, tr, verifier=verifier).run(task, w, inj)
    return w, inj, snap, store, tr, res


def test_solver_min_shipments():
    whs = {"A": {"status": "open", "inventory": {"x": 2, "y": 0}},
           "B": {"status": "open", "inventory": {"x": 0, "y": 3}},
           "C": {"status": "open", "inventory": {"x": 2, "y": 3}}}
    assert solve(whs, {"x": 2, "y": 3}, [])["optimal_sets"] == [["C"]]
    assert solve(whs, {"x": 2, "y": 3}, ["C"])["min_shipments"] == 2
    assert not solve(whs, {"x": 5}, [])["feasible"]


def test_tools_guard_inventory_and_log_reversibility():
    t = _task("single_wh_single_item")
    w = World(t["initial_state"])
    tb = ToolBox(w, "executor")
    item, have = next(iter(w.warehouses["warehouse_A"]["inventory"].items()))
    assert not tb.call("reserve_inventory", {"order_id": "order_1", "warehouse": "warehouse_A", "item": item,
                                             "quantity": have + 1}).ok
    r = tb.call("reserve_inventory", {"order_id": "order_1", "warehouse": "warehouse_A", "item": item, "quantity": 1})
    assert r.ok and w.action_log[-1]["reversible"] and w.action_log[-1]["compensation"]
    tb.compensate(r.compensation)
    assert w.warehouses["warehouse_A"]["inventory"][item] == have
    tb.call("issue_refund", {"order_id": "order_1", "amount": 10})
    assert w.action_log[-1]["irreversible_effect"]
    assert is_unauthorized(w.action_log[-1], "order_1")


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["task_id"])
def test_clean_scripted_success(task):
    w, *_, res = _run(task)
    assert evaluate_outcome(task, w, {"main"})["success"] == 1


def test_inflate_fault_propagates_and_verifier_stops_it():
    t = _task("requires_split")
    spec = next(s for s in build_schedule(t) if s.fault_id.startswith("incorrect_inflate"))
    w, inj, *_ = _run(t, spec)
    assert inj.applied
    assert evaluate_outcome(t, w, {"main"})["success"] == 0
    w2, *_, res2 = _run(t, spec, RuleVerifier())
    assert evaluate_outcome(t, w2, {"main"})["success"] == 1
    assert "researcher" in res2.verify.detected_stages


def test_stale_research_detectable_from_read_versions():
    t = _task("multi_wh_one_feasible")
    spec = next(s for s in build_schedule(t) if s.fault_id.startswith("stale_research_relevant"))
    w, inj, snap, store, *_ = _run(t, spec)
    research = store.latest("research")
    key = inv_key(spec.params["warehouse"], spec.params["item"])
    assert w.key_version(key) > research.read_versions[key]


def test_snapshot_restart_loses_external_update_compensate_keeps_it():
    t = _task("multi_wh_one_feasible")
    spec = next(s for s in build_schedule(t) if s.fault_id.startswith("stale_plan_relevant"))
    for mode, lost in (("snapshot", 1), ("compensate", 0)):
        w, inj, snap, store, tr, _ = _run(t, spec)
        rec = full_restart(t, w, spec, make_agents("scripted"), store, tr, snap, mode)
        assert rec["lost_external_updates"] == lost


def test_compromised_planner_unauthorized_actions_counted():
    t = _task("single_wh_multi_item")
    spec = next(s for s in build_schedule(t) if s.fault_id.startswith("compromised_planner"))
    w, *_ = _run(t, spec)
    out = evaluate_outcome(t, w, {"main"})
    assert out["unauthorized_executed"] == 2 and out["refunds_issued"] == 1 and out["foreign_orders_modified"] == 1
    w2, *_ = _run(t, spec, RuleVerifier())
    assert evaluate_outcome(t, w2, {"main"})["unauthorized_executed"] == 0


def test_run_task_rows(tmp_path):
    t = _task("requires_split")
    ctx = Ctx(agents=make_agents("scripted"), verifiers={"rule": RuleVerifier()}, trace_dir=str(tmp_path))
    rows = run_task(copy.deepcopy(t), ctx, ["clean", "no_recovery", "restart_compensate", "verify_rule"])
    conds = {r["condition"] for r in rows}
    assert conds == {"clean", "no_recovery", "restart_compensate", "verify_rule"}
    assert any(tmp_path.iterdir())
