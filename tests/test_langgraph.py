"""LangGraph workflow: same scripted agents, checkpoint rewind as a recovery baseline."""
from masrec.agents import make_agents
from masrec.evaluator import evaluate_outcome
from masrec.faults import Injector, build_schedule
from masrec.langgraph_workflow import LangGraphWorkflow, stages_before
from masrec.tasks import generate_tasks
from masrec.tools import ToolBox
from masrec.trace import Trace
from masrec.world import World


def _task(tid="T01"):
    return next(t for t in generate_tasks() if t["task_id"] == tid)


def test_clean_matches_scripted():
    task = _task()
    lg = LangGraphWorkflow(make_agents("scripted"), Trace("lg"))
    world = World(task["initial_state"])
    res = lg.run(task, world, Injector(None), "t")
    assert evaluate_outcome(task, world, {"main"})["success"]
    assert res.research and res.plan and res.calls
    assert lg.n_checkpoints("t") >= 3


def test_checkpoint_before_faulty_stage():
    task = _task()
    spec = next(s for s in build_schedule(task) if s.stage == "planner")
    lg = LangGraphWorkflow(make_agents("scripted"), Trace("lg"))
    world = World(task["initial_state"])
    lg.run(task, world, Injector(spec), "t")
    cfg = lg.checkpoint_before("t", "planner")
    assert cfg is not None
    assert stages_before("planner") == 1


def test_checkpoint_alone_leaves_world_dirty():
    """LangGraph restores agent state, not the shared world."""
    task = _task()
    spec = next(s for s in build_schedule(task) if s.fault_id.startswith("exec_wrong"))
    lg = LangGraphWorkflow(make_agents("scripted"), Trace("lg"))
    world = World(task["initial_state"])
    inj = Injector(spec)
    inj.prepare_world(world, task)
    lg.run(task, world, inj, "t")
    n_ship = len(world.state["shipments"])
    cfg = lg.checkpoint_before("t", spec.stage)
    assert cfg is not None
    lg.resume(task, world, Injector(spec, enabled=False), cfg)
    # world was not rolled back: shipments from the faulted run are still there
    assert len(world.state["shipments"]) >= n_ship


def test_compensate_then_checkpoint_clears_reversible():
    task = _task()
    spec = next(s for s in build_schedule(task) if s.fault_id.startswith("exec_wrong"))
    lg = LangGraphWorkflow(make_agents("scripted"), Trace("lg"))
    world = World(task["initial_state"])
    inj = Injector(spec)
    inj.prepare_world(world, task)
    lg.run(task, world, inj, "t")
    cfg = lg.checkpoint_before("t", spec.stage)
    tb = ToolBox(world, "recovery")
    for e in reversed([e for e in world.action_log if e["actor"] == "executor" and e["ok"]]):
        if e["reversible"] and e.get("compensation"):
            tb.compensate(e["compensation"])
    lg.resume(task, world, Injector(spec, enabled=False), cfg)
    assert not [r for r in world.state["reservations"] if r["status"] == "active"]
