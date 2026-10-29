"""The researcher -> planner -> executor workflow built on LangGraph, with LangGraph checkpointing as a
recovery baseline.

Nodes reuse masrec's agents and fault hooks, so prompts (and cached LLM responses) match the native
workflow. LangGraph's ``MemorySaver`` checkpoints the graph state after every node. Rewinding to the
checkpoint before the faulty node and resuming is the framework's built-in recovery; it restores
*agent* state only, not the shared world (inventory, reservations, shipments, emails).
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .artifacts import ArtifactStore
from .faults import Injector
from .tools import ToolBox
from .workflow import STAGES, Workflow, WorkflowResult

NODE_OF = {"researcher": "research", "planner": "plan", "executor": "execute"}


class GraphState(TypedDict, total=False):
    research: Any
    plan: Any
    calls: Any
    exec_results: list
    aborted: str


class LangGraphWorkflow:
    def __init__(self, agents, trace, timeout_budget_s: float = 30.0):
        self.agents = agents
        self.trace = trace
        self.timeout_budget_s = timeout_budget_s
        self.checkpointer = MemorySaver()
        # per-invocation context set by run()/resume()
        self.task = self.world = self.inj = None
        self.res: WorkflowResult | None = None
        self.wf: Workflow | None = None
        self.nodes_run: list[str] = []
        g = StateGraph(GraphState)
        g.add_node("research", self._research)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.add_edge(START, "research")
        g.add_conditional_edges("research", lambda s: END if s.get("aborted") else "plan")
        g.add_conditional_edges("plan", lambda s: END if s.get("aborted") else "execute")
        g.add_edge("execute", END)
        self.graph = g.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------ nodes
    def _research(self, state: GraphState) -> GraphState:
        self.nodes_run.append("research")
        research = self.wf._research(self.task, self.world, self.inj, self.res, 0)
        self.res.research = research
        return {"research": research, "aborted": "" if research is not None else "research unavailable"}

    def _plan(self, state: GraphState) -> GraphState:
        self.nodes_run.append("plan")
        st, inj = self.res.stages["planner"], self.inj
        st.attempts += 1
        t = time.perf_counter()
        if inj.stage_times_out("planner", 0):
            plan = {"feasible": False, "allocation": {}, "actions": [], "rationale": "", "error": "planner timed out"}
            st.latency_s += self.timeout_budget_s
            st.status = "timeout"
            self.res.timed_out_stage = "planner"
        else:
            out = self.agents[1].run(self.task, state["research"])
            plan = inj.after_plan(out.content, 0)
            st.add_usage(out.usage)
            st.status = "ok" if plan is not None else "error"
        st.latency_s += time.perf_counter() - t
        inj.between("planner", self.world)
        self.res.plan = plan
        return {"plan": plan, "aborted": "" if plan is not None else "plan unavailable"}

    def _execute(self, state: GraphState) -> GraphState:
        self.nodes_run.append("execute")
        st, inj = self.res.stages["executor"], self.inj
        st.attempts += 1
        t = time.perf_counter()
        out = self.agents[2].run(self.task, state["plan"])
        calls = inj.after_calls(out.content, 0, self.world)
        st.add_usage(out.usage)
        results = []
        if calls is not None:
            tool_calls = calls.get("tool_calls", [])
            cutoff = inj.executor_cutoff(len(tool_calls), 0)
            todo = list(enumerate(tool_calls))
            if cutoff is not None:
                todo = todo[:cutoff]
                st.latency_s += self.timeout_budget_s
                self.res.timed_out_stage = "executor"
            tb = ToolBox(self.world, "executor")
            self.wf._execute(todo, tb, self.res, "lg_tool_calls")
            results = self.res.exec_results
        st.status = "ok" if calls is not None else "error"
        st.latency_s += time.perf_counter() - t
        self.res.calls = calls
        return {"calls": calls, "exec_results": results}

    # ------------------------------------------------------------------ driving
    def _bind(self, task, world, inj, phase):
        self.task, self.world, self.inj = task, world, inj
        self.res = WorkflowResult()
        self.wf = Workflow(self.agents, ArtifactStore(), self.trace, verifier=None,
                           timeout_budget_s=self.timeout_budget_s, phase=phase)
        self.nodes_run = []

    def run(self, task, world, inj: Injector, thread_id: str) -> WorkflowResult:
        self._bind(task, world, inj, "main")
        self.graph.invoke({}, {"configurable": {"thread_id": thread_id}})
        return self.res

    def checkpoint_before(self, thread_id: str, stage: str):
        """Config of the saved checkpoint whose next node is the faulty stage."""
        node = NODE_OF[stage]
        for snap in self.graph.get_state_history({"configurable": {"thread_id": thread_id}}):
            if snap.next == (node,):
                return snap.config
        return None

    def resume(self, task, world, inj: Injector, config) -> WorkflowResult:
        """Time-travel: replay from a past checkpoint (a new branch of the thread)."""
        self._bind(task, world, inj, "recovery")
        self.graph.invoke(None, config)
        return self.res

    def n_checkpoints(self, thread_id: str) -> int:
        return sum(1 for _ in self.graph.get_state_history({"configurable": {"thread_id": thread_id}}))


def stages_before(stage: str) -> int:
    return STAGES.index(stage)
