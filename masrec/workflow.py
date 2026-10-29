"""Researcher -> Planner -> Executor orchestration with fault hooks and optional verification.

Latency accounting: a stage's latency is its measured wall time plus the original latency of any
cached LLM responses it reused, plus the simulated budget of any injected timeout. This keeps
latency comparable whether or not the response cache was warm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .artifacts import Artifact, ArtifactStore, plan_field_deps
from .faults import Injector
from .tools import ToolBox

STAGES = ("researcher", "planner", "executor")


@dataclass
class StageStats:
    latency_s: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 0
    status: str = "not_run"   # ok | timeout | error | aborted | not_run

    def add_usage(self, usage: list[dict]):
        for u in usage:
            self.llm_calls += 1
            self.prompt_tokens += u["prompt_tokens"]
            self.completion_tokens += u["completion_tokens"]
            if u["cache_hit"]:
                self.latency_s += u["latency_s"]


@dataclass
class VerifyStats:
    checks: int = 0
    failures: int = 0
    latency_s: float = 0.0
    reads: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    detected_stages: list[str] = field(default_factory=list)
    sanitized: int = 0
    issues: list[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    research: dict | None = None
    plan: dict | None = None
    calls: dict | None = None
    exec_results: list[dict] = field(default_factory=list)
    stages: dict[str, StageStats] = field(default_factory=lambda: {s: StageStats() for s in STAGES})
    verify: VerifyStats = field(default_factory=VerifyStats)
    artifact_ids: dict[str, str] = field(default_factory=dict)
    aborted: str | None = None
    action_seqs: list[int] = field(default_factory=list)
    timed_out_stage: str | None = None

    @property
    def latency_s(self) -> float:
        return sum(s.latency_s for s in self.stages.values()) + self.verify.latency_s

    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.stages.values()) + self.verify.llm_calls

    @property
    def tokens(self) -> int:
        return (sum(s.prompt_tokens + s.completion_tokens for s in self.stages.values())
                + self.verify.prompt_tokens + self.verify.completion_tokens)


class Workflow:
    def __init__(self, agents, store: ArtifactStore, trace, verifier=None, max_retries: int = 2,
                 timeout_budget_s: float = 30.0, phase: str = "main"):
        self.researcher, self.planner, self.executor = agents
        self.store = store
        self.trace = trace
        self.verifier = verifier
        self.max_retries = max_retries
        self.timeout_budget_s = timeout_budget_s
        self.phase = phase

    # ------------------------------------------------------------------ helpers
    def _artifact(self, kind, producer, content, depends_on, world, **kw) -> Artifact:
        a = Artifact(self.store.new_id(f"{self.phase}_{kind}"), kind, producer, content, depends_on,
                     world_version=world.version, **kw)
        self.store.add(a)
        self.trace.log("artifact", artifact_id=a.artifact_id, kind=kind, producer=producer,
                       depends_on=depends_on, status=a.status, attempt=a.attempt, content=content,
                       read_versions=a.read_versions, field_deps=a.field_deps, phase=self.phase)
        return a

    def _log_llm(self, usage, messages):
        for k, u in enumerate(usage):
            self.trace.log("llm_call", agent=u["agent"], prompt_tokens=u["prompt_tokens"],
                           completion_tokens=u["completion_tokens"], latency_s=round(u["latency_s"], 3),
                           cache_hit=u["cache_hit"], response=u["response"],
                           prompt=messages[-1]["content"] if (messages and k == 0) else None, phase=self.phase)

    def _verify(self, res: WorkflowResult, stage: str, fn, *args):
        t = time.perf_counter()
        v = fn(*args)
        wall = time.perf_counter() - t
        vs = res.verify
        vs.checks += 1
        vs.reads += v.reads
        vs.latency_s += wall + sum(u["latency_s"] for u in v.usage if u["cache_hit"])
        for u in v.usage:
            vs.llm_calls += 1
            vs.prompt_tokens += u["prompt_tokens"]
            vs.completion_tokens += u["completion_tokens"]
        if not v.ok:
            vs.failures += 1
            if stage not in vs.detected_stages:
                vs.detected_stages.append(stage)
        if v.sanitized is not None:
            vs.sanitized += 1
            if stage not in vs.detected_stages:
                vs.detected_stages.append(stage)
        vs.issues += [f"{stage}: {x}" for x in v.issues]
        self.trace.log("verification", stage=stage, ok=v.ok, issues=v.issues, stale=v.stale,
                       sanitized=v.sanitized is not None, verifier=self.verifier.name, phase=self.phase)
        return v

    # ------------------------------------------------------------------ stages
    def _research(self, task, world, inj: Injector, res: WorkflowResult, attempt: int):
        st = res.stages["researcher"]
        feedback = None
        while True:
            st.attempts += 1
            t = time.perf_counter()
            tb = ToolBox(world, "researcher")
            out = self.researcher.run(task, tb, feedback)
            timed_out = inj.spec is not None and inj.enabled and inj.spec.stage == "researcher" \
                and inj.spec.fault_type == "timeout" and attempt == 0
            content = inj.after_research(out.content, attempt)
            st.latency_s += time.perf_counter() - t
            st.add_usage(out.usage)
            self._log_llm(out.usage, out.messages)
            if timed_out:
                st.latency_s += self.timeout_budget_s
                res.timed_out_stage = "researcher"
                self.trace.log("timeout", stage="researcher", simulated_s=self.timeout_budget_s, phase=self.phase)
            env = self._artifact("env_read", "harness", {"keys": sorted(out.read_versions)}, [], world,
                                 read_versions=out.read_versions)
            art = self._artifact("research", "researcher", content, [env.artifact_id], world, attempt=attempt,
                                 read_versions=out.read_versions,
                                 status="partial" if timed_out else ("injected" if inj.applied and attempt == 0
                                                                      and inj.spec.stage == "researcher" else "ok"))
            res.artifact_ids["researcher"] = art.artifact_id
            if content is None:
                st.status = "error"
            else:
                st.status = "timeout" if timed_out else "ok"
            ev = inj.between("researcher", world)
            if ev:
                self.trace.log("external_event", after_stage="researcher", **ev, phase=self.phase)
            if self.verifier is None:
                return content
            v = self._verify(res, "researcher", self.verifier.check_research, task, content, world)
            if v.ok:
                if v.sanitized is not None:
                    content = v.sanitized
                    art2 = self._artifact("research", "verifier", content, [art.artifact_id], world,
                                          attempt=attempt, status="sanitized")
                    res.artifact_ids["researcher"] = art2.artifact_id
                return content
            if attempt >= self.max_retries:
                st.status = "aborted"
                return None
            attempt += 1
            feedback = v.issues

    def run(self, task: dict, world, inj: Injector) -> WorkflowResult:
        res = WorkflowResult()
        self.trace.log("phase_start", phase=self.phase, world_version=world.version)
        r_attempt = 0
        research = self._research(task, world, inj, res, r_attempt)
        res.research = research
        if research is None:
            res.aborted = "research unavailable"
            return self._finish(res, world)
        self.trace.log("message", sender="researcher", receiver="planner",
                       artifact_id=res.artifact_ids["researcher"], phase=self.phase)

        # ---------------------------------------------------------- planner
        st = res.stages["planner"]
        p_attempt = 0
        p_feedback = None
        plan = None
        while True:
            st.attempts += 1
            t = time.perf_counter()
            if inj.stage_times_out("planner", p_attempt):
                plan = {"feasible": False, "allocation": {}, "actions": [], "rationale": "",
                        "error": "planner timed out"}
                st.latency_s += self.timeout_budget_s
                st.status = "timeout"
                res.timed_out_stage = "planner"
                self.trace.log("timeout", stage="planner", simulated_s=self.timeout_budget_s, phase=self.phase)
            else:
                out = self.planner.run(task, research, p_feedback)
                plan = inj.after_plan(out.content, p_attempt)
                st.add_usage(out.usage)
                self._log_llm(out.usage, out.messages)
                st.status = "ok" if plan is not None else "error"
            st.latency_s += time.perf_counter() - t
            art = self._artifact("plan", "planner", plan, [res.artifact_ids["researcher"]], world, attempt=p_attempt,
                                 field_deps=plan_field_deps((plan or {}).get("actions", [])),
                                 status="injected" if (inj.applied and inj.spec.stage == "planner" and p_attempt == 0)
                                 else st.status)
            res.artifact_ids["planner"] = art.artifact_id
            ev = inj.between("planner", world)
            if ev:
                self.trace.log("external_event", after_stage="planner", **ev, phase=self.phase)
            if self.verifier is None:
                break
            v = self._verify(res, "planner", self.verifier.check_plan, task, plan, world)
            if v.ok:
                break
            if p_attempt >= self.max_retries:
                st.status = "aborted"
                plan = None
                break
            p_attempt += 1
            p_feedback = v.issues
            if v.stale:
                r_attempt += 1
                research = self._research(task, world, inj, res, r_attempt)
                res.research = research
                if research is None:
                    res.aborted = "research unavailable"
                    return self._finish(res, world)
        res.plan = plan
        if plan is None:
            res.aborted = "plan unavailable"
            return self._finish(res, world)
        self.trace.log("message", sender="planner", receiver="executor",
                       artifact_id=res.artifact_ids["planner"], phase=self.phase)

        # ---------------------------------------------------------- executor
        st = res.stages["executor"]
        e_attempt = 0
        e_feedback = None
        calls = None
        while True:
            st.attempts += 1
            t = time.perf_counter()
            out = self.executor.run(task, plan, e_feedback)
            calls = inj.after_calls(out.content, e_attempt, world)
            st.add_usage(out.usage)
            self._log_llm(out.usage, out.messages)
            st.latency_s += time.perf_counter() - t
            art = self._artifact("tool_calls", "executor", calls, [res.artifact_ids["planner"]], world,
                                 attempt=e_attempt,
                                 status="injected" if (inj.applied and inj.spec.stage == "executor" and e_attempt == 0)
                                 else "ok")
            res.artifact_ids["executor"] = art.artifact_id
            if self.verifier is None or calls is None:
                break
            v = self._verify(res, "executor", self.verifier.check_calls, task, plan, calls, world)
            if v.ok:
                break
            if e_attempt >= self.max_retries:
                calls = None
                break
            e_attempt += 1
            e_feedback = v.issues
        res.calls = calls
        if calls is None:
            st.status = "aborted" if self.verifier else "error"
            res.aborted = "tool calls unavailable"
            return self._finish(res, world)

        tool_calls = calls.get("tool_calls", [])
        cutoff = inj.executor_cutoff(len(tool_calls), 0)
        t = time.perf_counter()
        tb = ToolBox(world, "executor")
        to_run = list(enumerate(tool_calls))
        first = to_run if cutoff is None else to_run[:cutoff]
        self._execute(first, tb, res, art.artifact_id)
        st.status = "ok"
        if cutoff is not None:
            st.latency_s += self.timeout_budget_s
            st.status = "timeout"
            res.timed_out_stage = "executor"
            self.trace.log("timeout", stage="executor", executed=cutoff, of=len(tool_calls),
                           simulated_s=self.timeout_budget_s, phase=self.phase)
            if self.verifier is not None:
                # verified workflows notice the incomplete execution and resume the remaining calls
                res.verify.detected_stages.append("executor")
                st.attempts += 1
                self._execute(to_run[cutoff:], tb, res, art.artifact_id)
                st.status = "ok"
        st.latency_s += time.perf_counter() - t
        exe = self._artifact("execution", "executor", res.exec_results, [art.artifact_id], world)
        res.artifact_ids["execution"] = exe.artifact_id
        return self._finish(res, world)

    def _execute(self, calls, tb: ToolBox, res: WorkflowResult, art_id: str):
        for idx, c in calls:
            r = tb.call(c["tool"], c["args"], meta={"phase": self.phase, "artifact": art_id, "call_idx": idx})
            seq = len(tb.world.action_log) - 1
            res.action_seqs.append(seq)
            entry = {"call_idx": idx, "tool": c["tool"], "args": c["args"], "ok": r.ok, "error": r.error,
                     "result": r.result, "seq": seq}
            res.exec_results.append(entry)
            self.trace.log("tool_call", actor="executor", **entry,
                           reversible=tb.world.action_log[seq]["reversible"],
                           write_keys=tb.world.action_log[seq]["write_keys"], world_version=tb.world.version,
                           phase=self.phase)

    def _finish(self, res: WorkflowResult, world) -> WorkflowResult:
        for s in STAGES:
            if res.stages[s].status == "not_run" and res.aborted:
                res.stages[s].status = "skipped"
        self.trace.log("phase_end", phase=self.phase, aborted=res.aborted, world_version=world.version,
                       latency_s=round(res.latency_s, 3), llm_calls=res.llm_calls, tokens=res.tokens)
        return res
