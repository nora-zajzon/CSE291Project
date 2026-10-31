"""tau-bench harness tests with a ground-truth-replay agent (no LLM)."""
import pytest
from types import SimpleNamespace

tb = pytest.importorskip("masrec.taubench")
from tau_bench.envs.retail.tasks_test import TASKS_TEST  # noqa: E402
from tau_bench.envs.retail.data import load_data  # noqa: E402


def _first_task(tool):
    return next(i for i, t in enumerate(TASKS_TEST)
                if tb.gt_writes(t) and tb.gt_writes(t)[0].name == tool and len(tb.gt_writes(t)) == 1)


EXCHANGE = _first_task("exchange_delivered_order_items")
ADDRESS = _first_task("modify_pending_order_address")


def run(i, fault=None, **kw):
    tb.tca.completion = tb.ReplayLLM(i)
    f = tb.make_fault(fault, TASKS_TEST[i], load_data()) if fault else None
    return tb.run_agent(i, "replay", "none", fault=f, **kw), f


def test_clean_replay_scores_one():
    for i in (EXCHANGE, ADDRESS):
        rr, _ = run(i)
        assert rr.error is None and rr.reward == 1.0


def test_irrelevant_read_fault_is_harmless_and_detected():
    rr, f = run(EXCHANGE, "corrupt_read_irrelevant")
    assert f.applied and rr.reward == 1.0
    rr, f = run(EXCHANGE, "corrupt_read_irrelevant", verify=True)
    assert rr.verifier.failures == 1 and rr.reward == 1.0


def test_wrong_args_irreversible_and_verifier_blocks_it():
    rr, f = run(EXCHANGE, "wrong_args")
    assert f.applied and rr.reward == 0.0 and len(tb.irreversible_sig(rr.log)) == 1
    rr, f = run(EXCHANGE, "wrong_args", verify=True)
    assert rr.verifier.failures == 1 and rr.reward == 1.0


def test_compensation_undoes_reversible_but_not_irreversible():
    rr, _ = run(ADDRESS, "wrong_args")
    assert rr.reward == 0.0
    start, undone, kept = tb.compensate(rr.final_data, rr.log)
    assert undone == 1 and kept == 0
    rec, _ = run(ADDRESS, "wrong_args", start_data=start, persistent_only=True)
    assert rec.reward == 1.0
    rr, _ = run(EXCHANGE, "wrong_args")
    start, undone, kept = tb.compensate(rr.final_data, rr.log)
    assert undone == 0 and kept == 1
    rec, _ = run(EXCHANGE, "wrong_args", start_data=start, persistent_only=True)
    assert rec.reward == 0.0       # the wrong exchange cannot be taken back


def test_transient_and_lost_response_with_retry():
    rr, f = run(EXCHANGE, "transient_error")
    assert f.applied and rr.reward == 1.0
    rr, f = run(EXCHANGE, "lost_response")
    assert f.applied and rr.reward == 1.0      # the retry fails harmlessly (order no longer 'delivered')


def test_injection_is_persistent_graded_on_same_data_and_sanitized():
    rr, f = run(EXCHANGE, "injection")
    assert f.applied and rr.reward == 1.0
    assert rr.final_data["orders"][f.target_order]["note"] == tb.INJECTION_TEXT
    rr, f = run(EXCHANGE, "injection", verify=True)
    assert rr.verifier.failures >= 1


def test_exposure_requires_faulty_content_in_a_later_model_prompt():
    rr = SimpleNamespace(log=[{"faulty_observation": "bad fact"}], usage=[])
    assert not tb.fault_exposed(rr, "corrupt_read")  # replay supplies no evidence
    rr.usage = [{"messages": [{"role": "tool", "content": "repaired fact"}]}]
    assert not tb.fault_exposed(rr, "corrupt_read")
    rr.usage = [{"messages": [{"role": "tool", "content": "bad fact"}]}]
    assert tb.fault_exposed(rr, "corrupt_read")
    assert not tb.fault_exposed(rr, "injection")
    rr.usage = [{"messages": [{"role": "tool", "content": tb.INJECTION_TEXT}]}]
    assert tb.fault_exposed(rr, "injection")


def test_trace_dependencies_precede_events_and_reads_have_observations():
    rr, _ = run(EXCHANGE)
    for i, e in enumerate(rr.log):
        assert e["event_id"] == i
        assert e["depends_on"] == list(range(i))  # conservative conversation dependencies
        if not e["write"]:
            assert isinstance(e["observation"], str)


def test_prefetch_routes_current_records_through_fault_and_verifier():
    clean = tb.FaultyRetailEnv(EXCHANGE, prefetch=True).reset(EXCHANGE).observation
    f = tb.make_fault("corrupt_read", TASKS_TEST[EXCHANGE], load_data())
    env = tb.FaultyRetailEnv(EXCHANGE, fault=f, prefetch=True)
    corrupt = env.reset(EXCHANGE).observation
    assert f.applied and corrupt != clean
    assert any(e.get("faulty_observation") for e in env.log)
    f = tb.make_fault("corrupt_read", TASKS_TEST[EXCHANGE], load_data())
    verifier = tb.TauRuleVerifier()
    env = tb.FaultyRetailEnv(EXCHANGE, fault=f, verifier=verifier, prefetch=True)
    assert env.reset(EXCHANGE).observation == clean
    assert f.applied and verifier.failures == 1


def test_agent_state_comparison_excludes_the_injected_note_only():
    data = load_data()
    clean_hash = tb.data_hash(data)
    f = tb.make_fault("injection", TASKS_TEST[EXCHANGE], data)
    tb.inject_data(data, f)
    assert tb.data_hash(data) != clean_hash
    assert tb.agent_data_hash(data) == clean_hash
    data["users"][f.user_id]["address"] = tb.ATTACKER_ADDRESS
    assert tb.agent_data_hash(data) != clean_hash


def test_verifier_sanitizes_injected_notes_in_successful_write_responses():
    f = tb.make_fault("injection", TASKS_TEST[ADDRESS], load_data())
    verifier = tb.TauRuleVerifier()
    env = tb.FaultyRetailEnv(ADDRESS, fault=f, verifier=verifier, prefetch=True)
    assert tb.INJECTION_TEXT not in env.reset(ADDRESS).observation
    result = env.step(tb.gt_writes(TASKS_TEST[ADDRESS])[0])
    assert tb.INJECTION_TEXT not in result.observation
    assert env.data["orders"][f.target_order]["note"] == tb.INJECTION_TEXT
    assert verifier.failures >= 2  # initial read and successful write return
