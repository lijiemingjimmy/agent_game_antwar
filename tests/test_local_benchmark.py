from __future__ import annotations

from tools.local_benchmark import AgentSpec, run_match, summarize


def test_agent_spec_parses_mcts_preset() -> None:
    spec = AgentSpec.parse("mcts:16:2")
    assert spec.kind == "mcts"
    assert spec.iterations == 16
    assert spec.depth == 2
    assert spec.label() == "mcts:16:2"


def test_local_benchmark_runs_random_mirror_match() -> None:
    result = run_match(AgentSpec.parse("random"), AgentSpec.parse("random"), seed=3, round_limit=8)
    assert result.rounds == 8
    assert result.base_hp[0] >= 0
    assert result.base_hp[1] >= 0
    summary = summarize([result])
    assert summary.games == 1
