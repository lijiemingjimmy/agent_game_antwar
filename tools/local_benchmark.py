from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution helper
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AI.ai_greedy.ai import AI as GreedyAI
from AI.ai_greedy.runtime import _to_greedy_info, _to_sdk_operation
from AI.ai_mcts import AI as DefaultMCTSAI, MCTSAgent
from AI.ai_random import AI as RandomAI
from SDK.engine import GameState


@dataclass(slots=True)
class AgentSpec:
    kind: str
    iterations: int = 24
    depth: int = 3

    @classmethod
    def parse(cls, raw: str) -> "AgentSpec":
        if raw == "random":
            return cls(kind="random")
        if raw == "greedy":
            return cls(kind="greedy")
        if raw == "mcts":
            return cls(kind="mcts")
        if raw.startswith("mcts:"):
            _, iterations, depth = raw.split(":")
            return cls(kind="mcts", iterations=int(iterations), depth=int(depth))
        raise ValueError(f"unsupported agent spec: {raw}")

    def label(self) -> str:
        if self.kind != "mcts":
            return self.kind
        return f"mcts:{self.iterations}:{self.depth}"


@dataclass(slots=True)
class MatchResult:
    winner: int | None
    rounds: int
    base_hp: tuple[int, int]
    decision_ms: tuple[float, float]
    terminal: bool


@dataclass(slots=True)
class AggregateResult:
    games: int
    wins: tuple[int, int]
    draws: int
    average_rounds: float
    average_base_hp: tuple[float, float]
    average_decision_ms: tuple[float, float]


def _build_agent(spec: AgentSpec, seed: int):
    if spec.kind == "random":
        return RandomAI(seed=seed)
    if spec.kind == "greedy":
        return GreedyAI()
    if spec.kind == "mcts":
        if spec.iterations == 12 and spec.depth == 2:
            agent = DefaultMCTSAI()
            agent.rng.seed(seed)
            return agent
        return MCTSAgent(iterations=spec.iterations, max_depth=spec.depth, seed=seed)
    raise ValueError(f"unsupported agent kind: {spec.kind}")


def _choose_operations(agent, state: GameState, player: int) -> list:
    if hasattr(agent, "choose_operations"):
        return agent.choose_operations(state, player)
    return [_to_sdk_operation(op) for op in agent(player, _to_greedy_info(state))]


def run_match(
    spec0: AgentSpec,
    spec1: AgentSpec,
    *,
    seed: int,
    round_limit: int,
) -> MatchResult:
    state = GameState.initial(seed=seed)
    agents = [_build_agent(spec0, seed), _build_agent(spec1, seed)]
    totals = [0.0, 0.0]
    turns = [0, 0]

    for player, agent in enumerate(agents):
        if hasattr(agent, "on_match_start"):
            agent.on_match_start(player, seed)

    while not state.terminal and state.round_index < round_limit:
        op_lists: list[list] = []
        for player, agent in enumerate(agents):
            started = time.perf_counter()
            op_lists.append(_choose_operations(agent, state, player))
            totals[player] += time.perf_counter() - started
            turns[player] += 1
        state.resolve_turn(op_lists[0], op_lists[1])

    avg_ms = tuple((totals[i] / turns[i] * 1000.0) if turns[i] else 0.0 for i in range(2))
    return MatchResult(
        winner=state.winner,
        rounds=state.round_index,
        base_hp=(state.bases[0].hp, state.bases[1].hp),
        decision_ms=(avg_ms[0], avg_ms[1]),
        terminal=state.terminal,
    )


def summarize(results: list[MatchResult]) -> AggregateResult:
    wins = [0, 0]
    draws = 0
    for result in results:
        if result.winner in (0, 1):
            wins[result.winner] += 1
        else:
            draws += 1
    return AggregateResult(
        games=len(results),
        wins=(wins[0], wins[1]),
        draws=draws,
        average_rounds=mean(result.rounds for result in results),
        average_base_hp=(
            mean(result.base_hp[0] for result in results),
            mean(result.base_hp[1] for result in results),
        ),
        average_decision_ms=(
            mean(result.decision_ms[0] for result in results),
            mean(result.decision_ms[1] for result in results),
        ),
    )


def _print_series(
    left: AgentSpec,
    right: AgentSpec,
    *,
    seeds: list[int],
    round_limit: int,
    label: str,
) -> AggregateResult:
    results = [run_match(left, right, seed=seed, round_limit=round_limit) for seed in seeds]
    summary = summarize(results)
    print(label)
    print(f"  players: P0={left.label()}  P1={right.label()}")
    print(f"  games: {summary.games}  wins: {summary.wins[0]}-{summary.wins[1]}  draws: {summary.draws}")
    print(f"  avg rounds: {summary.average_rounds:.1f}")
    print(f"  avg hp: {summary.average_base_hp[0]:.2f}-{summary.average_base_hp[1]:.2f}")
    print(f"  avg ms/turn: {summary.average_decision_ms[0]:.2f}-{summary.average_decision_ms[1]:.2f}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run lightweight local Antwar benchmarks without training.",
    )
    parser.add_argument("--agent-a", default="mcts", help="random | greedy | mcts | mcts:ITER:DEPTH")
    parser.add_argument("--agent-b", default="random", help="random | greedy | mcts | mcts:ITER:DEPTH")
    parser.add_argument("--games", type=int, default=4, help="number of seeds to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="starting seed")
    parser.add_argument("--round-limit", type=int, default=128, help="maximum rounds per game")
    parser.add_argument("--swap-sides", action="store_true", help="also run the mirrored side assignment")
    args = parser.parse_args(argv)

    left = AgentSpec.parse(args.agent_a)
    right = AgentSpec.parse(args.agent_b)
    seeds = [args.seed + offset for offset in range(args.games)]

    _print_series(left, right, seeds=seeds, round_limit=args.round_limit, label="Series A")
    if args.swap_sides:
        _print_series(right, left, seeds=seeds, round_limit=args.round_limit, label="Series B")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
