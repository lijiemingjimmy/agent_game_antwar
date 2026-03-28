from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from SDK.constants import (
    AntBehavior,
    MAX_ACTIONS,
    OperationType,
    PLAYER_BASES,
    STRATEGIC_BUILD_ORDER,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TOWER_UPGRADE_TREE,
    TowerType,
)
from SDK.engine import GameState
from SDK.features import FeatureExtractor
from SDK.geometry import hex_distance
from SDK.model import Operation, Tower


@dataclass(slots=True)
class ActionBundle:
    name: str
    operations: tuple[Operation, ...] = ()
    score: float = 0.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def protocol_lines(self) -> list[list[int]]:
        return [op.to_protocol_tokens() for op in self.operations]


class ActionCatalog:
    def __init__(self, max_actions: int = MAX_ACTIONS, feature_extractor: FeatureExtractor | None = None) -> None:
        self.max_actions = max_actions
        self.feature_extractor = feature_extractor or FeatureExtractor(max_actions=max_actions)

    def build(self, state: GameState, player: int) -> list[ActionBundle]:
        bundles: list[ActionBundle] = [ActionBundle(name="hold", score=0.0, tags=("noop",))]
        bundles.extend(self._build_candidates(state, player))
        bundles.extend(self._upgrade_candidates(state, player))
        bundles.extend(self._downgrade_candidates(state, player))
        bundles.extend(self._base_upgrade_candidates(state, player))
        bundles.extend(self._superweapon_candidates(state, player))
        bundles.extend(self._scripted_combo_candidates(state, player, bundles[1:]))
        bundles.extend(self._paired_candidates(state, player, bundles[1:]))
        unique: dict[tuple[tuple[int, int, int], ...], ActionBundle] = {}
        for bundle in bundles:
            key = tuple((int(op.op_type), op.arg0, op.arg1) for op in bundle.operations)
            if key not in unique or bundle.score > unique[key].score:
                unique[key] = bundle
        ordered = sorted(unique.values(), key=lambda item: item.score, reverse=True)
        staged = self._phase_select(state, player, ordered)
        reranked = self._rerank_with_one_step_rollout(state, player, staged[: min(len(staged), self.max_actions * 2)])
        return reranked[: self.max_actions]

    def action_mask(self, bundles: list[ActionBundle]) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=np.int8)
        mask[: len(bundles)] = 1
        return mask

    def bundle_for_index(self, bundles: list[ActionBundle], action_index: int) -> ActionBundle:
        if 0 <= action_index < len(bundles):
            return bundles[action_index]
        return bundles[0]

    def _build_candidates(self, state: GameState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        tower_count = state.tower_count(player)
        build_cost = state.build_tower_cost(tower_count)
        if state.coins[player] < build_cost:
            return results
        safe_gap = state.coins[player] - state.safe_coin_threshold(player)
        for x, y in STRATEGIC_BUILD_ORDER[player]:
            op = Operation(OperationType.BUILD_TOWER, x, y)
            if not state.can_apply_operation(player, op):
                continue
            pressure = self._local_enemy_pressure(state, player, x, y)
            lane_bonus = state.slot_priority(player, x, y)
            branch_fit = max(
                self._tower_type_fit(state, player, x, y, target)
                for target in TOWER_UPGRADE_TREE[TowerType.BASIC]
            )
            score = lane_bonus * 0.55 + pressure * 1.8 + branch_fit * 0.65
            score -= build_cost * (0.05 + 0.01 * tower_count)
            if tower_count >= 2 and state.bases[player].generation_level + state.bases[player].ant_level < 2:
                score -= 2.0
            if safe_gap < 0:
                score += safe_gap * 0.05
            results.append(ActionBundle(name=f"build@{x},{y}", operations=(op,), score=score, tags=("build",)))
        return results

    def _upgrade_candidates(self, state: GameState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        for tower in state.towers_of(player):
            for target in TOWER_UPGRADE_TREE.get(tower.tower_type, ()): 
                op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
                if not state.can_apply_operation(player, op):
                    continue
                fit = self._tower_type_fit(state, player, tower.x, tower.y, target)
                score = fit + tower.level * 1.5 + state.slot_priority(player, tower.x, tower.y) * 0.1
                results.append(
                    ActionBundle(
                        name=f"upgrade#{tower.tower_id}->{int(target)}",
                        operations=(op,),
                        score=score,
                        tags=("upgrade", f"tower:{int(target)}"),
                    )
                )
        return results

    def _downgrade_candidates(self, state: GameState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        for tower in state.towers_of(player):
            pressure = self._local_enemy_pressure(state, player, tower.x, tower.y)
            if pressure > 1.5:
                continue
            if state.round_index < 96 and tower.level == 0:
                # Early sell-churn showed up in replay samples and rarely pays back in a shallow search.
                continue
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if not state.can_apply_operation(player, op):
                continue
            refund = state._operation_income(player, op)
            score = refund * 0.04 - state.slot_priority(player, tower.x, tower.y) * 0.3 - tower.level * 3.0
            results.append(ActionBundle(name=f"downgrade#{tower.tower_id}", operations=(op,), score=score, tags=("sell",)))
        return results

    def _base_upgrade_candidates(self, state: GameState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        safe_coin = max(state.coins[player] - state.safe_coin_threshold(player), 0)
        tower_count = state.tower_count(player)
        early_rounds = max(0.0, 128.0 - state.round_index)
        if state.bases[player].ant_level < 2:
            op = Operation(OperationType.UPGRADE_GENERATED_ANT)
            if state.can_apply_operation(player, op):
                score = 24.0 - state.round_index * 0.012 + state.frontline_distance(player) * 0.35
                score += min(safe_coin, 120) * 0.02
                score += min(tower_count, 4) * 0.9
                score += early_rounds * 0.015
                results.append(ActionBundle("upgrade-ant", (op,), score, ("base", "offense")))
        if state.bases[player].generation_level < 2:
            op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
            if state.can_apply_operation(player, op):
                score = 20.0 - state.round_index * 0.016 + state.nearest_ant_distance(player) * 0.18
                score += min(safe_coin, 120) * 0.018
                score += max(0, 3 - tower_count) * 0.8
                score += early_rounds * 0.012
                results.append(ActionBundle(upgrade-gen", (op,), score, ("base", "tempo")))
        return results

    def _superweapon_candidates(self, state: GameState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        enemy = 1 - player
        enemy_ants = state.ants_of(enemy)
        my_ants = state.ants_of(player)
        enemy_towers = state.towers_of(enemy)
        if enemy_ants and state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost:
            best = max(((ant.x, ant.y, self._storm_value(state, player, ant.x, ant.y)) for ant in enemy_ants), key=lambda item: item[2], default=None)
            if best and best[2] > 2.5:
                op = Operation(OperationType.USE_LIGHTNING_STORM,best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"storm@{best[0]},{best[1]}", (op,), best[2], ("weapon", "storm")))
        if enemy_towers and state.weapon_cooldowns[player, SuperWeaponType.EMP_BLASTER] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost:
            centers = {(tower.x, tower.y) for tower in enemy_towers}
            scored = [(x, y, self._emp_value(state, player, x, y)) for x, y in centers]
            best = max(scored, key=lambda item: item[2], default=None)
            if best and best[2] > 2.0:
                op = Operation(OperationType.USE_EMP_BLASTER, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"emp@{best[0]},{best[1]}", (op,), best[2], ("weapon", "emp")))
        if my_ants and state.weapon_cooldowns[player, SuperWeaponType.DEFLECTOR] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost:
            best = max(((ant.x, ant.y, self._deflector_value(state, player, ant.x, ant.y)) for ant in my_ants), key=lambda item: item[2], default=None)
            if best and best[2] > 1.5:
                op = Operation(OperationType.USE_DEFLECTOR, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"deflect@{best[0]},{best[1]}", (op,), best[2], ("weapon", "shield")))
        if my_ants and state.weapon_cooldowns[player, SuperWeaponType.EMERGENCY_EVASION] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost:
            best = max(((ant.x, ant.y, self._evasion_value(state, player, ant.x, ant.y)) for ant in my_ants), key=lambda item: item[2], default=None)
            if best and best[2] > 1.0:
                op = Operation(OperationType.USE_EMERGENCY_EVASION, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"evasion@{best[0]},{best[1]}", (op,), best[2], ("weapon", "panic")))
        return results
    def _top_by_tag(self, singles: list[ActionBundle], tag: str, limit: int) -> list[ActionBundle]:
        return [bundle for bundle in singles if bundle.tags and bundle.tags[0] == tag][:limit]
    def _combine_bundles(self,state: GameState,player: int,first: ActionBundle,second: ActionBundle,*