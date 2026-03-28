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
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if not state.can_apply_operation(player, op):
                continue
            refund = state._operation_income(player, op)
            score = refund * 0.04 - state.slot_priority(player, tower.x, tower.y) * 0.3 - tower.level * 3.0
            results.append(ActionBundle(name=f"downgrade#{tower.tower_id}", operations=(op,), score=score, tags=("sell",)))
        return results

    def _base_upgrade_candidates(self, state: GameState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        if state.bases[player].ant_level < 2:
            op = Operation(OperationType.UPGRADE_GENERATED_ANT)
            if state.can_apply_operation(player, op):
                score = 22.0 - state.round_index * 0.015 + state.frontline_distance(player) * 0.3
                results.append(ActionBundle("upgrade-ant", (op,), score, ("base", "offense")))
        if state.bases[player].generation_level < 2:
            op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
            if state.can_apply_operation(player, op):
                score = 18.0 - state.round_index * 0.02 + state.nearest_ant_distance(player) * 0.15
                results.append(ActionBundle("upgrade-gen", (op,), score, ("base", "tempo")))
        return results

    def _superweapon_candidates(self, state: GameState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        enemy = 1 - player
        enemy_ants = state.ants_of(enemy)
        my_ants = state.ants_of(player)
        enemy_towers = state.towers_of(enemy)

        if enemy_ants and state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost:
            best = max(
                ((ant.x, ant.y, self._storm_value(state, player, ant.x, ant.y)) for ant in enemy_ants),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 2.5:
                op = Operation(OperationType.USE_LIGHTNING_STORM, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"storm@{best[0]},{best[1]}", (op,), best[2], ("weapon", "storm")))

        if enemy_towers and state.weapon_cooldowns[player, SuperWeaponType.EMP_BLASTER] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost:
            centers = {(tower.x, tower.y) for tower in enemy_towers}
            scored = [
                (x, y, self._emp_value(state, player, x, y))
                for x, y in centers
            ]
            best = max(scored, key=lambda item: item[2], default=None)
            if best and best[2] > 2.0:
                op = Operation(OperationType.USE_EMP_BLASTER, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"emp@{best[0]},{best[1]}", (op,), best[2], ("weapon", "emp")))

        if my_ants and state.weapon_cooldowns[player, SuperWeaponType.DEFLECTOR] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost:
            best = max(
                ((ant.x, ant.y, self._deflector_value(state, player, ant.x, ant.y)) for ant in my_ants),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 1.5:
                op = Operation(OperationType.USE_DEFLECTOR, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"deflect@{best[0]},{best[1]}", (op,), best[2], ("weapon", "shield")))

        if my_ants and state.weapon_cooldowns[player, SuperWeaponType.EMERGENCY_EVASION] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost:
            best = max(
                ((ant.x, ant.y, self._evasion_value(state, player, ant.x, ant.y)) for ant in my_ants),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 1.0:
                op = Operation(OperationType.USE_EMERGENCY_EVASION, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"evasion@{best[0]},{best[1]}", (op,), best[2], ("weapon", "panic")))

        return results

    def _top_by_tag(self, singles: list[ActionBundle], tag: str, limit: int) -> list[ActionBundle]:
        return [bundle for bundle in singles if bundle.tags and bundle.tags[0] == tag][:limit]

    def _combine_bundles(
        self,
        state: GameState,
        player: int,
        first: ActionBundle,
        second: ActionBundle,
        *,
        tag: str,
        bonus: float = 0.0,
    ) -> ActionBundle | None:
        operations = first.operations + second.operations
        trial = state.clone()
        for op in operations:
            if not trial.can_apply_operation(player, op):
                return None
            trial.apply_operation(player, op)
        score = first.score + second.score * 0.82 + bonus
        name = f"{first.name}+{second.name}"
        return ActionBundle(name=name, operations=tuple(operations), score=score, tags=("combo", tag))

    def _scripted_combo_candidates(self, state: GameState, player: int, singles: list[ActionBundle]) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        builds = self._top_by_tag(singles, "build", 4)
        upgrades = self._top_by_tag(singles, "upgrade", 4)
        sells = self._top_by_tag(singles, "sell", 3)
        bases = self._top_by_tag(singles, "base", 2)
        weapons = self._top_by_tag(singles, "weapon", 3)

        for build in builds:
            for base in bases:
                combo = self._combine_bundles(state, player, build, base, tag="build-base", bonus=1.8)
                if combo is not None:
                    results.append(combo)

        for build in builds:
            for upgrade in upgrades:
                combo = self._combine_bundles(state, player, build, upgrade, tag="build-upgrade", bonus=0.8)
                if combo is not None:
                    results.append(combo)

        for sell in sells:
            for base in bases:
                combo = self._combine_bundles(state, player, sell, base, tag="sell-base", bonus=1.2)
                if combo is not None:
                    results.append(combo)

        for weapon in weapons:
            for base in bases:
                combo = self._combine_bundles(state, player, weapon, base, tag="weapon-base", bonus=0.6)
                if combo is not None:
                    results.append(combo)

        for weapon in weapons:
            for build in builds[:2]:
                combo = self._combine_bundles(state, player, weapon, build, tag="weapon-build", bonus=0.4)
                if combo is not None:
                    results.append(combo)

        return results

    def _paired_candidates(self, state: GameState, player: int, singles: list[ActionBundle]) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        left = [bundle for bundle in singles if bundle.tags and bundle.tags[0] in {"sell", "build", "upgrade", "base"}]
        left = sorted(left, key=lambda item: item.score, reverse=True)[:8]
        scripted_pairs = {
            frozenset(("build", "base")),
            frozenset(("build", "upgrade")),
            frozenset(("sell", "base")),
        }
        for index, first in enumerate(left):
            for second in left[index + 1 :]:
                if first.tags[0] == second.tags[0]:
                    continue
                if frozenset((first.tags[0], second.tags[0])) in scripted_pairs:
                    continue
                combo = self._combine_bundles(state, player, first, second, tag="generic")
                if combo is not None:
                    results.append(combo)
        return results

    def _phase_limits(self, state: GameState, player: int) -> dict[str, int]:
        tower_count = state.tower_count(player)
        if state.round_index < 64:
            return {
                "combo:build-base": 4,
                "combo:build-upgrade": 1,
                "combo:sell-base": 0,
                "combo:weapon-base": 1,
                "combo:weapon-build": 1,
                "combo:generic": 1,
                "build": 8,
                "upgrade": 3,
                "base": 3,
                "weapon": 2,
                "sell": 1,
                "noop": 1,
            }
        if state.round_index < 160:
            return {
                "combo:build-base": 2,
                "combo:build-upgrade": 4,
                "combo:sell-base": 1,
                "combo:weapon-base": 2,
                "combo:weapon-build": 2,
                "combo:generic": 2,
                "build": 5,
                "upgrade": 6,
                "base": 3,
                "weapon": 3,
                "sell": 2,
                "noop": 1,
            }
        build_cap = 3 if tower_count < 5 else 1
        return {
            "combo:build-base": 1,
            "combo:build-upgrade": 2,
            "combo:sell-base": 3,
            "combo:weapon-base": 2,
            "combo:weapon-build": 1,
            "combo:generic": 2,
            "build": build_cap,
            "upgrade": 5,
            "base": 3,
            "weapon": 4,
            "sell": 4,
            "noop": 1,
        }

    def _bundle_bucket(self, bundle: ActionBundle) -> str:
        if not bundle.tags:
            return ""
        if bundle.tags[0] == "combo" and len(bundle.tags) > 1:
            return f"combo:{bundle.tags[1]}"
        return bundle.tags[0]

    def _phase_select(self, state: GameState, player: int, ordered: list[ActionBundle]) -> list[ActionBundle]:
        limits = self._phase_limits(state, player)
        selected: list[ActionBundle] = []
        counts: dict[str, int] = {}
        deferred: list[ActionBundle] = []
        for bundle in ordered:
            bucket = self._bundle_bucket(bundle)
            limit = limits.get(bucket, 2)
            if counts.get(bucket, 0) < limit:
                selected.append(bundle)
                counts[bucket] = counts.get(bucket, 0) + 1
            else:
                deferred.append(bundle)
        for bundle in deferred:
            if len(selected) >= self.max_actions * 2:
                break
            selected.append(bundle)
        return selected

    def _rerank_with_one_step_rollout(self, state: GameState, player: int, bundles: list[ActionBundle]) -> list[ActionBundle]:
        baseline = self.feature_extractor.evaluate(state, player)
        reranked: list[ActionBundle] = []
        for bundle in bundles:
            trial = state.clone()
            trial.apply_operation_list(player, bundle.operations)
            trial.advance_round()
            rollout_value = self.feature_extractor.evaluate(trial, player) - baseline
            reranked.append(ActionBundle(bundle.name, bundle.operations, bundle.score + rollout_value * 0.2, bundle.tags))
        reranked.sort(key=lambda item: item.score, reverse=True)
        if not reranked:
            return [ActionBundle(name="hold")]
        return reranked

    def _local_enemy_pressure(self, state: GameState, player: int, x: int, y: int) -> float:
        pressure = 0.0
        for ant in state.ants_of(1 - player):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance <= 6:
                pressure += max(0.0, 6.5 - distance) * (1.0 + ant.level * 0.4)
        return pressure

    def _enemy_profile(self, state: GameState, player: int, x: int, y: int, radius: int = 6) -> dict[str, float]:
        my_base = PLAYER_BASES[player]
        profile = {
            "swarm": 0.0,
            "brute": 0.0,
            "control_targets": 0.0,
            "progress": 0.0,
            "protected": 0.0,
            "randomized": 0.0,
        }
        for ant in state.ants_of(1 - player):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance > radius:
                continue
            weight = max(0.5, radius + 1 - distance)
            progress = max(0.0, 12.0 - hex_distance(ant.x, ant.y, *my_base))
            profile["swarm"] += weight * (1.25 if ant.level == 0 else 1.0)
            profile["brute"] += weight * (1.0 + ant.level * 0.9)
            if ant.behavior != AntBehavior.CONTROL_FREE:
                profile["control_targets"] += weight * (1.0 + progress * 0.08)
            else:
                profile["protected"] += weight * 1.2
            if ant.deflector or ant.shield > 0:
                profile["protected"] += weight
            if ant.behavior == AntBehavior.RANDOM:
                profile["randomized"] += weight
            profile["progress"] += weight * progress * 0.12
        return profile

    def _tower_type_fit(self, state: GameState, player: int, x: int, y: int, tower_type: TowerType) -> float:
        enemy_base = PLAYER_BASES[1 - player]
        forward_distance = hex_distance(x, y, *enemy_base)
        local_density = self._local_enemy_pressure(state, player, x, y)
        profile = self._enemy_profile(state, player, x, y)
        control = profile["control_targets"]
        swarm = profile["swarm"]
        brute = profile["brute"]
        progress = profile["progress"]
        protected = profile["protected"]
        randomized = profile["randomized"]

        if tower_type == TowerType.HEAVY:
            return brute * 0.95 + local_density * 0.45 - forward_distance * 0.08
        if tower_type == TowerType.HEAVY_PLUS:
            return brute * 1.2 + protected * 0.35 + local_density * 0.4 - forward_distance * 0.04
        if tower_type == TowerType.ICE:
            return control * 1.15 + progress * 0.9 + randomized * 0.35 - protected * 0.7
        if tower_type == TowerType.CANNON:
            return control * 1.3 + progress * 1.35 + brute * 0.4 - protected * 1.0
        if tower_type == TowerType.QUICK:
            return swarm * 0.95 + 3.0 + progress * 0.25
        if tower_type == TowerType.QUICK_PLUS:
            return swarm * 1.05 + 3.8 + progress * 0.25
        if tower_type == TowerType.DOUBLE:
            return swarm * 1.25 + control * 0.2 + 3.5
        if tower_type == TowerType.SNIPER:
            return max(0.0, 18 - forward_distance) + brute * 0.45 + protected * 0.45
        if tower_type == TowerType.MORTAR:
            return swarm * 1.0 + max(0.0, 12 - forward_distance) + progress * 0.35
        if tower_type == TowerType.MORTAR_PLUS:
            return swarm * 1.1 + max(0.0, 13 - forward_distance) + brute * 0.2
        if tower_type == TowerType.PULSE:
            return swarm * 1.4 + control * 0.75 + progress * 0.4 - protected * 0.9
        if tower_type == TowerType.MISSILE:
            return brute * 0.95 + swarm * 0.65 + protected * 0.5
        return local_density * 0.5

    def _storm_value(self, state: GameState, player: int, x: int, y: int) -> float:
        enemy = 1 - player
        total = 0.0
        for ant in state.ants_of(enemy):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance <= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].attack_range:
                total += ant.kill_reward + (4 - distance) * 0.5
        return total - SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost * 0.03

    def _emp_value(self, state: GameState, player: int, x: int, y: int) -> float:
        total = 0.0
        for tower in state.towers_of(1 - player):
            distance = hex_distance(x, y, tower.x, tower.y)
            if distance <= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].attack_range:
                total += 3.0 + tower.level * 2.5
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].attack_range + 2:
                total += 0.4 + ant.level * 0.5
        return total - SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost * 0.025

    def _tower_pressure_on_ant(self, state: GameState, player: int, ant, *, chip_only: bool) -> float:
        total = 0.0
        for tower in state.towers_of(1 - player):
            distance = hex_distance(ant.x, ant.y, tower.x, tower.y)
            effective_range = tower.attack_range
            if tower.tower_type in (TowerType.MORTAR, TowerType.MORTAR_PLUS):
                effective_range += 1
            elif tower.tower_type == TowerType.MISSILE:
                effective_range += 2
            if distance > effective_range:
                continue
            if chip_only and tower.damage * 2 >= ant.max_hp:
                continue
            pressure = min(tower.damage / max(float(ant.max_hp), 1.0), 2.5) + tower.level * 0.6
            if tower.tower_type in (TowerType.ICE, TowerType.CANNON, TowerType.PULSE):
                pressure += 0.8
            elif tower.tower_type in (TowerType.DOUBLE, TowerType.MORTAR, TowerType.MORTAR_PLUS, TowerType.MISSILE):
                pressure += 0.4
            if tower.cooldown_clock <= 1.0:
                pressure *= 1.15
            total += pressure / max(distance, 1)
        return total

    def _ant_objective_value(self, player: int, ant) -> float:
        enemy_base = PLAYER_BASES[1 - player]
        progress = max(0, 18 - hex_distance(ant.x, ant.y, *enemy_base))
        return 1.0 + ant.level * 1.1 + progress * 0.25

    def _deflector_value(self, state: GameState, player: int, x: int, y: int) -> float:
        total = 0.0
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].attack_range:
                if ant.deflector:
                    continue
                chip_pressure = self._tower_pressure_on_ant(state, player, ant, chip_only=True)
                if chip_pressure <= 0:
                    continue
                total += chip_pressure * (1.4 + ant.level * 0.3)
                total += self._ant_objective_value(player, ant) * 0.35
                if ant.behavior != AntBehavior.CONTROL_FREE:
                    total += 0.8
        return total - SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost * 0.02

    def _evasion_value(self, state: GameState, player: int, x: int, y: int) -> float:
        total = 0.0
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].attack_range:
                missing_layers = max(0, 2 - ant.shield)
                if missing_layers <= 0:
                    continue
                tower_pressure = self._tower_pressure_on_ant(state, player, ant, chip_only=False)
                if tower_pressure <= 0:
                    continue
                total += tower_pressure * missing_layers * (1.1 + ant.level * 0.25)
                total += self._ant_objective_value(player, ant) * 0.3
                if ant.behavior != AntBehavior.CONTROL_FREE and ant.shield == 0:
                    total += 0.6
        return total - SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost * 0.02
