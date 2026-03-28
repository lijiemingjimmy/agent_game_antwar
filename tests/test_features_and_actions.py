from __future__ import annotations

from SDK.actions import ActionCatalog
from SDK.constants import AntBehavior, TowerType
from SDK.engine import GameState
from SDK.features import FeatureExtractor
from SDK.model import Ant, Tower


def test_feature_summary_tracks_control_and_protection_state() -> None:
    state = GameState.initial(seed=21)
    my_ant = Ant(0, 0, 4, 9, hp=25, level=1, behavior=AntBehavior.CONTROL_FREE, shield=2, deflector=True)
    enemy_ant = Ant(1, 1, 14, 9, hp=10, level=0, behavior=AntBehavior.RANDOM, frozen=True)
    enemy_ant.refresh_status()
    state.ants.extend([my_ant, enemy_ant])

    summary = FeatureExtractor().summarize(state, 0).named

    assert summary["control_delta"] == 1.0
    assert summary["random_delta"] == 1.0
    assert summary["frozen_delta"] == 1.0
    assert summary["shield_delta"] == 2.0
    assert summary["protected_delta"] == 1.0
    assert summary["deflector_delta"] == 1.0


def test_emergency_evasion_value_rises_under_heavy_tower_pressure() -> None:
    state = GameState.initial(seed=22)
    ant = Ant(2, 0, 12, 9, hp=10, level=0, behavior=AntBehavior.DEFAULT)
    tower = Tower(3, 1, 14, 9, TowerType.HEAVY, cooldown_clock=0.0)
    state.ants.append(ant)
    state.towers.append(tower)

    catalog = ActionCatalog()
    pressured = catalog._evasion_value(state, 0, 12, 9)
    quiet = catalog._evasion_value(state, 0, 2, 9)

    assert pressured > quiet


def test_deflector_value_prefers_chip_damage_clusters() -> None:
    state = GameState.initial(seed=23)
    ant = Ant(4, 0, 12, 9, hp=25, level=1, behavior=AntBehavior.DEFAULT)
    tower = Tower(5, 1, 14, 9, TowerType.QUICK, cooldown_clock=0.0)
    state.ants.append(ant)
    state.towers.append(tower)

    catalog = ActionCatalog()
    pressured = catalog._deflector_value(state, 0, 12, 9)
    quiet = catalog._deflector_value(state, 0, 2, 9)

    assert pressured > quiet


def test_cannon_upgrade_is_preferred_against_advanced_controllable_intruders() -> None:
    state = GameState.initial(seed=24)
    state.coins[0] = 260
    state.towers.append(Tower(6, 0, 6, 9, TowerType.HEAVY, cooldown_clock=0.0))
    state.ants.append(Ant(7, 1, 8, 9, hp=25, level=1, behavior=AntBehavior.DEFAULT))
    state.ants.append(Ant(8, 1, 8, 10, hp=10, level=0, behavior=AntBehavior.RANDOM))

    catalog = ActionCatalog()
    candidates = {bundle.operations[0].arg1: bundle.score for bundle in catalog._upgrade_candidates(state, 0)}

    assert candidates[int(TowerType.CANNON)] > candidates[int(TowerType.HEAVY_PLUS)]
    assert candidates[int(TowerType.CANNON)] > candidates[int(TowerType.ICE)]


def test_build_score_remains_below_base_upgrade_after_two_towers() -> None:
    state = GameState.initial(seed=25)
    state.coins[0] = 260
    state.towers.append(Tower(9, 0, 6, 9, TowerType.BASIC, cooldown_clock=2.0))
    state.towers.append(Tower(10, 0, 5, 9, TowerType.BASIC, cooldown_clock=2.0))

    bundles = ActionCatalog().build(state, 0)

    assert bundles[0].tags[:2] == ("combo", "build-base")
    assert "+upgrade-ant" in bundles[0].name


def test_scripted_build_base_combo_is_generated() -> None:
    state = GameState.initial(seed=26)
    state.coins[0] = 220

    bundles = ActionCatalog().build(state, 0)
    combo_names = {bundle.name for bundle in bundles if bundle.tags[:2] == ("combo", "build-base")}

    assert any(name.startswith("build@") and "+upgrade-ant" in name for name in combo_names)


def test_generic_pairing_skips_duplicate_build_build_spam() -> None:
    state = GameState.initial(seed=27)
    state.coins[0] = 220
    state.towers.append(Tower(11, 0, 6, 9, TowerType.HEAVY, cooldown_clock=2.0))

    bundles = ActionCatalog().build(state, 0)[:20]

    assert all(
        not (
            bundle.tags
            and bundle.tags[0] == "combo"
            and len(bundle.operations) == 2
            and all(op.op_type.value == 11 for op in bundle.operations)
        )
        for bundle in bundles
    )


def test_generic_pairing_does_not_duplicate_scripted_build_base_in_reverse_order() -> None:
    state = GameState.initial(seed=28)
    state.coins[0] = 220

    bundles = ActionCatalog().build(state, 0)[:20]

    assert all(
        not (
            bundle.tags[:2] == ("combo", "generic")
            and bundle.name.startswith("upgrade-ant+build@")
        )
        for bundle in bundles
    )


def test_opening_phase_keeps_some_single_actions_near_front() -> None:
    state = GameState.initial(seed=29)
    state.coins[0] = 220

    bundles = ActionCatalog().build(state, 0)[:12]
    tag_heads = [bundle.tags[0] for bundle in bundles if bundle.tags]

    assert "base" in tag_heads
    assert "build" in tag_heads


def test_late_phase_limits_frontloaded_build_spam() -> None:
    state = GameState.initial(seed=30)
    state.round_index = 220
    state.coins[0] = 260
    state.towers.append(Tower(12, 0, 6, 9, TowerType.CANNON, cooldown_clock=0.0))
    state.towers.append(Tower(13, 0, 5, 9, TowerType.PULSE, cooldown_clock=0.0))
    state.towers.append(Tower(14, 0, 6, 11, TowerType.HEAVY_PLUS, cooldown_clock=0.0))
    state.towers.append(Tower(15, 0, 5, 11, TowerType.DOUBLE, cooldown_clock=0.0))

    bundles = ActionCatalog().build(state, 0)[:12]
    build_count = sum(1 for bundle in bundles if bundle.tags and bundle.tags[0] == "build")

    assert build_count <= 2


def test_early_basic_towers_are_not_eagerly_sold_for_churn() -> None:
    state = GameState.initial(seed=31)
    state.round_index = 40
    state.coins[0] = 260
    state.towers.append(Tower(16, 0, 6, 9, TowerType.BASIC, cooldown_clock=0.0))
    state.towers.append(Tower(17, 0, 5, 9, TowerType.BASIC, cooldown_clock=0.0))

    bundles = ActionCatalog()._downgrade_candidates(state, 0)

    assert bundles == []


def test_quick_branch_saturation_pushes_new_upgrade_toward_other_branches() -> None:
    state = GameState.initial(seed=32)
    state.coins[0] = 260
    state.towers.append(Tower(18, 0, 6, 9, TowerType.QUICK_PLUS, cooldown_clock=0.0))
    state.towers.append(Tower(19, 0, 5, 9, TowerType.QUICK_PLUS, cooldown_clock=0.0))
    state.towers.append(Tower(20, 0, 8, 7, TowerType.BASIC, cooldown_clock=0.0))
    state.ants.append(Ant(21, 1, 9, 8, hp=25, level=1, behavior=AntBehavior.DEFAULT))
    state.ants.append(Ant(22, 1, 10, 8, hp=10, level=0, behavior=AntBehavior.DEFAULT))

    candidates = {
        bundle.operations[0].arg1: bundle.score
        for bundle in ActionCatalog()._upgrade_candidates(state, 0)
        if bundle.operations[0].arg0 == 20
    }

    assert candidates[int(TowerType.HEAVY)] > candidates[int(TowerType.QUICK)]
