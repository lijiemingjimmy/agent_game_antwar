from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


OP_NAMES = {
    11: "build",
    12: "upgrade",
    13: "downgrade",
    21: "storm",
    22: "emp",
    23: "deflector",
    24: "evasion",
    31: "upgrade-gen",
    32: "upgrade-ant",
}


def summarize_player(replay: list[dict], key: str) -> dict[str, object]:
    counts: Counter[str] = Counter()
    build_positions: list[tuple[int, int, int]] = []
    upgrade_targets: list[tuple[int, int, int]] = []
    super_targets: list[tuple[int, str, int, int]] = []

    for round_index, entry in enumerate(replay):
        for op in entry.get(key, []):
            name = OP_NAMES.get(op["type"], f"op-{op['type']}")
            counts[name] += 1
            if op["type"] == 11:
                build_positions.append((round_index, op["pos"]["x"], op["pos"]["y"]))
            elif op["type"] == 12:
                upgrade_targets.append((round_index, op["id"], op["args"]))
            elif op["type"] >= 21:
                super_targets.append((round_index, name, op["pos"]["x"], op["pos"]["y"]))

    return {
        "counts": dict(sorted(counts.items())),
        "build_positions": build_positions,
        "upgrade_targets": upgrade_targets,
        "super_targets": super_targets,
    }


def summarize_replay(path: Path) -> str:
    replay = json.loads(path.read_text())
    final_state = replay[-1]["round_state"]
    parts = [
        f"Replay: {path.name}",
        f"  rounds: {len(replay)}",
        f"  winner: {final_state.get('winner')}",
        f"  camps: {final_state.get('camps')}",
    ]
    for key in ("op0", "op1"):
        summary = summarize_player(replay, key)
        parts.append(f"  {key}: {summary['counts']}")
        parts.append(f"    builds: {summary['build_positions']}")
        parts.append(f"    upgrades: {summary['upgrade_targets']}")
        parts.append(f"    supers: {summary['super_targets']}")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Saiblo Antwar replay JSON files.")
    parser.add_argument("paths", nargs="*", help="Replay files or directories. Defaults to fighting_examples.")
    args = parser.parse_args()

    if args.paths:
        inputs = [Path(item) for item in args.paths]
    else:
        inputs = [Path("fighting_examples")]

    replay_files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            replay_files.extend(sorted(item.glob("*.json")))
        elif item.suffix == ".json":
            replay_files.append(item)

    if not replay_files:
        raise SystemExit("no replay json files found")

    for index, replay_path in enumerate(replay_files):
        if index:
            print()
        print(summarize_replay(replay_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
