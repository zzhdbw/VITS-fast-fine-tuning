#!/usr/bin/env python3
"""Step 3: generate the final 16 kHz VITS config.json.

Inputs:
    <data-dir>/speakers.json
    <data-dir>/train.txt
    <data-dir>/val.txt
    config_16k_template.json

Output:
    <data-dir>/config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import RECIPE_DIR, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create config.json for the Genshin zh-en 16 kHz recipe.")
    parser.add_argument("--data-dir", type=Path, default=RECIPE_DIR / "data" / "genshin_zh_en_16k")
    parser.add_argument("--template", type=Path, default=RECIPE_DIR / "config_16k_template.json")
    parser.add_argument("--output", type=Path, default=None, help="Default: <data-dir>/config.json")
    return parser.parse_args()


def count_non_empty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else args.data_dir / "config.json"

    speakers_path = args.data_dir / "speakers.json"
    train_path = args.data_dir / "train.txt"
    val_path = args.data_dir / "val.txt"
    for path in (speakers_path, train_path, val_path, args.template):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    speakers = read_json(speakers_path)
    if not speakers:
        raise RuntimeError(f"No speakers found in {speakers_path}")

    train_lines = count_non_empty_lines(train_path)
    val_lines = count_non_empty_lines(val_path)
    if train_lines == 0 or val_lines == 0:
        raise RuntimeError(
            f"train.txt/val.txt must be non-empty (train={train_lines}, val={val_lines})."
        )

    n_speakers = max(int(v) for v in speakers.values()) + 1
    config = read_json(args.template)
    config["data"]["training_files"] = str(train_path.resolve())
    config["data"]["validation_files"] = str(val_path.resolve())
    config["data"]["n_speakers"] = n_speakers
    config["speakers"] = {str(k): int(v) for k, v in sorted(speakers.items(), key=lambda kv: int(kv[1]))}

    if int(config["data"]["sampling_rate"]) != 16000:
        raise RuntimeError(
            f"Template sampling_rate must be 16000, got {config['data']['sampling_rate']}"
        )

    write_json(output, config)
    print(
        f"[config] wrote {output} (speakers={n_speakers}, train={train_lines}, "
        f"val={val_lines}, sr={config['data']['sampling_rate']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
