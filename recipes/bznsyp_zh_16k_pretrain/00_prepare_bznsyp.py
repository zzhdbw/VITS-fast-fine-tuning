#!/usr/bin/env python3
"""Prepare BZNSYP for single-speaker VITS pretraining.

BZNSYP layout:
    <dataset-root>/
    ├── Wave/
    │   └── 000001.wav
    ├── ProsodyLabeling/
    │   └── 000001-010000.txt
    └── PhoneLabeling/
        └── ...

This script:
    1. parses ``ProsodyLabeling/*.txt``
    2. removes prosody marks such as ``#1`` / ``#4``
    3. cleans Chinese text with ``chinese_cleaners`` (Bopomofo / Zhuyin)
    4. resamples all wav files to the target sample rate
    5. writes ``train.txt`` / ``val.txt`` / ``config.json``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text import _clean_text  # noqa: E402


PROSODY_RE = re.compile(r"#[0-9]+")
SPACE_RE = re.compile(r"\s+")

# BZNSYP is pure Mandarin Chinese, so use the repository's Chinese cleaner and
# a compact Bopomofo/Zhuyin symbol table instead of the multilingual IPA table.
CLEANER_NAMES = ["chinese_cleaners"]
CHINESE_PUNCTUATION = "，。！？—…"
CHINESE_BOPOMOFO = (
    "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ"
    "ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ"
    "ˉˊˇˋ˙ "
)
CHINESE_SYMBOLS = ["_"] + list(CHINESE_PUNCTUATION) + list(CHINESE_BOPOMOFO)
CHINESE_SYMBOL_SET = set(CHINESE_SYMBOLS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BZNSYP for single-speaker VITS pretraining.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/afs/datasets/TTS/BZNSYP"),
        help="BZNSYP root which contains Wave/ and ProsodyLabeling/.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RECIPE_DIR / "data" / "bznsyp_zh_16k",
        help="Output data directory.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=RECIPE_DIR / "config_bznsyp_16k_template.json",
        help="Config template to fill.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Default: <data-dir>/config.json",
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--max-cleaned-length", type=int, default=200)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only process the first N wav files. 0 means all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel workers used for resampling.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse already resampled wav files in <data-dir>/wav.",
    )
    parser.add_argument(
        "--speaker-name",
        type=str,
        default="BZNSYP",
        help="Speaker name written into config.json.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Train epochs written to config.json.")
    return parser.parse_args()


def find_label_file(dataset_root: Path) -> Path:
    preferred = dataset_root / "ProsodyLabeling" / "000001-010000.txt"
    if preferred.exists():
        return preferred
    candidates = sorted((dataset_root / "ProsodyLabeling").glob("*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No label txt found under {dataset_root / 'ProsodyLabeling'}")
    return candidates[0]


def parse_labels(label_file: Path) -> Dict[str, str]:
    """Return ``{utt_id: chinese_text}`` from the alternating BZNSYP label file."""
    lines = label_file.read_text(encoding="utf-8").splitlines()
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected even number of lines in {label_file}, got {len(lines)}")

    labels: Dict[str, str] = {}
    for i in range(0, len(lines), 2):
        first = lines[i].strip()
        pinyin = lines[i + 1].strip()
        if not first or not pinyin:
            continue
        parts = first.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"Bad label line {i + 1} in {label_file}: {first!r}")
        utt_id, text = parts[0].strip(), parts[1].strip()
        if not utt_id:
            raise ValueError(f"Empty utterance id at line {i + 1} in {label_file}")
        labels[utt_id] = text
    return labels


def clean_chinese(raw_text: str, max_length: int) -> Tuple[str, str]:
    """Return ``(cleaned_text, error)``. error is empty on success."""
    text = PROSODY_RE.sub("", raw_text)
    text = text.replace("\u3000", " ").strip()
    if not text:
        return "", "empty raw text"

    try:
        cleaned = _clean_text(text, CLEANER_NAMES)
    except Exception as exc:  # noqa: BLE001 - report the row and keep going
        return "", f"{type(exc).__name__}: {exc}"

    cleaned = cleaned.replace("\u3000", " ").strip()
    # Keep only the Chinese Bopomofo symbol table.
    cleaned = "".join(ch for ch in cleaned if ch in CHINESE_SYMBOL_SET)
    cleaned = SPACE_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return "", "empty cleaned text"
    if len(cleaned) > max_length:
        return "", f"cleaned text too long ({len(cleaned)} > {max_length})"
    return cleaned, ""


def resample_audio(
    item: Dict[str, Any],
    data_dir: str,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    skip_existing: bool,
) -> Dict[str, Any]:
    """Resample one wav. Runs in a worker process."""
    row = dict(item)
    src = Path(row["src_wav"])
    dst = Path(data_dir) / "wav" / f"{row['uid']}.wav"

    try:
        info = sf.info(str(src))
        duration = float(info.frames) / float(info.samplerate)
        if duration < min_duration or duration > max_duration:
            row.update(
                status="error",
                error=f"duration out of range: {duration:.3f}s",
                duration=duration,
            )
            return row

        if skip_existing and dst.exists():
            dst_info = sf.info(str(dst))
            row.update(
                status="ok",
                dst_wav=str(dst),
                duration=float(dst_info.frames) / float(dst_info.samplerate),
                samplerate=int(dst_info.samplerate),
            )
            return row

        audio, sample_rate = sf.read(str(src), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if int(sample_rate) != int(target_sr):
            gcd = math.gcd(int(sample_rate), int(target_sr))
            up = int(target_sr) // gcd
            down = int(sample_rate) // gcd
            audio = resample_poly(audio, up, down)

        audio = np.asarray(audio, dtype=np.float32)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), audio, int(target_sr), subtype="PCM_16")

        row.update(
            status="ok",
            dst_wav=str(dst),
            duration=float(len(audio)) / float(target_sr),
            samplerate=int(target_sr),
        )
        return row
    except Exception as exc:  # noqa: BLE001 - keep the long job running
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return row


def collect_results(
    items: Sequence[Dict[str, Any]],
    *,
    data_dir: Path,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    skip_existing: bool,
    workers: int,
) -> List[Dict[str, Any]]:
    worker = partial(
        resample_audio,
        data_dir=str(data_dir),
        target_sr=target_sr,
        min_duration=min_duration,
        max_duration=max_duration,
        skip_existing=skip_existing,
    )

    if workers <= 1:
        return [worker(item) for item in tqdm(items, desc="[resample]", unit="file", dynamic_ncols=True)]

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(
                tqdm(
                    executor.map(worker, items, chunksize=1),
                    total=len(items),
                    desc="[resample]",
                    unit="file",
                    dynamic_ncols=True,
                )
            )
    except (PermissionError, OSError) as exc:
        print(f"[prepare] multiprocessing unavailable ({exc}); falling back to serial.", flush=True)
        return [worker(item) for item in tqdm(items, desc="[resample]", unit="file", dynamic_ncols=True)]


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['dst_wav']}|0|{row['text']}\n")


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.template = args.template.expanduser().resolve()

    wave_dir = args.dataset_root / "Wave"
    if not wave_dir.is_dir():
        raise FileNotFoundError(f"Wave directory not found: {wave_dir}")

    label_file = find_label_file(args.dataset_root)
    labels = parse_labels(label_file)
    print(f"[prepare] label file: {label_file}")
    print(f"[prepare] labels: {len(labels)}")

    items: List[Dict[str, Any]] = []
    parse_errors: List[str] = []
    for utt_id, raw_text in sorted(labels.items()):
        src_wav = wave_dir / f"{utt_id}.wav"
        if not src_wav.exists():
            parse_errors.append(f"{utt_id}: missing {src_wav}")
            continue
        cleaned, error = clean_chinese(raw_text, args.max_cleaned_length)
        if error:
            parse_errors.append(f"{utt_id}: {error}")
            continue
        items.append(
            {
                "uid": utt_id,
                "src_wav": str(src_wav),
                "raw_text": raw_text,
                "text": cleaned,
            }
        )

    if args.max_files > 0:
        items = items[: args.max_files]
    if not items:
        raise RuntimeError("No usable items after parsing and text cleaning.")

    print(f"[prepare] usable rows: {len(items)} (parse errors: {len(parse_errors)})")

    results = collect_results(
        items,
        data_dir=args.data_dir,
        target_sr=args.sampling_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        skip_existing=args.skip_existing,
        workers=args.workers,
    )

    good = [row for row in results if row.get("status") == "ok"]
    failed = [row for row in results if row.get("status") != "ok"]
    if not good:
        raise RuntimeError("All wav files failed to resample.")

    rng = random.Random(args.seed)
    rng.shuffle(good)
    val_count = min(len(good) - 1, max(1, int(round(len(good) * args.val_ratio)))) if len(good) > 1 else 1
    val_rows = good[:val_count]
    train_rows = good[val_count:]
    if not train_rows:
        train_rows, val_rows = val_rows, []  # single-file smoke test fallback

    train_txt = args.data_dir / "train.txt"
    val_txt = args.data_dir / "val.txt"
    write_manifest(train_txt, train_rows)
    if val_rows:
        write_manifest(val_txt, val_rows)
    else:
        write_manifest(val_txt, train_rows[:1])

    config_path = (
        args.output_config.expanduser().resolve()
        if args.output_config
        else args.data_dir / "config.json"
    )
    config = json.loads(args.template.read_text(encoding="utf-8"))
    config["data"]["training_files"] = str(train_txt.resolve())
    config["data"]["validation_files"] = str(val_txt.resolve())
    config["data"]["sampling_rate"] = int(args.sampling_rate)
    config["data"]["n_speakers"] = 1
    config["data"]["mel_fmax"] = 8000.0 if int(args.sampling_rate) == 16000 else None
    config["data"]["text_cleaners"] = CLEANER_NAMES
    config["train"]["epochs"] = int(args.epochs)
    config["symbols"] = CHINESE_SYMBOLS
    config["speakers"] = {args.speaker_name: 0}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "dataset_root": str(args.dataset_root),
        "label_file": str(label_file),
        "data_dir": str(args.data_dir),
        "sampling_rate": int(args.sampling_rate),
        "label_count": len(labels),
        "usable_items": len(items),
        "prepared_rows": len(good),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "failed_rows": len(failed),
        "parse_errors": parse_errors[:100],
        "failed_examples": failed[:20],
        "config": str(config_path),
    }
    summary_path = args.data_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[prepare] train={len(train_rows)}, val={len(val_rows)}, "
        f"failed={len(failed)} -> {args.data_dir}",
        flush=True,
    )
    print(f"[prepare] config -> {config_path}", flush=True)


if __name__ == "__main__":
    main()
