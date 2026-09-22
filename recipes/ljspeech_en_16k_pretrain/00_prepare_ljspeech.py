#!/usr/bin/env python3
"""Prepare LJSpeech-1.1 for single-speaker VITS pretraining.

This recipe intentionally reuses the repository's original English-capable
frontend instead of adding a new cleaner or a new vocabulary:

* cleaner: ``cjke_cleaners2``
* vocabulary: ``text.symbols.symbols`` (the active original 68-symbol table)

LJSpeech text is wrapped as ``[EN]... [EN]`` before cleaning, so
``cjke_cleaners2`` converts English text to IPA.  The resulting symbols are
written into ``config.json``; generated manifests therefore use exactly the
original repository vocabulary.

Expected LJSpeech-1.1 layout::

    <dataset-root>/
    ├── metadata.csv
    └── wavs/
        ├── LJ001-0001.wav
        └── ...

``metadata.csv`` is the official ``id|transcription|normalized`` file.  This
script prefers the normalized column, resamples audio to 16 kHz, splits
train/val, and writes ``train.txt`` / ``val.txt`` / ``config.json``.
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
from text.symbols import symbols as ORIGINAL_SYMBOLS  # noqa: E402


CLEANER_NAMES = ["cjke_cleaners2"]
SYMBOL_SET = set(ORIGINAL_SYMBOLS)
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LJSpeech-1.1 for single-speaker VITS pretraining.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/afs/datasets/TTS/LJSpeech-1.1"),
        help="LJSpeech-1.1 root containing metadata.csv and wavs/.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Default: <dataset-root>/metadata.csv.",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=None,
        help="Default: <dataset-root>/wavs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RECIPE_DIR / "data" / "ljspeech_en_16k",
        help="Output data directory.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=RECIPE_DIR / "config_ljspeech_16k_template.json",
        help="Config template to fill.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Default: <data-dir>/config.json.",
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=11.0)
    parser.add_argument(
        "--max-cleaned-length",
        type=int,
        default=190,
        help="Discard rows whose cleaned symbol sequence is longer than this.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only process the first N usable metadata rows. 0 means all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel workers used for cleaning and resampling.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse already resampled wav files in <data-dir>/wav.",
    )
    parser.add_argument(
        "--speaker-name",
        type=str,
        default="LJSpeech",
        help="Speaker name written into config.json.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Train epochs written to config.json.")
    return parser.parse_args()


def parse_metadata(metadata_path: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return ``[(utt_id, raw_text), ...]`` and non-fatal parse errors.

    The official file is pipe-delimited.  A few rows have malformed quoting
    that ``csv.reader`` parses incorrectly, so we split on the literal ``|``
    separators directly; the file is known to have exactly two per line.
    """
    rows: List[Tuple[str, str]] = []
    errors: List[str] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 3:
                errors.append(f"line {line_no}: expected 3 fields, got {len(parts)}")
                continue
            utt_id = parts[0].strip()
            transcription = parts[1].strip()
            normalized = parts[2].strip()
            text = normalized or transcription
            if not utt_id:
                errors.append(f"line {line_no}: empty utterance id")
                continue
            if not text:
                errors.append(f"line {line_no} ({utt_id}): empty text")
                continue
            rows.append((utt_id, text))
    return rows, errors


def clean_english_text(raw_text: str, max_length: int) -> Tuple[str, str]:
    """Clean English with the original ``cjke_cleaners2``.

    Returns ``(cleaned_text, error)``.  The text is tagged with ``[EN]`` so
    ``cjke_cleaners2`` invokes its existing ``english_to_ipa2`` path.  Any
    characters not present in the original ``text.symbols.symbols`` table are
    dropped, matching ``text.cleaned_text_to_sequence`` behavior at training
    time and ``text.text_to_sequence`` behavior at inference time.
    """
    tagged = f"[EN]{raw_text}[EN]"
    try:
        cleaned = _clean_text(tagged, CLEANER_NAMES)
    except Exception as exc:  # noqa: BLE001 - report the row and keep going
        return "", f"{type(exc).__name__}: {exc}"

    cleaned = "".join(ch for ch in cleaned if ch in SYMBOL_SET)
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return "", "empty cleaned text"
    if len(cleaned) > max_length:
        return "", f"cleaned text too long ({len(cleaned)} > {max_length})"
    return cleaned, ""


def process_item(
    item: Dict[str, Any],
    data_dir: str,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    max_cleaned_length: int,
    skip_existing: bool,
) -> Dict[str, Any]:
    """Clean text and resample one wav.  Runs in a worker process."""
    row = dict(item)

    cleaned, error = clean_english_text(row["raw_text"], max_cleaned_length)
    if error:
        row.update(status="error", error=error)
        return row
    row["text"] = cleaned

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
            if int(dst_info.samplerate) != int(target_sr):
                row.update(status="error", error=f"existing wav has wrong sr: {dst_info.samplerate}")
                return row
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
    max_cleaned_length: int,
    skip_existing: bool,
    workers: int,
) -> List[Dict[str, Any]]:
    worker = partial(
        process_item,
        data_dir=str(data_dir),
        target_sr=target_sr,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cleaned_length=max_cleaned_length,
        skip_existing=skip_existing,
    )
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(
                tqdm(
                    executor.map(worker, items, chunksize=1),
                    total=len(items),
                    desc="[clean+resample]",
                    unit="file",
                    dynamic_ncols=True,
                )
            )
    except (PermissionError, OSError) as exc:
        print(f"[prepare] multiprocessing unavailable ({exc}); falling back to serial.", flush=True)
        return [worker(item) for item in tqdm(items, desc="[clean+resample]", unit="file", dynamic_ncols=True)]


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['dst_wav']}|0|{row['text']}\n")


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.template = args.template.expanduser().resolve()

    metadata_path = (args.metadata or (args.dataset_root / "metadata.csv")).expanduser().resolve()
    wav_dir = (args.wav_dir or (args.dataset_root / "wavs")).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"wav directory not found: {wav_dir}")

    metadata_rows, parse_errors = parse_metadata(metadata_path)
    print(f"[prepare] metadata: {metadata_path}")
    print(f"[prepare] metadata rows: {len(metadata_rows)} (parse errors: {len(parse_errors)})")

    items: List[Dict[str, Any]] = []
    for utt_id, raw_text in metadata_rows:
        src_wav = wav_dir / f"{utt_id}.wav"
        if not src_wav.exists():
            parse_errors.append(f"{utt_id}: missing {src_wav}")
            continue
        items.append(
            {
                "uid": utt_id,
                "src_wav": str(src_wav),
                "raw_text": raw_text,
            }
        )

    if args.max_files > 0:
        items = items[: args.max_files]
    if not items:
        raise RuntimeError("No usable items after parsing.")

    print(f"[prepare] usable rows: {len(items)} (parse errors: {len(parse_errors)})")
    print(f"[prepare] cleaner: {CLEANER_NAMES}, original symbols: {len(ORIGINAL_SYMBOLS)}")

    results = collect_results(
        items,
        data_dir=args.data_dir,
        target_sr=args.sampling_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_cleaned_length=args.max_cleaned_length,
        skip_existing=args.skip_existing,
        workers=args.workers,
    )

    good = [row for row in results if row.get("status") == "ok"]
    failed = [row for row in results if row.get("status") != "ok"]
    if not good:
        raise RuntimeError("All rows failed to clean or resample.")

    rng = random.Random(args.seed)
    rng.shuffle(good)
    if len(good) > 1:
        val_count = min(len(good) - 1, max(1, int(round(len(good) * args.val_ratio))))
    else:
        val_count = 1
    val_rows = good[:val_count]
    train_rows = good[val_count:]
    if not train_rows:
        train_rows, val_rows = val_rows, []

    train_txt = args.data_dir / "train.txt"
    val_txt = args.data_dir / "val.txt"
    write_manifest(train_txt, train_rows)
    write_manifest(val_txt, val_rows if val_rows else train_rows[:1])

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
    config["data"]["mel_fmax"] = float(args.sampling_rate) / 2.0
    config["data"]["text_cleaners"] = CLEANER_NAMES
    config["data"]["cleaned_text"] = True
    config["data"]["max_text_len"] = int(args.max_cleaned_length)
    config["train"]["epochs"] = int(args.epochs)
    # Always take the vocabulary from the repository's original active symbol
    # table.  No new symbol is introduced by this recipe.
    config["symbols"] = list(ORIGINAL_SYMBOLS)
    config["speakers"] = {args.speaker_name: 0}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "dataset_root": str(args.dataset_root),
        "metadata": str(metadata_path),
        "wav_dir": str(wav_dir),
        "data_dir": str(args.data_dir),
        "sampling_rate": int(args.sampling_rate),
        "text_cleaners": CLEANER_NAMES,
        "symbols_count": len(ORIGINAL_SYMBOLS),
        "max_cleaned_length": int(args.max_cleaned_length),
        "metadata_rows": len(metadata_rows),
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
