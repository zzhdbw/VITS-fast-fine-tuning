#!/usr/bin/env python3
"""Prepare LJSpeech-1.1 + BZNSYP for bilingual 16 kHz VITS pretraining.

This recipe trains a two-speaker, Chinese/English 16 kHz VITS model from
scratch.  Both datasets share the repository's original multilingual frontend:

* cleaner: ``cjke_cleaners2``
* vocabulary: ``text.symbols.symbols`` (the active 68-symbol IPA table)

LJSpeech text is tagged as ``[EN]... [EN]`` and BZNSYP text is tagged as
``[ZH]... [ZH]``.  ``cjke_cleaners2`` then routes English through
``english_to_ipa2`` and Chinese through ``chinese_to_ipa``.  The two outputs
therefore share exactly the same symbol table, so a single model can be trained
on both datasets.

Speaker mapping written to ``config.json``::

    LJSpeech -> 0
    BZNSYP   -> 1

Output layout::

    <data-dir>/
    ├── wav/
    │   ├── ljspeech/<utt_id>.wav
    │   └── bznsyp/<utt_id>.wav
    ├── train.txt
    ├── val.txt
    ├── config.json
    └── prepare_summary.json

``train.txt`` / ``val.txt`` are ``path|speaker_id|cleaned_text`` manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
PROSODY_RE = re.compile(r"#[0-9]+")
SPACE_RE = re.compile(r"\s+")

# dataset key -> (speaker name, contiguous speaker id, language tag)
DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "ljspeech": {"speaker": "LJSpeech", "speaker_id": 0, "lang": "en"},
    "bznsyp": {"speaker": "BZNSYP", "speaker_id": 1, "lang": "zh"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LJSpeech-1.1 + BZNSYP for bilingual 16 kHz VITS pretraining."
    )
    parser.add_argument(
        "--ljspeech-root",
        type=Path,
        default=Path("/mnt/afs/datasets/TTS/LJSpeech-1.1"),
        help="LJSpeech-1.1 root containing metadata.csv and wavs/.",
    )
    parser.add_argument(
        "--ljspeech-metadata",
        type=Path,
        default=None,
        help="Default: <ljspeech-root>/metadata.csv.",
    )
    parser.add_argument(
        "--ljspeech-wav-dir",
        type=Path,
        default=None,
        help="Default: <ljspeech-root>/wavs.",
    )
    parser.add_argument(
        "--bznsyp-root",
        type=Path,
        default=Path("/mnt/afs/datasets/TTS/BZNSYP"),
        help="BZNSYP root containing Wave/ and ProsodyLabeling/.",
    )
    parser.add_argument(
        "--bznsyp-label",
        type=Path,
        default=None,
        help="Default: <bznsyp-root>/ProsodyLabeling/000001-010000.txt.",
    )
    parser.add_argument(
        "--bznsyp-wave-dir",
        type=Path,
        default=None,
        help="Default: <bznsyp-root>/Wave.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RECIPE_DIR / "data" / "ljspeech_bznsyp_zh_en_16k",
        help="Output data directory.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=RECIPE_DIR / "config_ljspeech_bznsyp_16k_template.json",
        help="Config template to fill.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Default: <data-dir>/config.json.",
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation ratio per dataset.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=16.0,
        help="Longest kept utterance. 16 s matches the current VITS bucket boundary at 16 kHz.",
    )
    parser.add_argument(
        "--max-cleaned-length",
        type=int,
        default=190,
        help="Discard rows whose cleaned symbol sequence is longer than this.",
    )
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=0,
        help="Only process the first N usable rows from each dataset (0 = all).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Alias for --max-files-per-dataset; useful for compatibility with other recipes.",
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
    parser.add_argument("--epochs", type=int, default=1000, help="Train epochs written to config.json.")
    args = parser.parse_args()

    if args.max_files is not None:
        args.max_files_per_dataset = args.max_files
    return args


def parse_ljspeech_metadata(metadata_path: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return ``[(utt_id, raw_text), ...]`` and non-fatal parse errors."""
    rows: List[Tuple[str, str]] = []
    errors: List[str] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 3:
                errors.append(f"LJSpeech line {line_no}: expected 3 fields, got {len(parts)}")
                continue
            utt_id = parts[0].strip()
            transcription = parts[1].strip()
            normalized = parts[2].strip()
            text = normalized or transcription
            if not utt_id:
                errors.append(f"LJSpeech line {line_no}: empty utterance id")
                continue
            if not text:
                errors.append(f"LJSpeech line {line_no} ({utt_id}): empty text")
                continue
            rows.append((utt_id, text))
    return rows, errors


def find_bznsyp_label(dataset_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        label = explicit.expanduser().resolve()
        if not label.is_file():
            raise FileNotFoundError(f"BZNSYP label file not found: {label}")
        return label

    preferred = dataset_root / "ProsodyLabeling" / "000001-010000.txt"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted((dataset_root / "ProsodyLabeling").glob("*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"No BZNSYP label txt found under {dataset_root / 'ProsodyLabeling'}"
        )
    return candidates[0].resolve()


def parse_bznsyp_labels(label_file: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Parse the alternating BZNSYP label file and remove ``#1``-style marks."""
    lines = label_file.read_text(encoding="utf-8").splitlines()
    rows: List[Tuple[str, str]] = []
    errors: List[str] = []
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected an even number of lines in {label_file}, got {len(lines)}")

    for i in range(0, len(lines), 2):
        first = lines[i].strip()
        pinyin = lines[i + 1].strip()
        line_no = i + 1
        if not first or not pinyin:
            continue
        parts = first.split("\t", 1)
        if len(parts) != 2:
            errors.append(f"BZNSYP line {line_no}: bad label format: {first!r}")
            continue
        utt_id, raw_text = parts[0].strip(), parts[1].strip()
        if not utt_id:
            errors.append(f"BZNSYP line {line_no}: empty utterance id")
            continue
        raw_text = PROSODY_RE.sub("", raw_text).strip()
        if not raw_text:
            errors.append(f"BZNSYP line {line_no} ({utt_id}): empty text after removing prosody marks")
            continue
        rows.append((utt_id, raw_text))
    return rows, errors


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("|", " ").replace("\u3000", " ")).strip()


def clean_dual_text(raw_text: str, lang: str, max_length: int) -> Tuple[str, str]:
    """Clean English/Chinese with ``cjke_cleaners2`` and the original symbols.

    Returns ``(cleaned_text, error)``.  Unknown symbols are dropped, matching
    ``cleaned_text_to_sequence`` at training time and ``text_to_sequence`` at
    inference time.
    """
    raw_text = normalize_text(raw_text)
    if not raw_text:
        return "", "empty raw text"

    if lang == "zh":
        tag = "[ZH]"
    elif lang == "en":
        tag = "[EN]"
    else:
        return "", f"unknown language: {lang}"

    tagged = f"{tag}{raw_text}{tag}"
    try:
        cleaned = _clean_text(tagged, CLEANER_NAMES)
    except Exception as exc:  # noqa: BLE001 - report the row and keep going
        return "", f"{type(exc).__name__}: {exc}"

    cleaned = normalize_text(cleaned)
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

    cleaned, error = clean_dual_text(row["raw_text"], row["lang"], max_cleaned_length)
    if error:
        row.update(status="error", error=error)
        return row
    row["cleaned_text"] = cleaned

    src = Path(row["src_wav"])
    dst = Path(data_dir) / "wav" / row["dataset"] / f"{row['uid']}.wav"

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
                row.update(
                    status="error",
                    error=f"existing wav has wrong sample rate: {dst_info.samplerate}",
                )
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

        audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".tmp.wav")
        sf.write(str(tmp), audio, int(target_sr), subtype="PCM_16")
        os.replace(tmp, dst)

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
    workers = max(1, int(workers))
    if workers == 1:
        return [
            worker(item)
            for item in tqdm(items, desc="[clean+resample]", unit="file", dynamic_ncols=True)
        ]

    def run_executor(executor) -> List[Dict[str, Any]]:
        with executor:
            return list(
                tqdm(
                    executor.map(worker, items, chunksize=1),
                    total=len(items),
                    desc="[clean+resample]",
                    unit="file",
                    dynamic_ncols=True,
                )
            )

    try:
        return run_executor(ProcessPoolExecutor(max_workers=workers))
    except (PermissionError, OSError) as exc:
        print(
            f"[prepare] process workers unavailable ({exc}); falling back to threads.",
            flush=True,
        )

    try:
        return run_executor(ThreadPoolExecutor(max_workers=workers))
    except (PermissionError, OSError) as exc:
        print(
            f"[prepare] thread workers unavailable ({exc}); falling back to serial.",
            flush=True,
        )
        return [
            worker(item)
            for item in tqdm(items, desc="[clean+resample]", unit="file", dynamic_ncols=True)
        ]


def build_items(
    *,
    ljspeech_metadata: Path,
    ljspeech_wav_dir: Path,
    bznsyp_label: Path,
    bznsyp_wave_dir: Path,
    max_files_per_dataset: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    items: List[Dict[str, Any]] = []
    errors: List[str] = []

    lj_rows, lj_errors = parse_ljspeech_metadata(ljspeech_metadata)
    errors.extend(lj_errors)
    lj_items: List[Dict[str, Any]] = []
    for utt_id, raw_text in lj_rows:
        src_wav = ljspeech_wav_dir / f"{utt_id}.wav"
        if not src_wav.exists():
            errors.append(f"LJSpeech {utt_id}: missing {src_wav}")
            continue
        spec = DATASET_SPECS["ljspeech"]
        lj_items.append(
            {
                "uid": utt_id,
                "dataset": "ljspeech",
                "lang": spec["lang"],
                "speaker": spec["speaker"],
                "speaker_id": int(spec["speaker_id"]),
                "src_wav": str(src_wav.resolve()),
                "raw_text": raw_text,
            }
        )
    if max_files_per_dataset > 0:
        lj_items = lj_items[:max_files_per_dataset]
    items.extend(lj_items)

    bz_rows, bz_errors = parse_bznsyp_labels(bznsyp_label)
    errors.extend(bz_errors)
    bz_items: List[Dict[str, Any]] = []
    for utt_id, raw_text in bz_rows:
        src_wav = bznsyp_wave_dir / f"{utt_id}.wav"
        if not src_wav.exists():
            errors.append(f"BZNSYP {utt_id}: missing {src_wav}")
            continue
        spec = DATASET_SPECS["bznsyp"]
        bz_items.append(
            {
                "uid": utt_id,
                "dataset": "bznsyp",
                "lang": spec["lang"],
                "speaker": spec["speaker"],
                "speaker_id": int(spec["speaker_id"]),
                "src_wav": str(src_wav.resolve()),
                "raw_text": raw_text,
            }
        )
    if max_files_per_dataset > 0:
        bz_items = bz_items[:max_files_per_dataset]
    items.extend(bz_items)

    print(f"[prepare] LJSpeech usable items: {len(lj_items)}", flush=True)
    print(f"[prepare] BZNSYP usable items: {len(bz_items)}", flush=True)
    return items, errors


def split_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    val_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    rng.shuffle(rows)
    if not rows:
        return [], []
    if len(rows) == 1:
        return rows, list(rows)
    val_count = min(len(rows) - 1, max(1, int(round(len(rows) * val_ratio))))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    if not train_rows:
        train_rows = list(val_rows)
    return train_rows, val_rows


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['dst_wav']}|{row['speaker_id']}|{row['cleaned_text']}\n")


def main() -> None:
    args = parse_args()
    args.ljspeech_root = args.ljspeech_root.expanduser().resolve()
    args.bznsyp_root = args.bznsyp_root.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.template = args.template.expanduser().resolve()

    ljspeech_metadata = (
        args.ljspeech_metadata.expanduser().resolve()
        if args.ljspeech_metadata
        else (args.ljspeech_root / "metadata.csv").resolve()
    )
    ljspeech_wav_dir = (
        args.ljspeech_wav_dir.expanduser().resolve()
        if args.ljspeech_wav_dir
        else (args.ljspeech_root / "wavs").resolve()
    )
    bznsyp_label = find_bznsyp_label(args.bznsyp_root, args.bznsyp_label)
    bznsyp_wave_dir = (
        args.bznsyp_wave_dir.expanduser().resolve()
        if args.bznsyp_wave_dir
        else (args.bznsyp_root / "Wave").resolve()
    )

    for path in (ljspeech_metadata, ljspeech_wav_dir, bznsyp_label, bznsyp_wave_dir, args.template):
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")
    if int(args.sampling_rate) != 16000:
        raise ValueError(f"This recipe is fixed to 16000 Hz, got {args.sampling_rate}")

    print(f"[prepare] LJSpeech metadata : {ljspeech_metadata}", flush=True)
    print(f"[prepare] LJSpeech wav dir  : {ljspeech_wav_dir}", flush=True)
    print(f"[prepare] BZNSYP label      : {bznsyp_label}", flush=True)
    print(f"[prepare] BZNSYP wav dir    : {bznsyp_wave_dir}", flush=True)

    items, parse_errors = build_items(
        ljspeech_metadata=ljspeech_metadata,
        ljspeech_wav_dir=ljspeech_wav_dir,
        bznsyp_label=bznsyp_label,
        bznsyp_wave_dir=bznsyp_wave_dir,
        max_files_per_dataset=int(args.max_files_per_dataset),
    )
    if not items:
        raise RuntimeError("No usable items after parsing both datasets.")
    print(
        f"[prepare] total usable items: {len(items)} "
        f"(parse errors: {len(parse_errors)}); cleaner={CLEANER_NAMES}; "
        f"symbols={len(ORIGINAL_SYMBOLS)}",
        flush=True,
    )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(
        items,
        data_dir=args.data_dir,
        target_sr=int(args.sampling_rate),
        min_duration=float(args.min_duration),
        max_duration=float(args.max_duration),
        max_cleaned_length=int(args.max_cleaned_length),
        skip_existing=bool(args.skip_existing),
        workers=int(args.workers),
    )

    good = [row for row in results if row.get("status") == "ok"]
    failed = [row for row in results if row.get("status") != "ok"]
    if not good:
        raise RuntimeError("All rows failed to clean or resample.")

    rng = random.Random(args.seed)
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    per_dataset_counts: Dict[str, Dict[str, int]] = {}
    for dataset in ("ljspeech", "bznsyp"):
        ds_rows = [row for row in good if row["dataset"] == dataset]
        ds_train, ds_val = split_rows(ds_rows, val_ratio=float(args.val_ratio), rng=rng)
        train_rows.extend(ds_train)
        val_rows.extend(ds_val)
        per_dataset_counts[dataset] = {
            "speaker_id": int(DATASET_SPECS[dataset]["speaker_id"]),
            "prepared": len(ds_rows),
            "train": len(ds_train),
            "val": len(ds_val),
        }

    # Validation loader must be non-empty.
    if not val_rows and train_rows:
        val_rows = [dict(train_rows[0])]
    if not train_rows and val_rows:
        train_rows = [dict(val_rows[0])]

    # Stable ordering for reproducibility; DataLoader itself shuffles later.
    train_rows.sort(key=lambda r: (str(r["dataset"]), str(r["uid"])))
    val_rows.sort(key=lambda r: (str(r["dataset"]), str(r["uid"])))

    train_txt = args.data_dir / "train.txt"
    val_txt = args.data_dir / "val.txt"
    write_manifest(train_txt, train_rows)
    write_manifest(val_txt, val_rows)

    config_path = (
        args.output_config.expanduser().resolve()
        if args.output_config
        else args.data_dir / "config.json"
    )
    config = json.loads(args.template.read_text(encoding="utf-8"))
    config["data"]["training_files"] = str(train_txt.resolve())
    config["data"]["validation_files"] = str(val_txt.resolve())
    config["data"]["sampling_rate"] = int(args.sampling_rate)
    config["data"]["n_speakers"] = len(DATASET_SPECS)
    config["data"]["mel_fmax"] = float(args.sampling_rate) / 2.0
    config["data"]["text_cleaners"] = CLEANER_NAMES
    config["data"]["cleaned_text"] = True
    config["data"]["max_text_len"] = int(args.max_cleaned_length)
    config["train"]["epochs"] = int(args.epochs)
    config["symbols"] = list(ORIGINAL_SYMBOLS)
    config["speakers"] = {
        str(spec["speaker"]): int(spec["speaker_id"])
        for spec in sorted(DATASET_SPECS.values(), key=lambda s: int(s["speaker_id"]))
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "ljspeech_root": str(args.ljspeech_root),
        "ljspeech_metadata": str(ljspeech_metadata),
        "ljspeech_wav_dir": str(ljspeech_wav_dir),
        "bznsyp_root": str(args.bznsyp_root),
        "bznsyp_label": str(bznsyp_label),
        "bznsyp_wave_dir": str(bznsyp_wave_dir),
        "data_dir": str(args.data_dir),
        "sampling_rate": int(args.sampling_rate),
        "text_cleaners": CLEANER_NAMES,
        "symbols_count": len(ORIGINAL_SYMBOLS),
        "speakers": config["speakers"],
        "total_items": len(items),
        "prepared_rows": len(good),
        "failed_rows": len(failed),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "per_dataset": per_dataset_counts,
        "parse_error_count": len(parse_errors),
        "parse_error_examples": parse_errors[:50],
        "failed_examples": failed[:50],
        "config": str(config_path),
    }
    summary_path = args.data_dir / "prepare_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[prepare] done: train={len(train_rows)}, val={len(val_rows)}, "
        f"failed={len(failed)}; config -> {config_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
