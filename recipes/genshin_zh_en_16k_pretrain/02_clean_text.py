#!/usr/bin/env python3
"""Step 2: clean .lab text and create VITS train/val annotation files.

Input:
    <data-dir>/resampled_rows.jsonl

Output:
    <data-dir>/cleaned_rows.jsonl
    <data-dir>/train.txt       path|speaker_id|cleaned_text
    <data-dir>/val.txt
    <data-dir>/clean_summary.json

The cleaner is ``cjke_cleaners2`` from the repository.  Only [ZH] and [EN]
tags are added by this recipe, so the resulting model is a Chinese/English model.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from common import CLEANER_NAMES, LANG_TAGS, RECIPE_DIR, normalize_lab_text, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean .lab text and build train.txt/val.txt.")
    parser.add_argument("--data-dir", type=Path, default=RECIPE_DIR / "data" / "genshin_zh_en_16k")
    parser.add_argument("--max-raw-text-len", type=int, default=150)
    parser.add_argument("--max-cleaned-text-len", type=int, default=190)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=64, help="Rows per worker batch.")
    parser.add_argument("--max-rows", type=int, default=0, help="Only process first N rows (0 = all).")
    return parser.parse_args()


def clean_batch(
    batch: Sequence[Dict[str, Any]],
    *,
    max_raw_text_len: int,
    max_cleaned_text_len: int,
) -> List[Tuple[Dict[str, Any], str]]:
    from text import _clean_text

    results: List[Tuple[Dict[str, Any], str]] = []
    for row in batch:
        try:
            lab_path = Path(row["src_lab"])
            raw_text = normalize_lab_text(lab_path.read_text(encoding="utf-8"))
            if not raw_text:
                results.append((row, "empty raw text"))
                continue
            if len(raw_text) > max_raw_text_len:
                results.append((row, f"raw text too long ({len(raw_text)})"))
                continue

            lang = row["lang"]
            tagged = f"{LANG_TAGS[lang]}{raw_text}{LANG_TAGS[lang]}"
            cleaned = _clean_text(tagged, CLEANER_NAMES)
            cleaned = normalize_lab_text(cleaned)
            if not cleaned:
                results.append((row, "empty cleaned text"))
                continue
            if len(cleaned) > max_cleaned_text_len:
                results.append((row, f"cleaned text too long ({len(cleaned)})"))
                continue

            new_row = dict(row)
            new_row["cleaned_text"] = cleaned
            results.append((new_row, ""))
        except Exception as exc:  # noqa: BLE001 - keep the long job running
            results.append((row, f"{type(exc).__name__}: {exc}"))
    return results


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['dst_wav']}|{row['speaker_id']}|{row['cleaned_text']}\n")


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    input_path = args.data_dir / "resampled_rows.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}; run 01_resample_audio.py first.")

    rows = list(read_jsonl(input_path))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise RuntimeError("No rows to clean.")

    print(
        f"[clean] rows={len(rows)}, cleaners={CLEANER_NAMES}, workers={args.workers}",
        flush=True,
    )

    worker = partial(
        clean_batch,
        max_raw_text_len=args.max_raw_text_len,
        max_cleaned_text_len=args.max_cleaned_text_len,
    )
    batches = [rows[i : i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

    good: List[Dict[str, Any]] = []
    errors: List[str] = []

    def consume(batch_results: Sequence[Tuple[Dict[str, Any], str]]) -> None:
        for row, error in batch_results:
            if error:
                errors.append(f"{row.get('src_lab')}: {error}")
            else:
                good.append(row)

    if args.workers <= 1:
        for batch in batches:
            consume(worker(batch))
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=max(1, args.workers))
        except (PermissionError, OSError) as exc:
            print(
                f"[clean] multiprocessing unavailable ({exc}); falling back to serial. "
                "Set --workers 1 to silence this message.",
                flush=True,
            )
            for batch in batches:
                consume(worker(batch))
        else:
            with executor:
                futures = [executor.submit(worker, batch) for batch in batches]
                for future in as_completed(futures):
                    consume(future.result())

    if not good:
        raise RuntimeError("No text was successfully cleaned.")

    good.sort(key=lambda r: (r["lang"], r["voice"], r["split"], r["dst_wav"]))
    train_rows = [r for r in good if r["split"] == "train"]
    val_rows = [r for r in good if r["split"] == "val"]
    # The training code requires a non-empty validation loader.
    if not val_rows and train_rows:
        val_rows = [dict(train_rows[0])]

    train_txt = args.data_dir / "train.txt"
    val_txt = args.data_dir / "val.txt"
    write_manifest(train_txt, train_rows)
    write_manifest(val_txt, val_rows)

    cleaned_path = args.data_dir / "cleaned_rows.jsonl"
    write_jsonl(cleaned_path, good)
    summary = {
        "input_rows": len(rows),
        "cleaned_rows": len(good),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "skipped_or_failed": len(errors),
        "max_raw_text_len": args.max_raw_text_len,
        "max_cleaned_text_len": args.max_cleaned_text_len,
        "error_examples": errors[:50],
    }
    write_json(args.data_dir / "clean_summary.json", summary)
    print(
        f"[clean] train={len(train_rows)}, val={len(val_rows)}, "
        f"skipped/failed={len(errors)} -> {train_txt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
