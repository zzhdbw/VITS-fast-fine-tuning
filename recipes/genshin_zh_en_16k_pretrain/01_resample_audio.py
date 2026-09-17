#!/usr/bin/env python3
"""Step 1: resample scanned Genshin wav files to 16 kHz mono PCM16.

Input:
    <data-dir>/scan_rows.jsonl

Output:
    <data-dir>/wav/<lang>/<voice>/<category>/<name>.wav
    <data-dir>/resampled_rows.jsonl
    <data-dir>/resample_summary.json

This is the heavy step.  For quick validation use, e.g.:
    python 01_resample_audio.py --data-dir /tmp/gs_test --max-files 20
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from common import RECIPE_DIR, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample Genshin audio to 16 kHz.")
    parser.add_argument("--data-dir", type=Path, default=RECIPE_DIR / "data" / "genshin_zh_en_16k")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=0.6)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=16.0,
        help="Longest kept utterance.  16 s matches the current VITS bucket boundary at 16 kHz.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=32, help="Rows per worker batch.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only process the first N rows (0 = all).  Useful for validation.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing output wav files and only rebuild manifests.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing audio.")
    return parser.parse_args()


def add_dst_path(row: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
    row = dict(row)
    rel_dir = row.get("rel_dir") or "."
    if rel_dir in (".", ""):
        subdir = "_direct"
    else:
        subdir = rel_dir
    src_name = Path(row["src_wav"]).name
    dst = data_dir / "wav" / row["lang"] / row["voice"] / subdir / src_name
    row["dst_wav"] = str(dst)
    return row


def process_batch(
    batch: Sequence[Dict[str, Any]],
    *,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    skip_existing: bool,
) -> List[Tuple[Dict[str, Any], str]]:
    """Resample one batch.  Returns (row, error) pairs; empty error means success."""
    results: List[Tuple[Dict[str, Any], str]] = []
    for row in batch:
        try:
            src = Path(row["src_wav"])
            dst = Path(row["dst_wav"])
            if skip_existing and dst.exists():
                results.append((row, ""))
                continue

            audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            duration = len(audio) / float(sr)
            if duration < min_duration or duration > max_duration:
                results.append((row, f"skip duration={duration:.2f}s"))
                continue

            if sr != target_sr:
                g = math.gcd(int(sr), int(target_sr))
                audio = resample_poly(audio, target_sr // g, int(sr) // g).astype(np.float32)

            audio = np.clip(audio, -1.0, 1.0)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.stem + ".tmp.wav")
            sf.write(str(tmp), audio, target_sr, subtype="PCM_16")
            os.replace(tmp, dst)

            row = dict(row)
            row["src_sr"] = int(sr)
            row["duration"] = float(duration)
            results.append((row, ""))
        except Exception as exc:  # noqa: BLE001 - keep the long job running
            results.append((row, f"{type(exc).__name__}: {exc}"))
    return results


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    scan_path = args.data_dir / "scan_rows.jsonl"
    if not scan_path.exists():
        raise FileNotFoundError(f"Missing {scan_path}; run 00_scan_voices.py first.")

    rows = list(read_jsonl(scan_path))
    if args.max_files > 0:
        rows = rows[: args.max_files]
    if not rows:
        raise RuntimeError("No rows to resample.")

    print(
        f"[resample] rows={len(rows)}, target_sr={args.target_sr}, "
        f"duration=({args.min_duration}, {args.max_duration}]s, workers={args.workers}",
        flush=True,
    )
    if args.dry_run:
        print(f"[resample] dry-run only, no audio will be written. rows={len(rows)}", flush=True)
        for row in rows[:5]:
            preview = add_dst_path(row, args.data_dir)
            print(f"  {preview['src_wav']} -> {preview['dst_wav']}")
        print("[resample] dry-run done.", flush=True)
        return

    rows = [add_dst_path(row, args.data_dir) for row in rows]

    worker = partial(
        process_batch,
        target_sr=args.target_sr,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        skip_existing=args.skip_existing,
    )
    batches = [rows[i : i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

    good: List[Dict[str, Any]] = []
    errors: List[str] = []
    progress = tqdm(total=len(rows), desc="[resample]", unit="row", dynamic_ncols=True)

    def consume(batch_results: Sequence[Tuple[Dict[str, Any], str]]) -> None:
        for row, error in batch_results:
            if error:
                errors.append(f"{row.get('src_wav')}: {error}")
            else:
                good.append(row)
        progress.update(len(batch_results))

    if args.workers <= 1:
        for batch in batches:
            consume(worker(batch))
    else:
        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=max(1, args.workers))
            backend = "process"
        except (PermissionError, OSError) as exc:
            print(
                f"[resample] process workers unavailable ({exc}); trying threads.",
                flush=True,
            )
            try:
                executor = ThreadPoolExecutor(max_workers=max(1, args.workers))
                backend = "thread"
            except Exception as thread_exc:  # noqa: BLE001
                print(
                    f"[resample] thread workers unavailable ({thread_exc}); falling back to serial.",
                    flush=True,
                )
                for batch in batches:
                    consume(worker(batch))

        if executor is not None:
            with executor:
                futures = [executor.submit(worker, batch) for batch in batches]
                for future in as_completed(futures):
                    consume(future.result())

    progress.set_postfix(good=len(good), errors=len(errors))
    progress.close()

    if not good:
        for err in errors[:20]:
            print(f"[resample] {err}", flush=True)
        raise RuntimeError("No audio was successfully resampled.")

    good.sort(key=lambda r: (r["lang"], r["voice"], r["split"], r["dst_wav"]))
    out_path = args.data_dir / "resampled_rows.jsonl"
    write_jsonl(out_path, good)
    summary = {
        "input_rows": len(rows),
        "success_rows": len(good),
        "skipped_or_failed": len(errors),
        "target_sr": args.target_sr,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "error_examples": errors[:50],
    }
    write_json(args.data_dir / "resample_summary.json", summary)
    print(f"[resample] wrote {len(good)} good rows -> {out_path}", flush=True)
    if errors:
        print(f"[resample] skipped/failed={len(errors)}; see resample_summary.json", flush=True)


if __name__ == "__main__":
    main()
