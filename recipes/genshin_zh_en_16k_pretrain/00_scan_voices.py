#!/usr/bin/env python3
"""Step 0: scan Genshin voices and build train/val manifests (file paths only).

This step does not read audio or .lab contents, so it is cheap and fast.
It writes:
    <data-dir>/scan_rows.jsonl
    <data-dir>/speakers.json
    <data-dir>/scan_summary.json
"""

from __future__ import annotations

import argparse
import fnmatch
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from common import LANG_DIRS, RECIPE_DIR, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Genshin zh/en voices into JSONL manifests.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/afs/datasets/TTS/Genshin6.3"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RECIPE_DIR / "data" / "genshin_zh_en_16k",
        help="Recipe data directory.",
    )
    parser.add_argument("--languages", nargs="+", choices=["zh", "en"], default=["zh", "en"])
    parser.add_argument(
        "--min-utts",
        type=int,
        default=5,
        help="Keep only voices with at least this many .wav/.lab pairs.",
    )
    parser.add_argument(
        "--max-utts-per-speaker",
        type=int,
        default=0,
        help="Randomly keep at most this many utterances per voice (0 = all).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=0,
        help="Randomly keep at most this many voices after filtering (0 = all).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation ratio per voice.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include the #Unknown bucket (mixed speakers; excluded by default).",
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=RECIPE_DIR / "exclude_speakers.txt",
        help="Editable speaker blacklist file.  One fnmatch pattern per line; # starts a comment.",
    )
    return parser.parse_args()


def find_wav_lab_pairs(voice_dir: Path) -> List[Tuple[Path, Path]]:
    """Collect .wav/.lab pairs directly under voice_dir.  Subdirectories are ignored."""
    voice_dir = voice_dir.resolve()
    pairs: List[Tuple[Path, Path]] = []
    for entry in sorted(voice_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_file() or not entry.name.lower().endswith(".wav"):
            continue
        lab_path = entry.with_suffix(".lab")
        if lab_path.exists():
            pairs.append((entry, lab_path))
    return pairs

def load_exclude_patterns(path: Path) -> List[str]:
    """Read editable exclusion patterns.  Blank lines and # comments are ignored."""
    if not path.exists():
        print(f"[scan] warning: exclude file not found: {path}", flush=True)
        return []
    patterns: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def match_exclude_pattern(voice_name: str, lang: str, patterns: List[str]) -> Optional[str]:
    """Return the matched pattern, or None.  Patterns are matched case-insensitively."""
    candidates = [voice_name, f"{lang}:{voice_name}"]
    for pattern in patterns:
        for candidate in candidates:
            if fnmatch.fnmatchcase(candidate.lower(), pattern.lower()):
                return pattern
    return None


def scan_language(
    dataset_root: Path,
    lang: str,
    args: argparse.Namespace,
    exclude_patterns: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lang_root = dataset_root / LANG_DIRS[lang]
    if not lang_root.is_dir():
        raise FileNotFoundError(f"Language directory not found: {lang_root}")

    records: List[Dict[str, Any]] = []
    excluded_records: List[Dict[str, Any]] = []
    voice_dirs = sorted([p for p in lang_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for voice_dir in tqdm(voice_dirs, desc=f"[scan {lang}]", unit="voice", dynamic_ncols=True):
        voice_name = voice_dir.name
        if not args.include_unknown and voice_name.strip().lower() == "#unknown":
            excluded_records.append(
                {"lang": lang, "voice": voice_name, "pattern": "#Unknown (built-in)"}
            )
            continue

        matched_pattern = match_exclude_pattern(voice_name, lang, exclude_patterns)
        if matched_pattern is not None:
            excluded_records.append(
                {"lang": lang, "voice": voice_name, "pattern": matched_pattern}
            )
            continue

        pairs = find_wav_lab_pairs(voice_dir)
        if len(pairs) < args.min_utts:
            continue

        rng = random.Random(f"{args.seed}:{lang}:{voice_name}")
        if args.max_utts_per_speaker > 0 and len(pairs) > args.max_utts_per_speaker:
            pairs = rng.sample(pairs, args.max_utts_per_speaker)

        rng.shuffle(pairs)
        n = len(pairs)
        if n >= 2 and args.val_ratio > 0:
            val_n = max(1, int(round(n * args.val_ratio)))
            val_n = min(val_n, n - 1)
        else:
            val_n = 0

        items = []
        for wav_path, lab_path in pairs:
            wav_path = wav_path.resolve()
            lab_path = lab_path.resolve()
            rel_dir = wav_path.parent.relative_to(voice_dir.resolve())
            items.append(
                {
                    "lang": lang,
                    "voice": voice_name,
                    # 中英同名音色合并为同一个 speaker。
                    "speaker": voice_name,
                    "src_wav": str(wav_path),
                    "src_lab": str(lab_path),
                    "rel_dir": "." if rel_dir == Path(".") else str(rel_dir),
                }
            )

        for item in items[:val_n]:
            item["split"] = "val"
            records.append(item)
        for item in items[val_n:]:
            item["split"] = "train"
            records.append(item)

    return records, excluded_records


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    args.exclude_file = args.exclude_file.expanduser().resolve()
    exclude_patterns = load_exclude_patterns(args.exclude_file)
    print(
        f"[scan] speaker exclude file: {args.exclude_file} "
        f"({len(exclude_patterns)} patterns)",
        flush=True,
    )

    all_rows: List[Dict[str, Any]] = []
    all_excluded: List[Dict[str, Any]] = []
    for lang in args.languages:
        rows, excluded = scan_language(
            args.dataset_root,
            lang,
            args,
            exclude_patterns,
        )
        all_rows.extend(rows)
        all_excluded.extend(excluded)
        print(
            f"[scan] {lang}: {len(rows)} rows, {len(excluded)} voices excluded",
            flush=True,
        )

    if not all_rows:
        raise RuntimeError("No usable wav/lab pairs found. Check --dataset-root/--min-utts.")

    # Optional global speaker cap.  Keep language balance as much as possible by
    # shuffling all speakers and then sorting back.
    speaker_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        speaker_rows.setdefault(row["speaker"], []).append(row)

    speakers_sorted = sorted(speaker_rows.keys())
    if args.max_speakers > 0 and len(speakers_sorted) > args.max_speakers:
        chosen = random.Random(args.seed).sample(speakers_sorted, args.max_speakers)
        chosen = sorted(chosen)
        all_rows = [row for speaker in chosen for row in speaker_rows[speaker]]

    # Assign contiguous speaker ids in a deterministic order.
    used_speakers = sorted({row["speaker"] for row in all_rows})
    speaker2id = {speaker: i for i, speaker in enumerate(used_speakers)}
    for row in all_rows:
        row["speaker_id"] = speaker2id[row["speaker"]]

    # Stable output order for reproducibility.
    all_rows.sort(key=lambda r: (r["lang"], r["voice"], r["split"], r["src_wav"]))

    scan_path = args.data_dir / "scan_rows.jsonl"
    n_rows = write_jsonl(scan_path, all_rows)
    write_json(args.data_dir / "speakers.json", speaker2id)

    n_train = sum(1 for r in all_rows if r["split"] == "train")
    n_val = sum(1 for r in all_rows if r["split"] == "val")
    summary = {
        "dataset_root": str(args.dataset_root),
        "data_dir": str(args.data_dir),
        "languages": args.languages,
        "min_utts": args.min_utts,
        "max_utts_per_speaker": args.max_utts_per_speaker,
        "max_speakers": args.max_speakers,
        "val_ratio": args.val_ratio,
        "include_unknown": args.include_unknown,
        "exclude_file": str(args.exclude_file),
        "exclude_patterns": exclude_patterns,
        "excluded_voices": len(all_excluded),
        "excluded_examples": all_excluded[:50],
        "speakers": len(used_speakers),
        "rows": n_rows,
        "train_rows": n_train,
        "val_rows": n_val,
    }
    write_json(args.data_dir / "scan_summary.json", summary)
    print(
        f"[scan] wrote {n_rows} rows, {len(used_speakers)} speakers "
        f"(train={n_train}, val={n_val}) -> {scan_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
