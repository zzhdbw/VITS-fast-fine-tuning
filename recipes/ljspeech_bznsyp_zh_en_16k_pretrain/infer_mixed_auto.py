#!/usr/bin/env python3
"""Auto-tag Chinese/English spans and delegate to the original cmd_inference.py.

``cmd_inference.py -l Mix`` does not add language tags itself.  This wrapper
takes plain mixed text such as::

    你好，这是 hello world 测试。

and converts it to::

    [ZH]你好，这是 [ZH][EN]hello world [EN][ZH] 测试。[ZH]

Then it calls the repository's original ``cmd_inference.py`` with ``--language
Mix``, so the checkpoint and config do not need to be changed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parents[1]


def auto_tag_zh_en(text: str) -> str:
    """Add ``[ZH]`` / ``[EN]`` tags to Chinese and English spans.

    If the text already contains either tag, it is treated as manually tagged
    and returned unchanged.
    """
    if "[ZH]" in text or "[EN]" in text:
        return text

    result: list[str] = []
    current: str | None = None

    def close_if_needed() -> None:
        nonlocal current
        if current == "ZH":
            result.append("[ZH]")
        elif current == "EN":
            result.append("[EN]")
        current = None

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            lang = "ZH"
        elif ch.isascii() and ch.isalpha():
            lang = "EN"
        else:
            # 空格、标点、数字等保留在当前语言段内。
            result.append(ch)
            continue

        if lang != current:
            close_if_needed()
            result.append("[ZH]" if lang == "ZH" else "[EN]")
            current = lang
        result.append(ch)

    close_if_needed()
    return "".join(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-tag Chinese/English mixed text and run original cmd_inference.py."
    )
    parser.add_argument("--model_path", required=True, help="G_latest.pth / G_xxx.pth.")
    parser.add_argument("--config_path", required=True, help="Recipe config.json.")
    parser.add_argument("--output_path", required=True, help="Output directory.")
    parser.add_argument("--text", required=True, help="Plain Chinese/English mixed text.")
    parser.add_argument("--spk", required=True, help="Speaker name in config.json.")
    parser.add_argument("--output_name", required=True, help="Output wav name without .wav.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tagged_text = auto_tag_zh_en(args.text)
    print(f"[auto-tag] {tagged_text}", flush=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "cmd_inference.py"),
        "--model_path",
        args.model_path,
        "--config_path",
        args.config_path,
        "--output_path",
        args.output_path,
        "--language",
        "Mix",
        "--text",
        tagged_text,
        "--spk",
        args.spk,
        "--output_name",
        args.output_name,
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
