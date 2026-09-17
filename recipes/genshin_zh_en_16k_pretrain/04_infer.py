#!/usr/bin/env python3
"""Step 6: simple CLI inference for the Genshin zh-en 16 kHz model.

Examples:
    python 04_infer.py --list-speakers
    python 04_infer.py --speaker 魔女M --language zh --text "你好，这是测试语音。"
    python 04_infer.py --speaker 魔女M --language en --text "Hello, this is a test."
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from common import RECIPE_DIR
from models_infer import SynthesizerTrn
from text import text_to_sequence
import commons
from utils import get_hparams_from_file, load_checkpoint


LANG_TAGS = {
    "zh": "[ZH]",
    "en": "[EN]",
    "mix": "",
}

ZH_OPEN = "[ZH]"
EN_OPEN = "[EN]"


def auto_tag_zh_en(text: str) -> str:
    """Tag Chinese and English spans automatically, e.g.

    "你好 hello 世界" -> "[ZH]你好[ZH][EN] hello [EN][ZH]世界[ZH]"
    """
    result: list[str] = []
    current: str | None = None

    def close_if_needed() -> None:
        nonlocal current
        if current == "ZH":
            result.append(ZH_OPEN)
        elif current == "EN":
            result.append(EN_OPEN)
        current = None

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            lang = "ZH"
        elif ch.isascii() and ch.isalpha():
            lang = "EN"
        else:
            # 标点、空格、数字等暂时归到当前语言段。
            if current is not None:
                result.append(ch)
            else:
                result.append(ch)
            continue

        if lang != current:
            close_if_needed()
            result.append(ZH_OPEN if lang == "ZH" else EN_OPEN)
            current = lang
        result.append(ch)

    close_if_needed()
    return "".join(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple VITS inference for 16 kHz Genshin zh/en model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=RECIPE_DIR / "output" / "genshin_zh_en_16k" / "G_latest.pth",
        help="Path to G_latest.pth / G_xxx.pth.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=RECIPE_DIR / "data" / "genshin_zh_en_16k" / "config.json",
        help="Path to config.json.",
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize.")
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Text file; each non-empty line is synthesized to one wav.",
    )
    parser.add_argument("--speaker", type=str, default=None, help="Speaker name in config.json.")
    parser.add_argument(
        "--language",
        type=str,
        choices=["auto", "zh", "en", "mix"],
        default="auto",
        help="auto tags Chinese/English spans; zh/en wrap the whole text; mix adds no tag.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output wav path for a single text.  Default: <output-dir>/<speaker>_<time>.wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RECIPE_DIR / "output" / "genshin_zh_en_16k" / "infer",
        help="Directory for generated wav files.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed.")
    parser.add_argument("--noise-scale", type=float, default=0.667, help="TTS noise_scale.")
    parser.add_argument("--noise-scale-w", type=float, default=0.8, help="TTS noise_scale_w.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print available speaker names and exit.",
    )
    return parser.parse_args()


def text_to_ids(text: str, hps, verbose: bool = False) -> torch.LongTensor:
    # text_to_sequence prints the cleaned text and lengths; capture it so the CLI
    # can decide whether to show it.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ids = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if verbose:
        debug = buffer.getvalue().strip()
        if debug:
            print("[infer] text frontend output:")
            for line in debug.splitlines():
                print(f"[infer]   {line}")
    if hps.data.add_blank:
        ids = commons.intersperse(ids, 0)
    return torch.LongTensor(ids)


def load_model(checkpoint: Path, config: Path, device: torch.device) -> SynthesizerTrn:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    hps = get_hparams_from_file(str(config))
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()
    load_checkpoint(str(checkpoint), net_g, None)
    return net_g


def synthesize_one(
    net_g: SynthesizerTrn,
    hps,
    text: str,
    speaker: str,
    language: str,
    device: torch.device,
    speed: float,
    noise_scale: float,
    noise_scale_w: float,
) -> np.ndarray:
    if speaker not in hps.speakers:
        available = ", ".join(hps.speakers.keys())
        raise KeyError(f"Unknown speaker {speaker!r}. Available: {available}")

    if language == "auto":
        tagged_text = auto_tag_zh_en(text)
    else:
        tag = LANG_TAGS[language]
        tagged_text = f"{tag}{text}{tag}" if tag else text

    print(f"[infer] raw text   : {text}")
    print(f"[infer] tagged text: {tagged_text}")
    text_ids = text_to_ids(tagged_text, hps, verbose=True).to(device)
    print(f"[infer] text ids   : {text_ids.tolist()}")
    text_lengths = torch.LongTensor([text_ids.size(0)]).to(device)
    speaker_id = torch.LongTensor([int(hps.speakers[speaker])]).to(device)

    with torch.no_grad():
        audio = net_g.infer(
            text_ids.unsqueeze(0),
            text_lengths,
            sid=speaker_id,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=1.0 / speed,
        )[0][0, 0].float().cpu().numpy()
    return audio


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.list_speakers:
        if not args.config.exists():
            raise FileNotFoundError(f"Config not found: {args.config}")
        hps = get_hparams_from_file(str(args.config))
        for name, sid in sorted(hps.speakers.items(), key=lambda kv: int(kv[1])):
            print(f"{sid:4d}  {name}")
        return

    if args.text is None and args.text_file is None:
        args.text = "你好，这是一段测试语音。"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"[infer] checkpoint: {args.checkpoint}")
    print(f"[infer] config    : {args.config}")
    print(f"[infer] device    : {device}")
    net_g = load_model(args.checkpoint, args.config, device)
    hps = get_hparams_from_file(str(args.config))

    speakers = list(hps.speakers.keys())
    speaker = args.speaker or speakers[0]

    if args.text is not None and args.text_file is not None:
        raise ValueError("Use either --text or --text-file, not both.")

    if args.text is not None:
        texts = [args.text]
        if args.output is not None:
            out_paths = [args.output.expanduser().resolve()]
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_paths = [args.output_dir / f"{speaker}_{stamp}.wav"]
    else:
        lines = [
            line.strip()
            for line in args.text_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise RuntimeError(f"No non-empty lines in {args.text_file}")
        texts = lines
        stem = args.text_file.stem
        out_paths = [args.output_dir / f"{stem}_{i:03d}.wav" for i in range(len(lines))]

    for i, (text, out_path) in enumerate(zip(texts, out_paths), 1):
        audio = synthesize_one(
            net_g,
            hps,
            text,
            speaker,
            args.language,
            device,
            args.speed,
            args.noise_scale,
            args.noise_scale_w,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, int(hps.data.sampling_rate), subtype="PCM_16")
        print(f"[infer] {i}/{len(texts)} wrote {out_path}")

    print("[infer] done.")


if __name__ == "__main__":
    main()
