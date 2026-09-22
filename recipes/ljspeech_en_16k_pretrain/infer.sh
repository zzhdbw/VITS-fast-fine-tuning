#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RECIPE_DIR="recipes/ljspeech_en_16k_pretrain"
DATA_DIR="$RECIPE_DIR/data/ljspeech_en_16k"
OUT_DIR="$RECIPE_DIR/output/ljspeech_en_16k"
PYTHON="${PYTHON:-.venv/bin/python}"

MODEL="${MODEL:-$OUT_DIR/G_latest.pth}"
CONFIG="${CONFIG:-$DATA_DIR/config.json}"
SPEAKER="${SPEAKER:-LJSpeech}"
LANGUAGE="${LANGUAGE:-English}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_DIR/infer}"

if [[ $# -lt 1 ]]; then
  cat <<'USAGE'
用法:
  bash recipes/ljspeech_en_16k_pretrain/infer.sh "<英文文本>" [输出文件名]

可选环境变量:
  MODEL       默认 recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/G_latest.pth
  CONFIG      默认 recipes/ljspeech_en_16k_pretrain/data/ljspeech_en_16k/config.json
  SPEAKER     默认 LJSpeech
  LANGUAGE    默认 English（cmd_inference 会加 [EN] 标签，cjke_cleaners2 会转 IPA）
  OUTPUT_DIR  默认 recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/infer
USAGE
  exit 1
fi

TEXT="$1"
OUTPUT_NAME="${2:-test}"

exec "$PYTHON" cmd_inference.py \
  -m "$MODEL" \
  -c "$CONFIG" \
  -o "$OUTPUT_DIR" \
  -l "$LANGUAGE" \
  -t "$TEXT" \
  -s "$SPEAKER" \
  -on "$OUTPUT_NAME"
