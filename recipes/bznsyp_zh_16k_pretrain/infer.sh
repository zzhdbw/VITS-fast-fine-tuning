#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RECIPE_DIR="recipes/bznsyp_zh_16k_pretrain"
DATA_DIR="$RECIPE_DIR/data/bznsyp_zh_16k"
OUT_DIR="$RECIPE_DIR/output/bznsyp_zh_16k"
PYTHON="${PYTHON:-.venv/bin/python}"

MODEL="${MODEL:-$OUT_DIR/G_latest.pth}"
CONFIG="${CONFIG:-$DATA_DIR/config.json}"
SPEAKER="${SPEAKER:-BZNSYP}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_DIR/infer}"

if [[ $# -lt 1 ]]; then
  cat <<'USAGE'
用法:
  bash recipes/bznsyp_zh_16k_pretrain/infer.sh "<文本>" [输出文件名]

可选环境变量:
  MODEL       默认 recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/G_latest.pth
  CONFIG      默认 recipes/bznsyp_zh_16k_pretrain/data/bznsyp_zh_16k/config.json
  SPEAKER     默认 BZNSYP
  OUTPUT_DIR  默认 recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/infer
USAGE
  exit 1
fi

TEXT="$1"
OUTPUT_NAME="${2:-test}"

exec "$PYTHON" cmd_inference.py \
  -m "$MODEL" \
  -c "$CONFIG" \
  -o "$OUTPUT_DIR" \
  -l 简体中文 \
  -t "$TEXT" \
  -s "$SPEAKER" \
  -on "$OUTPUT_NAME"
