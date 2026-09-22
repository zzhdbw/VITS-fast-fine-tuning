#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RECIPE_DIR="recipes/ljspeech_bznsyp_zh_en_16k_pretrain"
DATA_DIR="$RECIPE_DIR/data/ljspeech_bznsyp_zh_en_16k"
OUT_DIR="$RECIPE_DIR/output/ljspeech_bznsyp_zh_en_16k"
PYTHON="${PYTHON:-.venv/bin/python}"

MODEL="${MODEL:-$OUT_DIR/G_latest.pth}"
CONFIG="${CONFIG:-$DATA_DIR/config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_DIR/infer}"

if [[ $# -lt 1 ]]; then
  cat <<'USAGE'
用法:
  bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer.sh "<文本>" [输出文件名] [语言] [音色]

示例:
  bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer.sh \
    "Hello, this is an LJSpeech test." lj_test English LJSpeech

  bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer.sh \
    "你好，这是 BZNSYP 测试语音。" bz_test 简体中文 BZNSYP

  # 中英混读需要手写标签，并用 Mix（cmd_inference.py 不额外加标签）：
  bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer.sh \
    "[ZH]你好[ZH][EN] hello world [EN]" mix_test Mix LJSpeech

可选环境变量:
  MODEL       默认 recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/G_latest.pth
  CONFIG      默认 recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json
  OUTPUT_DIR  默认 recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/infer
  PYTHON      默认 .venv/bin/python
USAGE
  exit 1
fi

TEXT="$1"
OUTPUT_NAME="${2:-test}"
LANGUAGE="${3:-${LANGUAGE:-English}}"
SPEAKER="${4:-${SPEAKER:-LJSpeech}}"

if [[ ! -f "$MODEL" ]]; then
  echo "找不到模型文件: $MODEL" >&2
  echo "请先训练或通过 MODEL=/path/to/G_latest.pth 指定 checkpoint。" >&2
  exit 1
fi

exec "$PYTHON" cmd_inference.py \
  -m "$MODEL" \
  -c "$CONFIG" \
  -o "$OUTPUT_DIR" \
  -l "$LANGUAGE" \
  -t "$TEXT" \
  -s "$SPEAKER" \
  -on "$OUTPUT_NAME"
