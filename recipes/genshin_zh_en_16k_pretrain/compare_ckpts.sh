#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/zzh/code/VITS-fast-fine-tuning

PYTHON=".venv/bin/python"
MODEL_DIR="recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k"
CONFIG="recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k/config.json"
OUT_DIR="recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/compare_ckpts"

SPEAKER="八重神子"
LANGUAGE="auto"
TEXT="你好，这是一段测试语音。"

CKPTS=(
  G_10000.pth
  G_20000.pth
  G_30000.pth
  G_40000.pth
  G_50000.pth
  G_56000.pth
)

mkdir -p "$OUT_DIR"

for ckpt in "${CKPTS[@]}"; do
  step="${ckpt%.pth}"
  echo "[compare] $step"
  "$PYTHON" recipes/genshin_zh_en_16k_pretrain/04_infer.py \
    --checkpoint "$MODEL_DIR/$ckpt" \
    --config "$CONFIG" \
    --speaker "$SPEAKER" \
    --language "$LANGUAGE" \
    --text "$TEXT" \
    --output "$OUT_DIR/${step}_${SPEAKER}.wav" \
    --device cpu
done

echo "[compare] done: $OUT_DIR"
