#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/zzh/code/VITS-fast-fine-tuning

PYTHON=".venv/bin/python"
MODEL_DIR="recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k"
CONFIG="recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k/config.json"
OUT_DIR="recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/compare_ckpts_mixed"

SPEAKER="八重神子"
LANGUAGE="auto"
TEXT="你好，今天我想测试一下中英混说。We can start with a simple sentence, then continue with something a little longer. 如果效果不错，以后就可以用它来读一些 mixed text。"

CKPTS=(
  G_70000.pth
  G_75000.pth
  G_80000.pth
  G_82000.pth

)

mkdir -p "$OUT_DIR"

for ckpt in "${CKPTS[@]}"; do
  step="${ckpt%.pth}"
  echo "[compare-mixed] $step"
  "$PYTHON" recipes/genshin_zh_en_16k_pretrain/04_infer.py \
    --checkpoint "$MODEL_DIR/$ckpt" \
    --config "$CONFIG" \
    --speaker "$SPEAKER" \
    --language "$LANGUAGE" \
    --text "$TEXT" \
    --output "$OUT_DIR/${step}_mixed.wav" \
    --device cpu
done

echo "[compare-mixed] done: $OUT_DIR"
