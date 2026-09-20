#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/zzh/code/VITS-fast-fine-tuning

ORIG_DIR="recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k"
EXP_DIR="recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_400ep"
CONFIG="recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k/config.json"

mkdir -p "$EXP_DIR"


cp -v "$ORIG_DIR/G_latest.pth" "$EXP_DIR/G_latest.pth"
cp -v "$ORIG_DIR/D_latest.pth" "$EXP_DIR/D_latest.pth"



export MASTER_ADDR="localhost"
export MASTER_PORT=18082
export CUDA_VISIBLE_DEVICES="0,1,4,5"
nohup .venv/bin/python -u finetune_speaker_v2.py \
  -m "$EXP_DIR" \
  -c "$CONFIG" \
  --max_epochs 400 \
  --preserved 50 \
  --num_workers 16 \
  --grad_clip 500 \
  --warmup_steps 2000 \
  --train_with_pretrained_model False \
  --drop_speaker_embed False \
  --cont True \
  --use_swanlab True \
  --swanlab_project vits-fast-fine-tuning-pretrain \
  --swanlab_name genshin_zh_en_16k_400ep \
  --swanlab_mode online \
  > "$EXP_DIR/nohup_400ep.log" 2>&1 &

echo $! > "$EXP_DIR/train.pid"
echo "[400ep] started"
echo "PID : $(cat "$EXP_DIR/train.pid")"
echo "LOG : $EXP_DIR/nohup_400ep.log"
