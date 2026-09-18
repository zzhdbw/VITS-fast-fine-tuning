#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/zzh/code/VITS-fast-fine-tuning

ORIG_DIR="recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k"
EXP_DIR="recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_lr1e4"
CONFIG="recipes/genshin_zh_en_16k_pretrain/experiments/lr1e-4/config.json"

# 续训起点：
#   STEP=auto   -> 自动使用原实验最新的 G_xxxxx.pth / D_xxxxx.pth
#   STEP=59000  -> 固定从原实验的 G_59000.pth / D_59000.pth 开始
#
# 如果新实验目录里已经存在 G_latest.pth / D_latest.pth，默认会复用，不会覆盖。
# 想换起始 ckpt：
#   RESET_RESUME=1 STEP=59000 bash run_resume_lr1e4.sh
STEP="${STEP:-auto}"
RESET_RESUME="${RESET_RESUME:-0}"

mkdir -p "$EXP_DIR"

if [[ "$STEP" == "auto" ]]; then
  STEP=$(ls "$ORIG_DIR"/G_*.pth 2>/dev/null | sed -E 's#.*/G_([0-9]+)\.pth#\1#' | sort -n | tail -1)
fi

if [[ -z "$STEP" ]]; then
  echo "找不到原实验的 G_*.pth，无法续训。" >&2
  exit 1
fi

if [[ ! -f "$ORIG_DIR/G_${STEP}.pth" || ! -f "$ORIG_DIR/D_${STEP}.pth" ]]; then
  echo "找不到 checkpoint: $ORIG_DIR/G_${STEP}.pth 或 D_${STEP}.pth" >&2
  exit 1
fi

if [[ "$RESET_RESUME" == "1" ]]; then
  echo "[lr1e4] remove old resume checkpoints in $EXP_DIR"
  rm -f "$EXP_DIR/G_latest.pth" "$EXP_DIR/D_latest.pth"
fi

if [[ ! -f "$EXP_DIR/G_latest.pth" || ! -f "$EXP_DIR/D_latest.pth" ]]; then
  echo "[lr1e4] copy checkpoint step $STEP"
  cp -v "$ORIG_DIR/G_${STEP}.pth" "$EXP_DIR/G_latest.pth"
  cp -v "$ORIG_DIR/D_${STEP}.pth" "$EXP_DIR/D_latest.pth"
  echo "$STEP" > "$EXP_DIR/resume_step.txt"
else
  echo "[lr1e4] reuse existing $EXP_DIR/G_latest.pth and D_latest.pth"
  echo "[lr1e4] current resume step: $(cat "$EXP_DIR/resume_step.txt" 2>/dev/null || echo unknown)"
  echo "[lr1e4] if you want another start step, set RESET_RESUME=1 STEP=xxxxx"
fi

# 用不同 GPU，避免和原实验抢卡。确认这些卡空闲后再运行。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

# 原实验占用 8000 端口，低学习率实验用另一个端口。
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-8001}"

nohup .venv/bin/python -u finetune_speaker_v2.py \
  -m "$EXP_DIR" \
  -c "$CONFIG" \
  --max_epochs 100 \
  --preserved 20 \
  --num_workers 8 \
  --grad_clip 500 \
  --warmup_steps 2000 \
  --train_with_pretrained_model False \
  --drop_speaker_embed False \
  --cont True \
  --use_swanlab True \
  --swanlab_project vits-fast-fine-tuning-pretrain \
  --swanlab_name genshin_zh_en_16k_lr1e4 \
  --swanlab_mode online \
  > "$EXP_DIR/nohup_lr1e4.log" 2>&1 &

echo $! > "$EXP_DIR/train.pid"
echo "[lr1e4] started"
echo "PID : $(cat "$EXP_DIR/train.pid")"
echo "STEP: $(cat "$EXP_DIR/resume_step.txt" 2>/dev/null || echo unknown)"
echo "LOG : $EXP_DIR/nohup_lr1e4.log"
