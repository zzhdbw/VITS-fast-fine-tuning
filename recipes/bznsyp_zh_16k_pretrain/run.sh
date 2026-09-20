#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/zzh/code/VITS-fast-fine-tuning

# ========================== Step 1: prepare data ==========================
# 解析 BZNSYP 标签，重采样到 16 kHz，并生成 train.txt / val.txt / config.json。
# 已经准备过时，--skip-existing 会复用已有 wav。
# 快速验证可以加：--max-files 3
.venv/bin/python recipes/bznsyp_zh_16k_pretrain/00_prepare_bznsyp.py \
  --dataset-root /mnt/afs/datasets/TTS/BZNSYP \
  --data-dir recipes/bznsyp_zh_16k_pretrain/data/bznsyp_zh_16k \
  --sampling-rate 16000 \
  --val-ratio 0.02 \
  --workers 16 \
  --skip-existing

# ======================= Step 2: pretrain 1000 epochs =======================
# 后台训练，日志写到：
#   recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/nohup_train.log
#   recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/train.pid
# 暂时不训练：把下面整段 Step 2 注释掉。
# 默认开启 SwanLab，项目名 vits-fast-fine-tuning-pretrain。
# 默认使用 6,7 号卡；也可以这样覆盖：
#   CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=8010 bash recipes/bznsyp_zh_16k_pretrain/run.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-8002}"
mkdir -p recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k
nohup .venv/bin/python -u finetune_speaker_v2.py \
  -m recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k \
  -c recipes/bznsyp_zh_16k_pretrain/data/bznsyp_zh_16k/config.json \
  --max_epochs 1000 \
  --preserved 100 \
  --num_workers 8 \
  --grad_clip 500 \
  --warmup_steps 1000 \
  --train_with_pretrained_model False \
  --drop_speaker_embed False \
  --use_swanlab True \
  --swanlab_project vits-fast-fine-tuning-pretrain \
  --swanlab_name bznsyp_zh_16k \
  --swanlab_mode online \
  > recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/nohup_train.log 2>&1 &
echo $! > recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/train.pid
echo "Training started in background."
echo "PID : $(cat recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/train.pid)"
echo "Log : recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/nohup_train.log"

# =========================== Step 3: inference ===========================
# 训练完成后合成一条测试音频；训练没跑完或没有 G_latest.pth 时注释掉。
.venv/bin/python cmd_inference.py \
  -m recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/G_latest.pth \
  -c recipes/bznsyp_zh_16k_pretrain/data/bznsyp_zh_16k/config.json \
  -o recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/infer \
  -l 简体中文 \
  -t "你好，这是 BZNSYP 中文单人音色的测试语音。are you happy?" \
  -s BZNSYP \
  -on test
