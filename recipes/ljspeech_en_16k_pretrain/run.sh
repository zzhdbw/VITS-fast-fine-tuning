#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/zzh/code/VITS-fast-fine-tuning

# ========================== Step 1: prepare data ==========================
# 解析 LJSpeech-1.1 metadata.csv，复用仓库原本的 cjke_cleaners2/68 音素词表，
# 重采样到 16 kHz，并生成 train.txt / val.txt / config.json。
# 已经准备过时，--skip-existing 会复用已有 wav。
# 快速验证可以加：--max-files 3
.venv/bin/python recipes/ljspeech_en_16k_pretrain/00_prepare_ljspeech.py \
  --dataset-root /mnt/afs/datasets/TTS/LJSpeech-1.1 \
  --data-dir recipes/ljspeech_en_16k_pretrain/data/ljspeech_en_16k \
  --sampling-rate 16000 \
  --val-ratio 0.02 \
  --workers 16 \
  --skip-existing

# ======================= Step 2: pretrain 1000 epochs =======================
# 后台训练，日志写到：
#   recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/nohup_train.log
#   recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/train.pid
# 暂时不训练：把下面整段 Step 2 注释掉。
# 默认开启 SwanLab，项目名 vits-fast-fine-tuning-pretrain。
# 默认使用 6,7 号卡；也可以这样覆盖：
#   CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=8010 bash recipes/ljspeech_en_16k_pretrain/run.sh
export CUDA_VISIBLE_DEVICES="6,7"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-9103}"
mkdir -p recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k
nohup .venv/bin/python -u finetune_speaker_v2.py \
  -m recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k \
  -c recipes/ljspeech_en_16k_pretrain/data/ljspeech_en_16k/config.json \
  --max_epochs 1000 \
  --preserved 100 \
  --num_workers 8 \
  --grad_clip 500 \
  --warmup_steps 1000 \
  --train_with_pretrained_model False \
  --drop_speaker_embed False \
  --use_swanlab True \
  --swanlab_project vits-fast-fine-tuning-pretrain \
  --swanlab_name ljspeech_en_16k \
  --swanlab_mode online \
  > recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/nohup_train.log 2>&1 &
echo $! > recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/train.pid
echo "Training started in background."
echo "PID : $(cat recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/train.pid)"
echo "Log : recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/nohup_train.log"

# =========================== Step 3: inference ===========================
# 训练完成后合成一条测试音频；训练没跑完或没有 G_latest.pth 时注释掉。
.venv/bin/python cmd_inference.py \
  -m recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/G_latest.pth \
  -c recipes/ljspeech_en_16k_pretrain/data/ljspeech_en_16k/config.json \
  -o recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/infer \
  -l English \
  -t "Hello, this is an LJSpeech English IPA test." \
  -s LJSpeech \
  -on test
