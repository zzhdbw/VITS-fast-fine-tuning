#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/zzh/code/VITS-fast-fine-tuning

# ============================ Step 1: scan ============================
# 只扫描每个音色目录下直接存在的 wav，不扫描子文件夹。
# 整个音色的排除名单在 recipes/genshin_zh_en_16k_pretrain/exclude_speakers.txt 里改。
# 需要快速验证就改这里：--max-speakers 2 --max-utts-per-speaker 4
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/00_scan_voices.py \
  --dataset-root /mnt/afs/datasets/TTS/Genshin6.3 \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k \
  --exclude-file recipes/genshin_zh_en_16k_pretrain/exclude_speakers.txt \
  --min-utts 20 \
  --max-speakers 0 \
  --max-utts-per-speaker 500 \
  --val-ratio 0.02 


# ======================= Step 2: resample 16k =======================
# 快速验证就改 --max-files 20 --workers 2
# 只看处理计划、不写文件：在命令末尾加 --dry-run
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/01_resample_audio.py \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k \
  --target-sr 16000 \
  --min-duration 0.6 \
  --max-duration 16.0 \
  --workers 32 \
  --batch-size 10 \
  --max-files 0 \
  --skip-existing \
  # --dry-run   # 取消注释这行可只看计划，不写文件

# ======================== Step 3: clean text ========================
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/02_clean_text.py \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k \
  --workers 8

# ======================== Step 4: make config =======================
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/03_make_config.py \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k \
  --template recipes/genshin_zh_en_16k_pretrain/config_16k_template.json

# ========================== Step 5: pretrain ========================
# 后台 nohup 训练，日志写到本地：
#   recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/nohup_train.log
#   recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/train.pid
# 暂时不训练：把下面整段 Step 5 注释掉。
export CUDA_VISIBLE_DEVICES=0,1,4,5
mkdir -p recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k
nohup .venv/bin/python -u finetune_speaker_v2.py \
  -m recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k \
  -c recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k/config.json \
  --max_epochs 100 \
  --preserved 50 \
  --num_workers 8 \
  --grad_clip 500 \
  --warmup_steps 2000 \
  --train_with_pretrained_model False \
  --drop_speaker_embed False \
  > recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/nohup_train.log 2>&1 &
echo $! > recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/train.pid
echo "Training started in background."
echo "PID : $(cat recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/train.pid)"
echo "Log : recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/nohup_train.log"

# ========================== Step 6: inference ========================
# 用训练好的 checkpoint 合成音频试听。
# 如果不想一运行 run.sh 就推理，或训练还没跑完，把下面整段 Step 6 注释掉。
# 想固定某个 step，把 G_latest.pth 改成 G_5000.pth / G_10000.pth 等。
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/04_infer.py \
  --checkpoint recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/G_latest.pth \
  --speaker 丝柯克 \
  --language auto \
  --text "你好，这是一段测试语音。"
