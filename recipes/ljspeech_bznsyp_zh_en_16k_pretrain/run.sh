#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="recipes/ljspeech_bznsyp_zh_en_16k_pretrain"
DATA_DIR="$RECIPE_DIR/data/ljspeech_bznsyp_zh_en_16k"
OUT_DIR="$RECIPE_DIR/output/ljspeech_bznsyp_zh_en_16k"

# ========================== Step 1: prepare data ==========================
# 同时解析 LJSpeech-1.1 和 BZNSYP，统一复用仓库原始 cjke_cleaners2 + 68 音素词表：
#   LJSpeech 文本 -> [EN] -> IPA
#   BZNSYP   文本 -> [ZH] -> IPA
# 音频统一重采样到 16 kHz 单声道 PCM16，并生成 train.txt / val.txt / config.json。
# 已经准备过时，--skip-existing 会复用已有 wav。
# 快速验证可以加：--max-files-per-dataset 3
.venv/bin/python "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/00_prepare_ljspeech_bznsyp.py" \
  --ljspeech-root "/mnt/afs/datasets/TTS/LJSpeech-1.1" \
  --bznsyp-root "/mnt/afs/datasets/TTS/BZNSYP" \
  --data-dir "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k" \
  --sampling-rate 16000 \
  --val-ratio 0.02 \
  --max-duration 16.0 \
  --workers 16 \
  --skip-existing

# ======================= Step 2: pretrain 1000 epochs =======================
# 后台训练，日志写到：
#   $OUT_DIR/nohup_train.log
#   $OUT_DIR/train.pid
# 暂时不训练：把下面整段 Step 2 注释掉。
# 默认开启 SwanLab，项目名 vits-fast-fine-tuning-pretrain。
# 默认使用 6,7 号卡；也可以这样覆盖：
#   CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=8010 bash "$RECIPE_DIR/run.sh"
export CUDA_VISIBLE_DEVICES="4,5"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="9105"
mkdir -p "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k"
nohup python finetune_speaker_v2.py \
  -m "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k" \
  -c "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json" \
  --max_epochs 1000 \
  --preserved 100 \
  --num_workers 8 \
  --grad_clip 500 \
  --warmup_steps 1000 \
  --train_with_pretrained_model False \
  --drop_speaker_embed False \
  --use_swanlab True \
  --swanlab_project vits-fast-fine-tuning-pretrain \
  --swanlab_name ljspeech_bznsyp_zh_en_16k \
  --swanlab_mode online \
  > "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/nohup_train.log" 2>&1 &
echo $! > "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/train.pid"
echo "Training started in background."

# =========================== Step 3: inference ===========================
# 训练完成后合成中/英测试音频；没有 G_latest.pth 时自动跳过。
if [[ -f "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/G_latest.pth" ]]; then
  .venv/bin/python cmd_inference.py \
    --model_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/G_latest.pth" \
    --config_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json" \
    --output_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/infer" \
    --language English \
    --text "Hello, this is a bilingual LJSpeech and BZNSYP test." \
    --spk LJSpeech \
    --output_name "ljspeech_en"

  .venv/bin/python cmd_inference.py \
    --model_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/G_latest.pth" \
    --config_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json" \
    --output_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/infer" \
    --language 简体中文 \
    --text "你好，这是 LJSpeech 和 BZNSYP 中英双语模型的测试语音。" \
    --spk BZNSYP \
    --output_name "bznsyp_zh"

  # 中英混说：infer_mixed_auto.py 会自动给中文/英文片段加 [ZH]/[EN] 标签，
  # 然后调用原版 cmd_inference.py 完成推理。
  # 这里默认使用 LJSpeech 音色，也可以按需改成 BZNSYP。
  .venv/bin/python "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer_mixed_auto.py" \
    --model_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/G_latest.pth" \
    --config_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json" \
    --output_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/infer" \
    --text "你好，这是中英混说测试。Hello, this is a mixed Chinese and English test." \
    --spk BZNSYP \
    --output_name "mixed_zh_en"
else
  echo "Skip inference: G_latest.pth not found yet."
  echo "After training saves G_latest.pth, run Step 3 inference manually."
fi
