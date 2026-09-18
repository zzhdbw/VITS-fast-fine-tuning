# 本地训练

这里写当前仓库在本机环境下从零到能训练的最短路径。

## 1. 系统依赖

确认有：

```bash
ffmpeg -version
cmake --version
gcc --version
g++ --version
```

没有的话先装系统包。Ubuntu 一般：

```bash
sudo apt install ffmpeg cmake build-essential
```

## 2. Python 环境

推荐直接用仓库里的 uv 虚拟环境：

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

如果 `Cython==0.29.21` 在 Python 3.11 下编译失败，直接升到较新的 0.29.x：

```bash
uv pip install "Cython==0.29.36"
```

PyTorch 单独装和 CUDA 匹配的 GPU 版本。装完后确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` 必须是 `True`。

## 3. 编译 monotonic_align

```bash
cd monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd ..
```

验证：

```bash
python -c "from monotonic_align import maximum_path; print('ok')"
```

## 4. 路线一：原神中英 16 kHz 预训练

这条路线不需要 `pretrained_models/` 里的 CJE / CJ / C 底模，脚本会从随机初始化开始训练。

直接看：

```text
recipes/genshin_zh_en_16k_pretrain/README.md
```

最短流程：

```bash
cd /mnt/afs/zzh/code/VITS-fast-fine-tuning
bash recipes/genshin_zh_en_16k_pretrain/run.sh
```

如果只想做数据处理，不想训练，编辑 `run.sh`，把 Step 5 整段注释掉。  
如果只想训练，把 Step 1~4 和 Step 6 注释掉，保留 Step 5。

训练日志：

- TensorBoard：`tensorboard --logdir recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k`
- nohup 输出：`recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/nohup_train.log`
- SwanLab：在训练命令里加 `--use_swanlab True --swanlab_mode local`

## 5. 路线二：原来的 VITS 快速微调

先准备自己的数据，格式见 `DATA.MD`。

### 下载底模

可用底模：

```text
CJE：中、日、英
CJ ：中、日
C ：中
```

下载 CJE 示例：

```bash
mkdir -p pretrained_models
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/configs/uma_trilingual.json -O ./configs/finetune_speaker.json
```

CJ：

```bash
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0-p.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0-p.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/config.json -O ./configs/finetune_speaker.json
```

C：

```bash
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/D_0.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/G_0.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/config.json -O ./configs/finetune_speaker.json
```

下载不同底模时会覆盖前一组文件。

### 训练

数据处理：

```bash
python scripts/video2audio.py
python scripts/denoise_audio.py
python scripts/long_audio_transcribe.py --languages CJE --whisper_size large
python scripts/short_audio_transcribe.py --languages CJE --whisper_size large
python scripts/resample.py
python preprocess_v2.py --add_auxiliary_data True --languages CJE
```

训练：

```bash
python finetune_speaker_v2.py \
  -m ./OUTPUT_MODEL \
  -c ./configs/modified_finetune_speaker.json \
  --max_epochs 100 \
  --train_with_pretrained_model True \
  --drop_speaker_embed False
```

继续训练：

```bash
python finetune_speaker_v2.py \
  -m ./OUTPUT_MODEL \
  -c ./OUTPUT_MODEL/config.json \
  --max_epochs 300 \
  --train_with_pretrained_model True \
  --drop_speaker_embed False \
  --cont True
```

注意：

- `--max_epochs` 必须大于 checkpoint 里已经保存的 epoch，否则会立刻停止。
- `--cont True` 会加载 `G_latest.pth` 和 `D_latest.pth`，但不会恢复 Adam 优化器状态。

## 6. 推理

Gradio：

```bash
python VC_inference.py \
  --model_dir ./OUTPUT_MODEL/G_latest.pth \
  --config_dir ./OUTPUT_MODEL/config.json
```

命令行：

```bash
python cmd_inference.py \
  -m ./OUTPUT_MODEL/G_latest.pth \
  -c ./OUTPUT_MODEL/config.json \
  -o ./output \
  -l 简体中文 \
  -t "你好，这是一段测试语音。" \
  -s 角色名
```

16 kHz 配方：

```bash
bash recipes/genshin_zh_en_16k_pretrain/infer.sh \
  --speaker 魔女M \
  --text "你好，这是一段测试语音。"
```
