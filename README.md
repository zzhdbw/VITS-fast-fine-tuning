# VITS Fast Fine-tuning

这个仓库现在有两条使用路线：

1. **原神中文/英文 16 kHz 预训练配方**：从原神语音数据整理、重采样、清洗文本，到训练和试听，都在 `recipes/genshin_zh_en_16k_pretrain/` 下。
2. **原来的 VITS 快速微调流程**：准备自己的音频或视频数据，基于 CJE / CJ / C 底模微调，然后做 TTS 或声线转换。

如果你是第一次用，建议先看第 1 条。它目前是仓库里最完整、最容易复现的一条链路。

## 路线一：原神中英 16 kHz 预训练

详细说明在：

```text
recipes/genshin_zh_en_16k_pretrain/README.md
```

这条路线大致做这些事：

- 扫描 `/mnt/afs/datasets/TTS/Genshin6.3` 里的 `Chinese/` 和 `English/`
- 每个音色目录只取一级目录下的 wav/lab，忽略子文件夹
- 中英同名音色合并成一个 speaker
- 重采样到 16 kHz 单声道 PCM16
- 清洗成 `[ZH]` / `[EN]` 标签的 IPA 文本
- 生成 16 kHz VITS 配置
- 支持 DDP、TensorBoard、SwanLab、gradient clipping、warmup
- 训练后可以用 `04_infer.py` 或 `infer.sh` 合成试听

常用入口：

```bash
bash recipes/genshin_zh_en_16k_pretrain/run.sh
```

也可以直接看 `run.sh` 里的 Step 1 到 Step 6，按需要注释或修改。

### 检查点对比与续训实验

训练中或训练后可以用这些脚本试听不同 checkpoint：

```bash
# 比较 G_10000.pth ~ G_56000.pth
bash recipes/genshin_zh_en_16k_pretrain/compare_ckpts.sh

# 比较后期 checkpoint 的中英混说效果
bash recipes/genshin_zh_en_16k_pretrain/compare_ckpts_mixed.sh
```

脚本顶部的 `MODEL_DIR`、`CONFIG`、`CKPTS` 可以按实际实验修改，输出会写到 `output/` 下（已 gitignore）。

低学习率续训实验说明见：

```text
recipes/genshin_zh_en_16k_pretrain/experiments/lr1e-4/README.md
```

## 路线二：原来的 VITS 快速微调

原来的脚本还在：

```text
finetune_speaker_v2.py
preprocess_v2.py
VC_inference.py
cmd_inference.py
```

适用场景：

- 有自己的角色音频或视频；
- 想基于 CJE / CJ / C 预训练底模微调；
- 想做 TTS 或声线转换。

准备数据看：

```text
DATA.MD
```

本地安装、编译和训练流程看：

```text
LOCAL.md
```

## 环境

系统依赖：

- ffmpeg
- cmake
- gcc / g++
- git

Python 依赖：

```bash
pip install -r requirements.txt
```

当前仓库里的 `.venv` 使用 Python 3.11。`requirements.txt` 已固定 `Cython==0.29.36`，避免 Python 3.11 下编译 `monotonic_align` 时提示 `longintrepr.h` 找不到。

PyTorch 建议单独装和机器 CUDA 匹配的 GPU 版本。

## 编译 monotonic_align

原来的快速微调流程需要：

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

## 训练日志

原来的 `finetune_speaker_v2.py` 现在同时支持：

- TensorBoard
- SwanLab

默认只开 TensorBoard。需要 SwanLab 时加：

```bash
--use_swanlab True
--swanlab_project vits-fast-fine-tuning
--swanlab_name genshin_zh_en_16k
--swanlab_mode local
```

`swanlab_mode` 可选：

```text
online    云端
local     本地
offline   离线
disabled  关闭 SwanLab
```

## 推理

原版的 Gradio 页面：

```bash
python VC_inference.py --model_dir ./OUTPUT_MODEL/G_latest.pth --config_dir ./OUTPUT_MODEL/config.json
```

命令行合成：

```bash
python cmd_inference.py -m 模型路径 -c config.json -o 输出目录 -t "你好" -s 音色名
```

16 kHz 配方自己的推理支持两种入口：

```bash
# 直接调用 Python
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/04_infer.py --speaker 魔女M --text "你好"

# 或使用包装脚本
bash recipes/genshin_zh_en_16k_pretrain/infer.sh --speaker 魔女M --text "你好"
```

## 文档

- `recipes/genshin_zh_en_16k_pretrain/README.md`：原神中英 16 kHz 预训练配方
- `DATA.MD`：微调数据格式
- `LOCAL.md`：本地环境、编译、训练和推理

英文 README 已移除，后续以中文文档为准。
