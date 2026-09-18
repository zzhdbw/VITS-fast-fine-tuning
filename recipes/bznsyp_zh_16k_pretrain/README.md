# BZNSYP 中文单人 16 kHz VITS 预训练配方

本配方使用 `/mnt/afs/datasets/TTS/BZNSYP`，从零预训练一个**中文单人、输出采样率 16 kHz**的 VITS 模型，默认训练 `1000` 个 epoch。

数据集结构：

```text
/mnt/afs/datasets/TTS/BZNSYP/
├── Wave/
│   ├── 000001.wav
│   ├── 000002.wav
│   └── ...
├── ProsodyLabeling/
│   └── 000001-010000.txt
└── PhoneLabeling/
    └── 000001.interval
```

数据特点：

- 10000 条 wav，单声道，原始采样率 48 kHz，PCM16
- 全部为同一说话人，适合做单人音色预训练
- 文本在 `ProsodyLabeling/000001-010000.txt` 中
- 文本里的 `#1`、`#2`、`#3`、`#4` 是韵律标记，预处理时会去掉

## 目录说明

```text
recipes/bznsyp_zh_16k_pretrain/
├── 00_prepare_bznsyp.py              # 解析标签、重采样、切分 train/val、生成 config.json
├── config_bznsyp_16k_template.json   # 16 kHz 单人 VITS 配置模板
├── infer.sh                          # 单音色推理包装脚本
├── run.sh                            # 统一入口
├── README.md
├── data/                             # 运行后生成的数据（已 gitignore）
└── output/                           # checkpoint / 日志 / 推理结果（已 gitignore）
```

输出数据：

```text
recipes/bznsyp_zh_16k_pretrain/data/bznsyp_zh_16k/
├── wav/
│   ├── 000001.wav
│   └── ...
├── train.txt
├── val.txt
├── config.json
└── prepare_summary.json
```

训练输出：

```text
recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/
├── G_latest.pth
├── D_latest.pth
├── G_xxxxx.pth
├── D_xxxxx.pth
├── nohup_train.log
└── train.pid
```

## 快速开始

在仓库根目录执行：

```bash
# 默认按 Step 1~3 执行：准备数据 -> 1000 epoch 后台训练 -> 测试推理
bash recipes/bznsyp_zh_16k_pretrain/run.sh
```

`run.sh` 按下面的顺序直接写命令，和原神配方的风格一致：

- Step 1：准备 BZNSYP 数据，默认重采样到 16 kHz
- Step 2：后台启动 1000 epoch 预训练
- Step 3：训练完成后用 `G_latest.pth` 合成测试音频

如果只想做其中一步，打开 `run.sh`，把不需要的 Step 整段注释掉即可。

默认训练使用：

```text
CUDA_VISIBLE_DEVICES=6,7
MASTER_PORT=8002
```

如果显卡不同，可以覆盖：

```bash
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=8010   bash recipes/bznsyp_zh_16k_pretrain/run.sh
```

## 数据预处理

单独运行：

```bash
.venv/bin/python recipes/bznsyp_zh_16k_pretrain/00_prepare_bznsyp.py \
  --dataset-root /mnt/afs/datasets/TTS/BZNSYP \
  --data-dir recipes/bznsyp_zh_16k_pretrain/data/bznsyp_zh_16k \
  --sampling-rate 16000 \
  --val-ratio 0.02 \
  --workers 16 \
  --skip-existing
```

脚本会：

1. 读取 `ProsodyLabeling/000001-010000.txt`
2. 去掉文本里的 `#1` ~ `#4` 韵律标记
3. 直接调用仓库已有的 `chinese_cleaners` 清洗成注音符号（Bopomofo）
4. 过滤到纯中文注音符号表（约 50 个符号），不使用 `cjke_cleaners2` 的多语言 IPA 词表
5. 把 48 kHz wav 重采样到 16 kHz，写成 PCM16
6. 随机切 98% train / 2% val
7. 生成 `train.txt`、`val.txt`、`config.json`、`prepare_summary.json`

快速验证只处理 3 条：

```bash
.venv/bin/python recipes/bznsyp_zh_16k_pretrain/00_prepare_bznsyp.py \
  --data-dir /tmp/bznsyp_test \
  --max-files 3 \
  --workers 1
```

## 训练

`run.sh` 的 Step 2 实际执行的是：

```bash
.venv/bin/python -u finetune_speaker_v2.py \
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
  --swanlab_mode online
```

说明：

- `--train_with_pretrained_model False`：不加载 16 kHz 预训练底模，从随机初始化开始训练
- `--max_epochs 1000`：训练 1000 轮
- `--preserved 100`：只保留最近 100 个带 step 的 checkpoint
- 默认每 `5000` 个 step 保存一次 checkpoint，可在模板里改 `train.eval_interval`
- Step 2 默认开启 SwanLab，项目名为 `vits-fast-fine-tuning-pretrain`，运行名为 `bznsyp_zh_16k`
- 如果暂时不想用 SwanLab，把命令里的 `--use_swanlab True` 改成 `False`

查看日志：

```bash
tail -f recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/nohup_train.log
```

停止训练：

```bash
kill "$(cat recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/train.pid)"
```

## 推理

训练完成后，推荐直接使用配方自带的包装脚本：

```bash
bash recipes/bznsyp_zh_16k_pretrain/infer.sh "你好，欢迎使用 BZNSYP 中文单人音色。" test
```

输出：

```text
recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/infer/test.wav
```

也可以直接调用仓库原有的 `cmd_inference.py`：

```bash
.venv/bin/python cmd_inference.py \
  -m recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/G_latest.pth \
  -c recipes/bznsyp_zh_16k_pretrain/data/bznsyp_zh_16k/config.json \
  -o recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/infer \
  -l 简体中文 \
  -t "你好，欢迎使用 BZNSYP 中文单人音色。" \
  -s BZNSYP \
  -on test
```

输出：

```text
recipes/bznsyp_zh_16k_pretrain/output/bznsyp_zh_16k/infer/test.wav
```

## 备注

- 默认输出采样率是 `16000 Hz`。
- 数据量约 11.9 小时，1000 epoch 是比较充分的训练轮数。
- 如果你的机器显存较小，可以把 `config_bznsyp_16k_template.json` 里的 `train.batch_size` 从 `16` 调小。
- 训练数据、checkpoint、日志和推理结果都已经在 `.gitignore` 中忽略，不会提交到 Git。
