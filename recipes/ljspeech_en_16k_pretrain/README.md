# LJSpeech 英文单人 16 kHz VITS 预训练配方

本配方使用 `/mnt/afs/datasets/TTS/LJSpeech-1.1`，从零预训练一个**英文单人、输出采样率 16 kHz** 的 VITS 模型。

> 重要说明：本配方**不新增词表、不新增 cleaner**。
> 仓库里原本没有“纯 ASCII 英文”词表；对英文可直接复用的只有 `text/symbols.py` 中
> 原本激活的 68 个 IPA 音素符号，以及原本的 `cjke_cleaners2` cleaner。
> 因此本配方复用这一套原始词表/cleaner，生成的是英文 IPA 模型，而不是 ASCII 字符级模型。
> 如果必须使用纯 ASCII 英文字符词表，则需要新增 cleaner 或 symbols，这超出了“不新增”的要求。

数据集结构：

```text
/mnt/afs/datasets/TTS/LJSpeech-1.1/
├── metadata.csv          # id|transcription|normalized
└── wavs/
    ├── LJ001-0001.wav
    ├── LJ001-0002.wav
    └── ...
```

数据特点：

- 13100 条单人英文语音，原始采样率 22050 Hz，单声道 PCM16
- 总时长约 24 小时，适合做成英文单人底模
- 优先使用 `metadata.csv` 的 `normalized` 列；缺失时回退到 `transcription`
- 超过 `--max-duration`（默认 11 秒）的音频会被过滤

## 使用的原始词表

本配方直接使用仓库原本激活的 `text.symbols.symbols`，共 68 个符号：

```text
_ , . ! ? - ~ … N Q a b d e f g h i j k l m n o p s t u v w x y z
ɑ æ ʃ ʑ ç ɯ ɪ ɔ ɛ ɹ ð ə ɫ ɥ ɸ ʊ ɾ ʒ θ β ŋ ɦ ⁼ ʰ ` ^ # * = ˈ ˌ → ↓ ↑ 空格
```

对应 cleaner：

```text
cjke_cleaners2
```

预处理时脚本会把 LJSpeech 文本包成：

```text
[EN]<text>[EN]
```

然后调用原本的 `cjke_cleaners2`，让它走已有的 `english_to_ipa2` 路径，把英文转换成 IPA。
生成的 `config.json` 里 `symbols` 会被写成 `text.symbols.symbols`，保证和仓库原始词表完全一致；
不在该词表内的字符会被过滤，训练和推理行为一致。

## 目录说明

```text
recipes/ljspeech_en_16k_pretrain/
├── 00_prepare_ljspeech.py              # 解析 metadata、复用 cjke_cleaners2、重采样、切 train/val、生成 config
├── config_ljspeech_16k_template.json   # 16 kHz 英文单人 VITS 模板，symbols 为原始 68 音素
├── infer.sh                            # 单音色推理包装脚本
├── run.sh                              # 统一入口
├── README.md
└── .gitignore
```

运行后数据：

```text
recipes/ljspeech_en_16k_pretrain/data/ljspeech_en_16k/
├── wav/
│   ├── LJ001-0001.wav
│   └── ...
├── train.txt
├── val.txt
├── config.json
└── prepare_summary.json
```

训练输出：

```text
recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/
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
# 默认：准备数据 -> 后台训练 1000 epoch -> 测试推理
# 没有 G_latest.pth 时，请先注释掉 run.sh 的 Step 3
bash recipes/ljspeech_en_16k_pretrain/run.sh
```

可覆盖的常用环境变量：

```bash
# 使用 2,3 号卡
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=8010 \
  bash recipes/ljspeech_en_16k_pretrain/run.sh
```

`run.sh` 风格和 `recipes/bznsyp_zh_16k_pretrain/run.sh` 一致：Step 1/2/3 命令直接写在脚本里。
需要改 epoch 数或关闭 SwanLab 时，直接编辑 `run.sh` 里的 `--max_epochs` / `--use_swanlab`。

默认训练参数：

```text
CUDA_VISIBLE_DEVICES=6,7
MASTER_PORT=8003
max_epochs=1000
batch_size=16
```

## 数据预处理

单独运行：

```bash
.venv/bin/python recipes/ljspeech_en_16k_pretrain/00_prepare_ljspeech.py \
  --dataset-root /mnt/afs/datasets/TTS/LJSpeech-1.1 \
  --data-dir recipes/ljspeech_en_16k_pretrain/data/ljspeech_en_16k \
  --sampling-rate 16000 \
  --val-ratio 0.02 \
  --workers 16 \
  --skip-existing
```

脚本会：

1. 读取 `metadata.csv` 的 `id|transcription|normalized`，优先使用 `normalized` 文本
2. 给文本加 `[EN]` 标签，调用原本的 `cjke_cleaners2` 转成 IPA
3. 过滤掉不在 `text.symbols.symbols` 中的字符
4. 把 22050 Hz wav 重采样到 16 kHz，写成 PCM16
5. 随机切 98% train / 2% val
6. 生成 `train.txt`、`val.txt`、`config.json`、`prepare_summary.json`

快速验证只处理 3 条：

```bash
.venv/bin/python recipes/ljspeech_en_16k_pretrain/00_prepare_ljspeech.py \
  --data-dir /tmp/ljspeech_test \
  --max-files 3 \
  --workers 1
```

说明：`cjke_cleaners2` 内部会调用 `eng_to_ipa`，比纯文本 lower/去重音慢。
本配方把文本清洗和重采样都放进 worker 池并行执行，统一入口默认 `--workers 16`。

## 训练

`run.sh` 的 Step 2 实际执行的是：

```bash
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
```

说明：

- `--train_with_pretrained_model False`：不加载底模，从随机初始化开始训练
- `--max_epochs 1000`：训练 1000 轮，不想用 SwanLab 时把 `--use_swanlab` 改为 `False`
- `--preserved 100`：只保留最近 100 个带 step 的 checkpoint
- 默认每 `5000` 个 step 保存一次 checkpoint，可在模板 `train.eval_interval` 中修改

查看日志：

```bash
tail -f recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/nohup_train.log
```

停止训练：

```bash
kill "$(cat recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/train.pid)"
```

## 推理

训练完成后，推荐使用配方自带包装脚本：

```bash
bash recipes/ljspeech_en_16k_pretrain/infer.sh "Hello, this is an LJSpeech English test." test
```

输出：

```text
recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/infer/test.wav
```

也可以直接调用仓库原有的 `cmd_inference.py`：

```bash
.venv/bin/python cmd_inference.py \
  -m recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/G_latest.pth \
  -c recipes/ljspeech_en_16k_pretrain/data/ljspeech_en_16k/config.json \
  -o recipes/ljspeech_en_16k_pretrain/output/ljspeech_en_16k/infer \
  -l English \
  -t "Hello, this is an LJSpeech English test." \
  -s LJSpeech \
  -on test
```

`cmd_inference.py -l English` 会自动在文本两侧加 `[EN]`，和预处理时的标签方式一致，
因此训练和推理都会走原本的 `cjke_cleaners2` English → IPA 路径。
