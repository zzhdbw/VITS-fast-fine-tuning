# LJSpeech + BZNSYP 中英双语 16 kHz VITS 预训练配方

本配方同时使用 `/mnt/afs/datasets/TTS/LJSpeech-1.1` 和
`/mnt/afs/datasets/TTS/BZNSYP`，从零预训练一个**中英双语、双说话人、输出采样率 16 kHz**
的 VITS 模型。

数据与说话人映射：

```text
LJSpeech-1.1  ->  英文单人  ->  speaker 0, name = LJSpeech
BZNSYP        ->  中文单人  ->  speaker 1, name = BZNSYP
```

两个数据集共用同一套原始文本前端，不新增 cleaner、不新增词表：

- cleaner：`text.cleaners.cjke_cleaners2`
- vocabulary：`text.symbols.symbols`，即仓库原本激活的 68 个 IPA 音素符号

LJSpeech 文本会加 `[EN]` 标签，BZNSYP 文本会加 `[ZH]` 标签，然后统一送入
`cjke_cleaners2`：

```text
LJSpeech 文本 -> [EN]... [EN] -> english_to_ipa2 -> 68 音素 IPA
BZNSYP   文本 -> [ZH]... [ZH] -> chinese_to_ipa  -> 68 音素 IPA
```

因此两个语言的清洗结果落在同一个符号空间里，可以共享一个模型训练。

## 数据集结构

LJSpeech-1.1：

```text
/mnt/afs/datasets/TTS/LJSpeech-1.1/
├── metadata.csv          # id|transcription|normalized
└── wavs/
    ├── LJ001-0001.wav
    ├── LJ001-0002.wav
    └── ...
```

BZNSYP：

```text
/mnt/afs/datasets/TTS/BZNSYP/
├── Wave/
│   ├── 000001.wav
│   ├── 000002.wav
│   └── ...
├── ProsodyLabeling/
│   └── 000001-010000.txt
└── PhoneLabeling/
    └── ...
```

预处理脚本会：

1. 读取 LJSpeech `metadata.csv`，优先使用 `normalized` 列，缺失时回退到 `transcription`；
2. 读取 BZNSYP `ProsodyLabeling/000001-010000.txt`，去掉 `#1`、`#2`、`#3`、`#4` 等韵律标记；
3. 分别加 `[EN]` / `[ZH]` 标签，调用仓库原本的 `cjke_cleaners2` 转成 IPA；
4. 过滤掉不在原始 68 音素词表中的字符，训练和推理行为一致；
5. 把 LJSpeech 的 22050 Hz wav、BZNSYP 的 48000 Hz wav 统一重采样到 16 kHz 单声道 PCM16；
6. 按数据集分别切分训练集/验证集，默认每个数据集约 2% 作为验证集；
7. 生成统一的 `train.txt`、`val.txt`、`config.json`、`prepare_summary.json`。

## 目录说明

```text
recipes/ljspeech_bznsyp_zh_en_16k_pretrain/
├── 00_prepare_ljspeech_bznsyp.py         # 解析两个数据集、清洗文本、重采样、切分、生成 config
├── config_ljspeech_bznsyp_16k_template.json
├── run.sh                                # 统一入口：准备数据 -> 后台训练 -> 可选推理
├── infer.sh                              # 单条文本推理包装脚本
├── infer_mixed_auto.py                   # 自动给中英混说加 [ZH]/[EN] 标签
├── README.md
├── data/                                 # 运行后生成的数据（已 gitignore）
└── output/                               # checkpoint / 日志 / 推理结果（已 gitignore）
```

运行后数据目录：

```text
recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/
├── wav/
│   ├── ljspeech/
│   │   ├── LJ001-0001.wav
│   │   └── ...
│   └── bznsyp/
│       ├── 000001.wav
│       └── ...
├── train.txt
├── val.txt
├── config.json
└── prepare_summary.json
```

训练输出目录：

```text
recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/
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
# 默认：准备数据 -> 后台训练 1000 epoch -> 如果已有 G_latest.pth 则追加推理
bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/run.sh
```

`run.sh` 和仓库里其他配方一样，把 Step 1/2/3 命令直接写在脚本里：

- Step 1：准备 LJSpeech + BZNSYP 数据
- Step 2：后台启动 1000 epoch 预训练
- Step 3：若存在 `G_latest.pth` 则合成中英测试音频；否则自动跳过并打印后续推理命令

如果只想做预处理，把 Step 2 和 Step 3 整段注释掉即可。

可覆盖常用环境变量：

```bash
# 使用 2,3 号卡，并修改 DDP 端口
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=8010 \
  bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/run.sh
```

`run.sh` 顶部已经显式写好两个数据集路径：

```bash
LJSPEECH_ROOT="/mnt/afs/datasets/TTS/LJSpeech-1.1"
BZNSYP_ROOT="/mnt/afs/datasets/TTS/BZNSYP"
```

需要换数据位置时，可以直接修改 `run.sh` 中的变量，或在外部用同名环境变量覆盖。

默认训练参数：

```text
CUDA_VISIBLE_DEVICES=6,7
MASTER_PORT=9104
max_epochs=1000
batch_size=16（在模板中设置）
```

不想用 SwanLab 时，直接编辑 `run.sh`，把 `--use_swanlab True` 改成 `False`。
需要减少 epoch 数做快速实验时，把 `run.sh` 里的 `--max_epochs 1000` 改小。

## 数据预处理

单独运行：

```bash
.venv/bin/python recipes/ljspeech_bznsyp_zh_en_16k_pretrain/00_prepare_ljspeech_bznsyp.py \
  --ljspeech-root /mnt/afs/datasets/TTS/LJSpeech-1.1 \
  --bznsyp-root /mnt/afs/datasets/TTS/BZNSYP \
  --data-dir recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k \
  --sampling-rate 16000 \
  --val-ratio 0.02 \
  --max-duration 16.0 \
  --workers 16 \
  --skip-existing
```

快速验证只处理每个数据集各 3 条：

```bash
.venv/bin/python recipes/ljspeech_bznsyp_zh_en_16k_pretrain/00_prepare_ljspeech_bznsyp.py \
  --data-dir /tmp/ljspeech_bznsyp_test \
  --max-files-per-dataset 3 \
  --workers 1
```

也可以为兼容其他配方使用 `--max-files 3`，它等价于
`--max-files-per-dataset 3`。

常用参数说明：

- `--sampling-rate 16000`：本配方固定输出 16 kHz，传其他会直接报错；
- `--val-ratio 0.02`：每个数据集内部单独切分验证集；
- `--min-duration 0.5`、`--max-duration 16.0`：过滤过短或过长的音频；
- `--max-cleaned-length 190`：过滤清洗后过长的文本，同时写入 `config.data.max_text_len`；
- `--skip-existing`：复用已经重采样好的 `wav/`，适合中断后继续跑；
- `--workers 16`：文本清洗和重采样并行度。

## 训练

`run.sh` 的 Step 2 实际执行的是：

```bash
nohup .venv/bin/python -u finetune_speaker_v2.py \
  -m recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k \
  -c recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json \
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
  > recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/nohup_train.log 2>&1 &
```

说明：

- `--train_with_pretrained_model False`：不加载底模，从随机初始化开始训练；
- `--max_epochs 1000`：完整预训练轮数；
- `--preserved 100`：只保留最近 100 个带 step 的 checkpoint；
- 默认每 `5000` 个 step 保存一次 checkpoint，可在模板 `train.eval_interval` 中修改；
- 训练出的模型有 2 个 speaker embedding，分别对应 `LJSpeech` 和 `BZNSYP`；
- 默认开启 SwanLab，项目名 `vits-fast-fine-tuning-pretrain`。

查看日志：

```bash
tail -f recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/nohup_train.log
```

停止训练：

```bash
kill "$(cat recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/train.pid)"
```

## 推理

### 使用包装脚本

英文使用 LJSpeech 音色：

```bash
bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer.sh \
  "Hello, this is an LJSpeech English test." \
  lj_test English LJSpeech
```

中文使用 BZNSYP 音色：

```bash
bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer.sh \
  "你好，这是 BZNSYP 中文测试语音。" \
  bz_test 简体中文 BZNSYP
```

中英混读有两种方式。

方式一：手动写好 `[ZH]` / `[EN]` 标签，并用 `Mix` 告诉 `cmd_inference.py` 不要再自动加语言标签：

```bash
bash recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer.sh \
  "[ZH]你好[ZH][EN] hello world [EN]" \
  mix_test Mix LJSpeech
```

方式二：使用 `infer_mixed_auto.py` 自动给中英片段加标签，文本按正常写法即可：

```bash
.venv/bin/python recipes/ljspeech_bznsyp_zh_en_16k_pretrain/infer_mixed_auto.py \
  --model_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/G_latest.pth" \
  --config_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json" \
  --output_path "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/infer" \
  --text "你好，这是中英混说测试。Hello, this is a mixed test." \
  --spk LJSpeech \
  --output_name "mixed_zh_en"
```

`run.sh` 的 Step 3 已经使用这种方式自动处理中英混说。

输出默认写到：

```text
recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/infer/
```

### 直接调用仓库原版 `cmd_inference.py`

英文：

```bash
.venv/bin/python cmd_inference.py \
  -m recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/G_latest.pth \
  -c recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json \
  -o recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/infer \
  -l English \
  -t "Hello, this is a bilingual test." \
  -s LJSpeech \
  -on test_en
```

中文：

```bash
.venv/bin/python cmd_inference.py \
  -m recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/G_latest.pth \
  -c recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json \
  -o recipes/ljspeech_bznsyp_zh_en_16k_pretrain/output/ljspeech_bznsyp_zh_en_16k/infer \
  -l 简体中文 \
  -t "你好，这是中英双语测试。" \
  -s BZNSYP \
  -on test_zh
```

### 查看可用音色

```bash
.venv/bin/python - <<'PY'
from utils import get_hparams_from_file
hps = get_hparams_from_file(
    "recipes/ljspeech_bznsyp_zh_en_16k_pretrain/data/ljspeech_bznsyp_zh_en_16k/config.json"
)
print(hps.speakers)
PY
```

输出应类似：

```text
{'LJSpeech': 0, 'BZNSYP': 1}
```

## 备注

- 默认输出采样率是 `16000 Hz`；
- 本配方使用 2 个说话人。英文默认用 `LJSpeech` 音色，中文默认用 `BZNSYP` 音色；
- 两个数据集都不是彼此语言的母语数据，跨语言合成（例如用 BZNSYP 音色说英文）效果取决于训练程度，建议按对应语言选择音色；
- 数据量合计约 35 小时，1000 epoch 比较充分；快速验证时把 `run.sh` 里的 `--max_epochs` 改小；
- 如果显存较小，可以把 `config_ljspeech_bznsyp_16k_template.json` 中 `train.batch_size` 从 `16` 调小；
- 数据、checkpoint、日志和推理结果都已在 `.gitignore` 中忽略，不会提交到 Git。
