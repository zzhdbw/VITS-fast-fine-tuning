# Genshin 中英 16 kHz VITS 预训练配方

本配方使用 `/mnt/afs/datasets/TTS/Genshin6.3` 下的 `Chinese` 和 `English` 原神语音，
从零预训练一个**中英双语、输出采样率 16 kHz** 的 VITS 模型。

数据目录结构：

```text
/mnt/afs/datasets/TTS/Genshin6.3/
├── Chinese/
│   └── <音色>/
│       ├── xxx.wav + xxx.lab
│       └── <子目录>/          # 忽略
│           └── yyy.wav + yyy.lab
└── English/
    └── <音色>/
        └── ...
```

每个 `xxx.lab` 是同一文件名的 `.wav` 对应文本。脚本会把音色目录当作说话人，
只收集音色目录下**直接存在**的 wav/lab 对，所有子文件夹全部忽略。

## 目录说明

```text
recipes/genshin_zh_en_16k_pretrain/
├── common.py                  # 公共常量/小工具
├── 00_scan_voices.py          # 扫描音色、过滤、切 train/val、生成 speakers.json
├── 01_resample_audio.py       # 重采样到 16 kHz 单声道 PCM16
├── 02_clean_text.py           # 清理 .lab 文本，生成 train.txt / val.txt
├── 03_make_config.py          # 生成最终 config.json
├── 04_infer.py                # 训练后合成音频试听
├── config_16k_template.json   # 16 kHz VITS 配置模板
├── exclude_speakers.txt       # 可编辑的音色排除名单（fnmatch 通配符）
├── run.sh                     # 统一入口脚本
└── data/                      # 运行后生成的数据（已 gitignore）
```

## 使用 uv 虚拟环境

默认使用仓库里的 uv 虚拟环境：

```text
/mnt/afs/zzh/code/VITS-fast-fine-tuning/.venv/bin/python
```

`run.sh` 里每条命令都直接写 `.venv/bin/python`；需要换环境时，直接改命令里的 Python 路径即可。

## 逐步单独运行

### 1. 扫描音色

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/00_scan_voices.py \
  --dataset-root /mnt/afs/datasets/TTS/Genshin6.3 \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k \
  --min-utts 5
```

输出：

- `scan_rows.jsonl`：所有保留的 wav/lab 对
- `speakers.json`：`音色名` → 整数 speaker id；中英目录下同名音色会合并为同一个 speaker
- `scan_summary.json`

扫描时会自动读取可编辑排除文件 `exclude_speakers.txt`：

- 匹配音色目录名或 `语言:音色名`；
- 支持 `*`、`?`、`[]` 通配符，每行一个规则，`#` 开头是注释；
- 默认已经排除了 `#Unknown`、众人、群众、观众、路人、旁白、杂音等音色；
- 后续发现新的混合音色，直接往这个文件里加一行即可。

快速验证：

```bash
.venv/bin/python .../00_scan_voices.py --max-speakers 2 --min-utts 1 \
  --data-dir /tmp/gs_scan_test
```

### 2. 重采样到 16 kHz

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/01_resample_audio.py \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k \
  --workers 8
```

输出：

- `wav/zh/<音色>/.../*.wav`
- `wav/en/<音色>/.../*.wav`
- `resampled_rows.jsonl`
- `resample_summary.json`

这是最慢的一步，会读取并重采样所有音频。快速验证只处理前 20 条：

```bash
.venv/bin/python .../01_resample_audio.py \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k \
  --max-files 20 --workers 2
```

如果要中断后继续，可加：

```bash
--skip-existing
```

### 3. 清理文本

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/02_clean_text.py \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k \
  --workers 8
```

输出：

- `train.txt`、`val.txt`：`wav路径|speaker_id|清洗后文本`
- `cleaned_rows.jsonl`
- `clean_summary.json`

清理使用仓库已有的 `cjke_cleaners2`，对中文加 `[ZH]`、英文加 `[EN]`。

### 4. 生成配置

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/03_make_config.py \
  --data-dir recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k
```

输出：

- `config.json`

其中：

- `data.sampling_rate = 16000`
- `data.n_speakers = max(speaker_id) + 1`
- `speakers` 写入完整音色映射
- `train.txt` / `val.txt` 使用绝对路径

## 使用统一入口

直接修改 `run.sh` 里各 Step 命令上的参数，然后运行：

```bash
cd /mnt/afs/zzh/code/VITS-fast-fine-tuning
bash recipes/genshin_zh_en_16k_pretrain/run.sh
```

`run.sh` 已经按步骤组织：

```text
Step 1: scan      扫描音色、切 train/val
Step 2: resample  重采样到 16 kHz
Step 3: clean     清洗 .lab 文本，生成 train.txt/val.txt
Step 4: config    生成 config.json
Step 5: pretrain  从零预训练
```

如果暂时不训练，把 `Step 5: pretrain` 整段注释掉即可。

每个 Step 的参数都直接写在命令里，例如：

- Step 1 修改 `--min-utts`、`--max-speakers`、`--max-utts-per-speaker`
- Step 2 修改 `--workers`、`--max-files`，需要断点续跑就加 `--skip-existing`
- Step 3 修改 `--workers`
- Step 5 修改 `--max_epochs`

如果只想验证某一步，也可以直接运行对应的 Python 脚本。

## 推理合成

训练保存的 checkpoint 在：

```text
recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/G_latest.pth
```

查看可用音色：

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/04_infer.py --list-speakers
```

合成中文：

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/04_infer.py \
  --checkpoint recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/G_latest.pth \
  --config recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k/config.json \
  --speaker 魔女M \
  --language zh \
  --text "你好，这是测试语音。"
```

合成英文：

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/04_infer.py \
  --speaker 魔女M \
  --language en \
  --text "Hello, this is a test."
```

中英混说（默认 `--language auto`，会自动给中文/英文加标签）：

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/04_infer.py \
  --speaker 魔女M \
  --language auto \
  --text "你好，这是 hello world 的测试。"
```

`--language mix` 表示不加任何标签，适合你已经在文本里手写好 `[ZH]... [ZH]`、`[EN]... [EN]` 的情况。

也可以用 `infer.sh`：

```bash
bash recipes/genshin_zh_en_16k_pretrain/infer.sh --speaker 魔女M --language zh --text "你好"
```

批量文本文件：

```bash
.venv/bin/python recipes/genshin_zh_en_16k_pretrain/04_infer.py \
  --speaker 魔女M --language zh --text-file my_texts.txt
```

默认输出目录：

```text
recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k/infer/
```

生成音频采样率由 config 决定，当前是：

```text
16000 Hz
```

如果训练正在写 `G_latest.pth`，推理时读 checkpoint 可能偶发不完整，建议改用已经写好的 `G_xxx.pth`，或者等一次 eval 保存完成后再试。

## 训练

Step 5 实际执行：

```bash
python finetune_speaker_v2.py \
  -m recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k \
  -c recipes/genshin_zh_en_16k_pretrain/data/genshin_zh_en_16k/config.json \
  --max_epochs 100 \
  --preserved 50 \
  --num_workers 8 \
  --grad_clip 500 \
  --warmup_steps 2000 \
  --train_with_pretrained_model False \
  --drop_speaker_embed False
```

新增训练稳定项：

- `--grad_clip 500`：按梯度范数裁剪，抑制大尖峰。
- `--warmup_steps 2000`：前 2000 个 optimizer step 线性 warmup。
- `--num_workers 8`：DataLoader worker 数；大规模数据时比默认 2 更不容易卡 I/O。
- `config.json` 里的 `train.eval_audio_samples`：每次 eval 记录多少个 `gen/audio` / `gt/audio` 样本，默认 5。

因为这里指定的是：

```text
--train_with_pretrained_model False
```

所以是**从随机初始化开始预训练**，不是加载已有的 22.05 kHz 预训练模型微调。

## 16 kHz 输出保证

最终模型输出采样率由 `config.json` 决定：

```json
{
  "data": {
    "sampling_rate": 16000,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024
  }
}
```

推理时 `VC_inference.py` 返回的音频采样率就是 `hps.data.sampling_rate`，即 16 kHz。

## 注意事项

- 默认过滤掉 `#Unknown` 音色，因为其中通常是多个说话人混合。
- 默认保留至少 5 条音频的音色；可以通过 `--min-utts` 调整。
- 默认只保留 0.6 秒到 16 秒的音频。16 秒上限是因为当前训练代码的 bucket boundary 最大为 1000 spec frames；16 kHz、hop=256 时 1000 帧约等于 16 秒。
- 原始数据是 48 kHz，`01_resample_audio.py` 会重采样到 16 kHz；重采样后的文件会占用较多磁盘空间。
- 训练需要 GPU；当前训练代码会 `assert torch.cuda.is_available()`。
