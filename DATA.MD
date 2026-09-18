# 数据准备

这份文档主要说两条路线要准备什么数据。

## 一、原神中英 16 kHz 预训练配方

这条路线不需要自己整理 zip，直接把数据集放在：

```text
/mnt/afs/datasets/TTS/Genshin6.3/
├── Chinese/
│   └── <音色>/
│       ├── xxx.wav
│       ├── xxx.lab
│       └── <子目录>/        # 会被忽略
└── English/
    └── <音色>/
        └── ...
```

规则：

- 每个音色目录当成一个 speaker；
- 只扫描音色目录下直接存在的 `.wav` / `.lab`，子目录全部忽略；
- 中英同名音色会合并成同一个 speaker；
- `.lab` 是同一文件名 `.wav` 的文本；
- 数据路径、排除名单、重采样、清洗都由 `recipes/genshin_zh_en_16k_pretrain/` 下的脚本处理。

具体命令看：

```text
recipes/genshin_zh_en_16k_pretrain/README.md
```

## 二、原来的 VITS 快速微调

支持四种数据形式，可以只选一种，也可以混用。

### 1. 短音频 zip

结构：

```text
custom_character_voice.zip
├── 角色A/
│   ├── 001.wav
│   ├── 002.mp3
│   └── ...
├── 角色B/
│   ├── 001.wav
│   └── ...
└── ...
```

要求：

- 每条 2 秒以上、10 秒以内；
- 尽量没有背景噪音；
- 每个角色至少 10 条，20 条以上更好；
- 音频格式不限制。

用法：

```bash
mkdir -p custom_character_voice
unzip custom_character_voice.zip -d custom_character_voice/
```

### 2. 长音频

命名：

```text
{CharacterName}_{random_number}.wav
```

例如：

```text
Diana_234135.wav
MinatoAqua_234252.wav
```

要求：

- 必须是 `.wav`；
- 单个文件只能有一个说话人；
- 建议 20 分钟以内；
- 背景音会被自动处理。

放到：

```text
raw_audio/
```

### 3. 长视频

命名：

```text
{CharacterName}_{random_number}.mp4
```

例如：

```text
Taffy_332452.mp4
Dingzhen_957315.mp4
```

要求：

- 必须是 `.mp4`；
- 单个视频只能有一个说话人；
- 建议 20 分钟以内。

放到：

```text
video_data/
```

### 4. B 站视频链接

新建一个 `.txt`，每行：

```text
角色名|视频链接
```

例如：

```text
Char1|https://www.bilibili.com/video/BVxxxx
Char2|https://www.bilibili.com/video/BVyyyy
```

目前主要测试了 bilibili，其它站点不保证。

## 数据命名注意

`CharacterName` 用英文字符，`random_number` 必须有，用来区分同一角色的多个文件。例如：

```text
Diana_0.wav
Diana_1.wav
Diana_999999.wav
```

不要写成：

```text
Diana.wav
Diana_abc.wav
```
