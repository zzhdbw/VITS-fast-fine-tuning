# 低学习率续训实验

这个实验从原实验的 checkpoint 继续训练，只把学习率从：

```text
2e-4
```

降到：

```text
1e-4
```

原实验和目录不会被覆盖。

## 运行

先确认 GPU 空闲，然后：

```bash
# 自动用原实验最新 checkpoint
bash recipes/genshin_zh_en_16k_pretrain/experiments/lr1e-4/run_resume_lr1e4.sh

# 固定从 G_59000.pth / D_59000.pth 开始
STEP=59000 bash recipes/genshin_zh_en_16k_pretrain/experiments/lr1e-4/run_resume_lr1e4.sh

# 如果新实验目录里已有旧的 G_latest/D_latest，要强制换成新的起始 step
RESET_RESUME=1 STEP=57000 bash recipes/genshin_zh_en_16k_pretrain/experiments/lr1e-4/run_resume_lr1e4.sh
```

脚本会：

- 自动找原实验最新的 `G_xxxxx.pth` / `D_xxxxx.pth`；
- 复制一份到新目录，命名为 `G_latest.pth` / `D_latest.pth`；
- 用 `--cont True` 从这份 checkpoint 续训；
- 训练输出写到新目录。

## 输出目录

```text
recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_lr1e4/
```

日志：

```text
recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_lr1e4/nohup_lr1e4.log
```

PID：

```text
recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_lr1e4/train.pid
```

## 查看训练

```bash
tail -f recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_lr1e4/nohup_lr1e4.log
```

TensorBoard：

```bash
tensorboard --logdir recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_lr1e4
```

SwanLab 会记录到：

```text
genshin_zh_en_16k_lr1e4
```

## GPU

脚本默认使用：

```text
CUDA_VISIBLE_DEVICES=6,7
```

原实验用的是：

```text
0,1,4,5
```

如果 6,7 被占用，直接在运行前改脚本里的这一行，或这样运行：

```bash
CUDA_VISIBLE_DEVICES=2,3 bash recipes/genshin_zh_en_16k_pretrain/experiments/lr1e-4/run_resume_lr1e4.sh
```

原实验占用 DDP 端口 `8000`，低学习率实验默认改用 `8001`。如果还冲突，可以手动指定：

```bash
MASTER_PORT=8010 bash recipes/genshin_zh_en_16k_pretrain/experiments/lr1e-4/run_resume_lr1e4.sh
```

## 注意

- 这个实验会复制一份 checkpoint，不会改动原实验的 `G_latest.pth` / `D_latest.pth`。
- 不要在原实验正在写 checkpoint 的瞬间复制，正常情况下使用已经保存的 `G_xxxxx.pth` / `D_xxxxx.pth` 即可。
- 如果不想开 SwanLab，把脚本里的 `--use_swanlab True` 改成 `False`。
