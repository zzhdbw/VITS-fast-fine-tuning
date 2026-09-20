# 400 epoch 续训实验

从已经训练完 100 epoch 的模型继续训练到 400 epoch。

- 原实验目录：
  `recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k`
- 新实验目录：
  `recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_400ep`

脚本会先把原实验的 `G_latest.pth` / `D_latest.pth` 复制到新目录，
然后用 `--cont True --max_epochs 400` 继续训练。

运行：

```bash
bash recipes/genshin_zh_en_16k_pretrain/experiments/continue_400ep/run_continue_400ep.sh
```

日志：

```text
recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_400ep/nohup_400ep.log
```

TensorBoard：

```bash
tensorboard --logdir recipes/genshin_zh_en_16k_pretrain/output/genshin_zh_en_16k_400ep
```

SwanLab：

```text
项目：vits-fast-fine-tuning-pretrain
Run：genshin_zh_en_16k_400ep
```

默认 GPU：`0,1,4,5`。  
DDP 端口默认手动设置为 `18080`。
如果和其他任务冲突，直接修改 `run_continue_400ep.sh` 里的 `export MASTER_PORT=18080`。
