#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "找不到可执行的 Python: $PYTHON" >&2
  echo "请先准备 .venv，或用 PYTHON=/path/to/python 指定解释器。" >&2
  exit 1
fi

exec "$PYTHON" recipes/genshin_zh_en_16k_pretrain/04_infer.py "$@"
