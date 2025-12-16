#!/usr/bin/env bash
set -euo pipefail

# 简易一键跑通脚本，默认子集规模，便于快速验证
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/imdb_venv"

source "$VENV_PATH/bin/activate"

python -m imdb_word2vec.cli all \
  --subset-rows 100000 \
  --max-rows 50000 \
  --max-seq 50000 \
  --vocab-limit 20000


