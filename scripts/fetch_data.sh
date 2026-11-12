#!/bin/bash
# Downloads the two benchmarks from their sources (we don't redistribute them) and a pinned
# checkout of Pinto Notes for its MCP server code.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
mkdir -p data/longmemeval
# LoCoMo (Maharana et al., ACL 2024), CC BY-NC 4.0
curl -fsSL -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
# LongMemEval-S, cleaned (Wu et al., 2024), MIT
curl -fsSL -o data/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
# Pinto Notes at the commit we tested
if [ ! -d pinto-notes ]; then git clone -q https://github.com/pinto-notes/pinto-notes.git pinto-notes; fi
git -C pinto-notes checkout -q 80ce88ca
echo "data/ and pinto-notes/ ready"
