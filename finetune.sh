#!/usr/bin/env bash
# Fine-tune a pretrained Khmer TrOCR model (finetune_trocr.py) on an NVIDIA GPU (Linux) or a Mac.
#
#   ./finetune.sh                               default model (lkhapple/Khmer-TrOCR-OCR)
#   ./finetune.sh --model channudam/khmer-trocr-base-printed --batch 8
#   ./finetune.sh --resume                      continue checkpoints/trocr/last
#   ./finetune.sh --eval-only                   test-set CER of checkpoints/trocr/best
#
# Uses the same .venv as train.sh and installs what fine-tuning needs on first use.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv and installing requirements..."
  python3 -m venv .venv
  if [ "$(uname)" = "Linux" ]; then
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
  fi
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python -c "import transformers" 2>/dev/null || .venv/bin/pip install -r requirements.txt
# optional: 8-bit AdamW saves GPU memory; fine-tuning falls back to Adafactor without it
if [ "$(uname)" = "Linux" ]; then
  .venv/bin/python -c "import bitsandbytes" 2>/dev/null || .venv/bin/pip install bitsandbytes || true
fi

.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
.venv/bin/python -u finetune_trocr.py "$@" 2>&1 | tee -a finetune.log
