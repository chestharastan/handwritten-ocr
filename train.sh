#!/usr/bin/env bash
# Train the Khmer handwriting model on an NVIDIA GPU (Linux) or a Mac.
#
#   ./train.sh                  new run (previous checkpoints are backed up to checkpoints/old_<time>/)
#   ./train.sh --resume         continue from checkpoints/last.pt
#   ./train.sh --height 96      any train.py option is passed through
#
# The first run creates .venv and installs PyTorch (CUDA build on Linux).
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

.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"

if [[ " $* " != *" --resume "* ]] && ls checkpoints/*.pt >/dev/null 2>&1; then
  old="checkpoints/old_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$old" && cp -p checkpoints/*.pt "$old"/   # copy: --init and the web app still use checkpoints/
  [ -f train.log ] && mv train.log "$old"/
  echo "Backed up previous checkpoints to $old"
fi

.venv/bin/python -u train.py "$@" 2>&1 | tee -a train.log
