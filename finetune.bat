@echo off
rem Fine-tune a pretrained Khmer TrOCR model (finetune_trocr.py) on an NVIDIA GPU (Windows).
rem
rem   finetune.bat                              default model (lkhapple/Khmer-TrOCR-OCR)
rem   finetune.bat --model channudam/khmer-trocr-base-printed --batch 8
rem   finetune.bat --resume                     continue checkpoints\trocr\last
rem   finetune.bat --eval-only                  test-set CER of checkpoints\trocr\best
rem
rem Uses the same .venv as train.bat and installs what fine-tuning needs on first use.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Creating .venv and installing requirements...
  python -m venv .venv || exit /b 1
  .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124 || exit /b 1
  .venv\Scripts\pip install -r requirements.txt || exit /b 1
)
.venv\Scripts\python -c "import transformers" 2>nul || .venv\Scripts\pip install -r requirements.txt || exit /b 1
rem optional: 8-bit AdamW saves GPU memory; fine-tuning falls back to Adafactor without it
.venv\Scripts\python -c "import bitsandbytes" 2>nul || .venv\Scripts\pip install bitsandbytes

.venv\Scripts\python -c "import torch; print('CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"

rem PowerShell Tee-Object shows progress and also writes finetune.log
powershell -NoProfile -Command ".venv\Scripts\python -u finetune_trocr.py %* 2>&1 | Tee-Object -FilePath finetune.log -Append"
