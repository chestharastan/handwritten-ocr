# Khmer Handwriting OCR: Train, Resume, Stop

Run every command from the project folder:

```bash
cd ~/Desktop/ocr/khmer_hand
```

## NVIDIA GPU setup

On a machine with an NVIDIA card (for example an RTX 3060 6 GB), install the CUDA build of PyTorch **before** the other requirements. Otherwise pip installs the CPU-only build:

```bash
python -m venv .venv
# Linux:   .venv/bin/pip ...        Windows:  .venv\Scripts\pip ...
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The first line of `train.log` should read `Device cuda (bf16)`.

Recommended settings for a 6 GB card:

```bash
.venv/bin/python -u train.py                          # defaults fit comfortably
.venv/bin/python -u train.py --height 96              # taller input keeps Khmer stacked vowels/subscripts sharper; try if 64 plateaus
.venv/bin/python -u train.py --height 96 --batch-size 12   # if you hit CUDA out of memory
```

On Windows, use `.venv\Scripts\python` and run training in its own terminal window instead of `nohup ... &`.

## Quick train script

`train.sh` (Linux/macOS) and `train.bat` (Windows) do the setup and training in one step:

```bash
./train.sh                 # Linux/macOS: new run
train.bat                  # Windows: new run
./train.sh --resume        # continue from checkpoints/last.pt
./train.sh --height 96     # any train.py option is passed through
```

- The first run creates `.venv` and installs PyTorch (CUDA build on Linux and Windows).
- It prints the GPU name, or `not available`, which means training will be slow on the CPU.
- A new run first moves the old `checkpoints/*.pt` and `train.log` into `checkpoints/old_<time>/`, so earlier models are kept.
- Progress is shown on screen and appended to `train.log`.

## Getting better results (recommended runs)

The default run already does the following:
- It shows each real line with a random loose or tight box, so it learns to ignore parts of neighbouring lines.
- It mixes in synthetic lines from Khmer fonts: many at first, fewer later.
- It uses heavy augmentation, and runs 400 epochs.

Validation CER should keep going down for much longer than before. Expect the full run to take several hours on an RTX 3060.

```bash
./train.sh                                        # full run from scratch (best, slowest)
./train.sh --init checkpoints/best.pt --epochs 150 --synth-start 0.5   # continue from your current model (faster)
./train.sh --height 96                            # taller input: more detail for stacked vowels/subscripts
```

- **Check synthetic lines work**: the log should show `Synthetic lines: 14 fonts, 1 -> 0.2 per real batch`. If you see `Pillow has no libraqm`, Khmer can't be drawn correctly on that machine, and training continues on real lines only. Pillow's standard wheels for Windows, Linux and macOS include libraqm.
- **Preview synthetic lines**: `.venv/bin/python synth.py` writes `synth_samples.png`.
- **In the log**, `synth 0.60` is the share of synthetic batches in that epoch. Loss jumps a little while the mix changes. Judge progress by `val ... CER`.

## Fine-tuning a pretrained model (TrOCR)

Instead of training from scratch, you can fine-tune **TrOCR**, a pretrained image-to-text Transformer that has already been trained on Khmer text by the community. It uses the same data, box jitter, augmentation and synthetic lines as `train.py`.

```bash
./finetune.sh                     # Linux/macOS (finetune.bat on Windows)
./finetune.sh --resume            # continue after stopping
./finetune.sh --eval-only         # test-set CER of checkpoints/trocr/best
.venv/bin/python finetune_trocr.py --predict dataset/lines/184.jpg
```

- **Default model:** [`lkhapple/Khmer-TrOCR-OCR`](https://huggingface.co/lkhapple/Khmer-TrOCR-OCR), with 334M parameters. Its tokenizer is the most efficient for Khmer: a median of 48 tokens per line, and 103 for the longest line. Unchanged, it gets about 82% of characters wrong on these lines, so fine-tuning is essential.
- **Lighter option:** `./finetune.sh --model channudam/khmer-trocr-base-printed --batch 8` (150M parameters, trained on printed text).
- **Folded lines:** TrOCR always sees 384×384 pixels. Each long line is cut into 2–5 pieces that are stacked, so letters are not squashed. `--no-fold` turns this off.
- **Memory on 6 GB:** it uses bf16, gradient checkpointing, batch 4 × 4 accumulation steps and 8-bit AdamW. The script installs `bitsandbytes` for 8-bit AdamW if it can, and uses Adafactor otherwise. If you still run out of memory, use `--batch 2 --accum 8`.
- **Time:** the default is 20 epochs, evaluated on val after every epoch, which takes a few hours on a 3060.
- **Output:** `checkpoints/trocr/best/` (lowest val CER) and `checkpoints/trocr/last/` (for `--resume`). Each is about 1.3 GB. Compare the final `test CER` with `train.py`'s model.
- **Licence:** these community checkpoints don't state a licence. Check with their authors before using them in something you publish.
- The web app and `predict.py` still use the `train.py` model. Test the TrOCR model with `--eval-only` or `--predict`.

## 1. Prepare data (before a new training run)

```bash
.venv/bin/python backup_supabase.py    # pull latest tables + new images from Supabase
.venv/bin/python prepare_dataset.py    # crop labeled lines, split train/val/test
```

## 2. Train (start from scratch)

Runs in the background and writes progress to `train.log`:

```bash
nohup .venv/bin/python -u train.py > train.log 2>&1 &
```

Options:

```bash
nohup .venv/bin/python -u train.py --epochs 200 --batch-size 16 > train.log 2>&1 &
```

Starting a new run overwrites `checkpoints/`. Use **Resume** below to continue an earlier run instead.

## 3. Watch progress

```bash
tail -f train.log          # live log (Ctrl+C only stops watching, not training)
pgrep -fl train.py         # is training still running?
```

- Every epoch prints the loss.
- Every 5 epochs it prints `val attn CER` and `ctc CER` (character error rate for each decoder, lower is better).
- `* saved best` means `checkpoints/best.pt` was updated.

## 4. Stop

```bash
pkill -f train.py
```

`checkpoints/last.pt` is saved at the end of every epoch, so stopping loses at most the current epoch.

## 5. Resume

Continues from `checkpoints/last.pt` with the same epoch, learning rate and best score:

```bash
nohup .venv/bin/python -u train.py --resume >> train.log 2>&1 &
```

`>>` appends to the existing log instead of replacing it.

## 6. Test and use the model

```bash
.venv/bin/python train.py --eval-only                 # CER on the test set (uses best.pt)
.venv/bin/python predict.py dataset/lines/184.jpg      # read one line image
```

## Files

| Path | What it is |
|---|---|
| `checkpoints/best.pt` | Best model so far (lowest val CER) |
| `checkpoints/last.pt` | Latest epoch, used by `--resume` |
| `train.log` | Training output |
| `dataset/` | Line crops + `train.tsv` / `val.tsv` / `test.tsv` |
