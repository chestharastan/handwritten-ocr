# Khmer Handwritten OCR

A Transformer model, trained from scratch, that reads handwritten Khmer text one line at a time.

The data comes from a labeling app backed by Supabase. Scanned pages are stored in Supabase Storage, and each text line is marked with a bounding box and its transcription in Postgres. This repo backs up that data, turns it into a line-level dataset, and trains and runs the recognizer.

## Dataset

The `dataset/` folder is ready to train on:

| Split | Pages | Lines |
|---|---|---|
| train | 118 | 2,311 |
| val | 15 | 291 |
| test | 14 | 289 |

- `lines/<id>.jpg`: grayscale line crops that keep some page around the box, so neighbouring lines partly show. The box plus an 8% margin is 128 px high.
- `boxes.json`: where the labeled box sits in each crop
- `train.tsv`, `val.tsv`, `test.tsv`: one `image path<TAB>text` per line
- `charset.txt`: all 117 characters that appear in the labels
- `stats.json`: counts, including how many annotations were dropped and why

Splits are made **by page**, so lines from one page never appear in both train and test. Labels are normalized to Unicode NFC. Zero-width spaces are removed, runs of whitespace are collapsed to a single space, and empty or multi-line boxes are dropped.

## Model

```
line image (64 px high)
  → CNN stem            one feature frame per 4 px of width
  → Transformer encoder 4 layers, d=256, 8 heads
  → Transformer decoder 4 layers, generates text character by character
    + CTC head on the encoder (auxiliary loss, weight 0.3)
```

- About 9.4M parameters, all trained from scratch with no pretrained weights.
- Training uses AdamW with warmup and cosine decay, label smoothing of 0.1 and dropout 0.2.
- **Loose and tight boxes**: each training line is cut with a random margin from its context crop. Often that shows parts of the lines above and below, as boxes from the line detector do. Validation and test always use the standard 8% margin.
- **Synthetic lines** (`synth.py`): random word sequences from the training labels, rendered in 14 Khmer fonts in `fonts/`, two of them handwriting styles, on ruled paper. In epoch 1 there is one synthetic batch for every real batch. This falls to 0.2 by half-way, so the model learns the glyphs first and the handwriting later.
- **Augmentation**: slant, rotation, elastic warp, baseline wave, stretch, margins, ruled lines and ticks, uneven lighting, contrast, blur, stroke thickness, noise, low resolution and JPEG artefacts.
- An exponential moving average (EMA) of the weights is what gets evaluated and saved.
- On NVIDIA GPUs it trains in bf16 mixed precision.
- It is evaluated by **CER** (character error rate) on the validation set, for both the attention decoder and the CTC head. The best checkpoint and whichever decoder scored better are kept, and `predict.py` uses that decoder.

## Quick start

```bash
git clone git@github.com:chestharastan/handwritten-ocr.git
cd handwritten-ocr
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -u train.py
```

Training uses an NVIDIA GPU (CUDA), Apple Silicon (MPS) or the CPU, whichever is available. For an NVIDIA GPU, install the CUDA build of PyTorch first (see [TRAINING.md](TRAINING.md#nvidia-gpu-setup)).

See **[TRAINING.md](TRAINING.md)** for running in the background, stopping and resuming.

## Usage

```bash
.venv/bin/python train.py --eval-only                  # test-set CER using checkpoints/best.pt
.venv/bin/python predict.py dataset/lines/184.jpg       # read one line image
.venv/bin/python predict.py --decoder ctc dataset/lines/184.jpg   # force the CTC head
```

Main training options:

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 400 | Training epochs |
| `--batch-size` | 16 | Lines per batch |
| `--lr` | 3e-4 | Peak learning rate |
| `--weight-decay` | 0.05 | AdamW weight decay |
| `--dropout` | 0.2 | Dropout in the Transformer |
| `--height` | 64 | Input line height in pixels (multiple of 16) |
| `--ctc-weight` | 0.3 | Weight of the auxiliary CTC loss (0 disables it) |
| `--ema` | 0.999 | EMA decay for the evaluated weights (0 disables it) |
| `--eval-every` | 5 | Validate every N epochs |
| `--workers` | up to 8 | Data-loading processes |
| `--no-amp` | off | Disable bf16 mixed precision on CUDA |
| `--synth-start` | 1.0 | Synthetic batches per real batch in epoch 1 (0 with `--synth-end 0` turns them off) |
| `--synth-end` | 0.2 | Synthetic batches per real batch from `--synth-decay` onwards |
| `--synth-decay` | epochs / 2 | Epochs to go from start to end |
| `--init` | off | Start from a trained checkpoint's weights, e.g. `checkpoints/best.pt` |
| `--resume` | off | Continue from `checkpoints/last.pt` |

## Rebuilding the dataset from Supabase

This step is only needed to pull newly labeled pages. Create a `.env` file (never commit it):

```
DATABASE_URL=postgresql://postgres.<project-ref>:<password>@<host>:5432/postgres
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service role key>
```

Then run:

```bash
.venv/bin/python backup_supabase.py    # tables → database/<timestamp>/, images → database/storage/
.venv/bin/python prepare_dataset.py    # crops lines from the newest backup → dataset/
```

`backup_supabase.py` only downloads images it doesn't already have, and keeps the 10 most recent table backups.

## Project layout

| File | Purpose |
|---|---|
| `backup_supabase.py` | Back up Supabase tables and Storage images to `database/` |
| `prepare_dataset.py` | Join annotations with page images and crop line images into `dataset/` |
| `train.py` | Model definition, training loop and evaluation |
| `synth.py` | Synthetic Khmer line generator (`python synth.py` writes a preview) |
| `fonts/` | Khmer fonts for synthetic lines (SIL Open Font License, see `fonts/licenses/`) |
| `train.sh`, `train.bat` | One-step setup and training on Linux/macOS or Windows |
| `system/` | Web app to test the model: upload, crop lines, read text ([system/README.md](system/README.md)) |
| `predict.py` | Run the trained model on line images |
| `TRAINING.md` | How to train, stop and resume |

## Limitations

- About 2,300 training lines is small for a Transformer trained from scratch, so expect overfitting. Accuracy should improve noticeably as more pages are labeled.
- The model reads **single lines**. Full pages must first be split into lines, either with the labeling app's boxes or a separate line detector.
