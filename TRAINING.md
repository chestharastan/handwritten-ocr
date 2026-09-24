# Khmer Handwriting OCR: Train, Resume, Stop

Run every command from the project folder:

```bash
cd ~/Desktop/ocr/khmer_hand
```

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
- Every 10 epochs it prints `val CER` (character error rate, lower is better).
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
.venv/bin/python predict.py dataset/lines/184.png      # read one line image
```

## Files

| Path | What it is |
|---|---|
| `checkpoints/best.pt` | Best model so far (lowest val CER) |
| `checkpoints/last.pt` | Latest epoch, used by `--resume` |
| `train.log` | Training output |
| `dataset/` | Line crops + `train.tsv` / `val.tsv` / `test.tsv` |
