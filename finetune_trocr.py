#!/usr/bin/env python3
"""Fine-tune a pretrained Khmer TrOCR model on the handwritten line dataset.

TrOCR (a ViT image encoder + Transformer text decoder) starts from weights that
already read Khmer, instead of training from scratch like train.py. It reuses
the same data as train.py: context crops with loose / tight boxes, the same
augmentation, and optionally synthetic lines from Khmer fonts.

TrOCR squeezes every image into 384 x 384. A handwritten line is ~12x wider
than tall, so squeezing it would crush the letters; instead each long line is
cut into 2-5 pieces that are stacked (folded) into a near-square image, and
the model learns to read the rows in order.

Usage:
    python finetune_trocr.py                         # fine-tune the default model
    python finetune_trocr.py --model channudam/khmer-trocr-base-printed --batch 8
    python finetune_trocr.py --resume                # continue checkpoints/trocr/last
    python finetune_trocr.py --eval-only             # test-set CER of checkpoints/trocr/best
    python finetune_trocr.py --predict line1.jpg line2.jpg

Outputs checkpoints/trocr/best/ (lowest val CER) and checkpoints/trocr/last/,
both loadable with VisionEncoderDecoderModel.from_pretrained.
"""

import argparse
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, VisionEncoderDecoderModel

from train import CKPT_DIR, DATA_DIR, PAD_RATIO, augment, crop_box, edit_distance, load_boxes, pick_device, read_tsv

OUT_DIR = CKPT_DIR / "trocr"
IMAGE_SIZE = 384  # TrOCR's fixed input size


# ------------------------------------------------------------------ images

def fold(img: Image.Image, max_rows: int = 5, overlap: float = 0.1) -> Image.Image:
    """Cut a long line into n pieces and stack them, so the square input keeps its detail.

    n is chosen so the stacked image is close to square. Neighbouring pieces
    overlap by a little (a fraction of the line height) so no letter is lost at a cut.
    """
    w, h = img.size
    n = max(1, min(max_rows, round(math.sqrt(w / h))))
    if n == 1:
        return img
    step, ov = w / n, overlap * h
    pieces = [img.crop((round(max(0, i * step - ov)), 0, round(min(w, (i + 1) * step + ov)), h)) for i in range(n)]
    out = Image.new("L", (max(p.width for p in pieces), h * n), 255)
    for i, p in enumerate(pieces):
        out.paste(p, (0, i * h))
    return out


def to_pixels(img: Image.Image, folded: bool) -> torch.Tensor:
    """Grayscale line -> 3x384x384 tensor normalized to [-1, 1], as TrOCR expects."""
    img = img.convert("L")
    if folded:
        img = fold(img)
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(IMAGE_SIZE, IMAGE_SIZE, 3)
    return (x.permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


# ------------------------------------------------------------------- data

class TrOCRLines(Dataset):
    def __init__(self, rows, boxes, tok, max_len, train, folded, synth=None, synth_ratio=0.0):
        self.rows, self.boxes, self.tok, self.max_len = rows, boxes, tok, max_len
        self.train, self.folded, self.synth = train, folded, synth
        self.n_synth = round(len(rows) * synth_ratio) if synth is not None and train else 0

    def __len__(self):
        return len(self.rows) + self.n_synth

    def __getitem__(self, i):
        if i >= len(self.rows):
            rng = random.Random(random.getrandbits(32))
            img, text = self.synth.sample(rng, rng.randint(4, 110))
        else:
            path, text = self.rows[i]
            img = Image.open(path).convert("L")
            box = self.boxes.get(path.relative_to(DATA_DIR).as_posix())
            if box:
                if self.train and random.random() < 0.75:  # loose or tight box, as in train.py
                    img = crop_box(img, box, random.uniform(0, 0.4), random.uniform(-0.03, 0.25),
                                   random.uniform(0, 0.4), random.uniform(-0.03, 0.25))
                else:
                    img = crop_box(img, box, *[PAD_RATIO] * 4)
        if self.train:
            img = augment(img)
        labels = self.tok(text, max_length=self.max_len, truncation=True).input_ids
        return to_pixels(img, self.folded), labels, text


def collate(batch):
    pixels, labels, texts = zip(*batch)
    n = max(len(l) for l in labels)
    y = torch.full((len(labels), n), -100, dtype=torch.long)  # -100 = ignored by the loss
    for i, l in enumerate(labels):
        y[i, : len(l)] = torch.tensor(l)
    return torch.stack(pixels), y, list(texts)


# ------------------------------------------------------------------ model

def load_model(name_or_path, device):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(name_or_path)
    if model.decoder.get_input_embeddings().num_embeddings < len(tok):
        model.decoder.resize_token_embeddings(len(tok))
    cfg = model.config
    # some community checkpoints leave these unset; generation and the label shift need them
    cfg.pad_token_id = cfg.pad_token_id if cfg.pad_token_id is not None else tok.pad_token_id
    if cfg.decoder_start_token_id is None:
        cfg.decoder_start_token_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    if cfg.eos_token_id is None:
        cfg.eos_token_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    return tok, model.to(device)


@torch.no_grad()
def generate(model, tok, pixels, max_len, beams, amp, device):
    cfg = model.config
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        out = model.generate(pixels, max_new_tokens=max_len, num_beams=beams, use_cache=True,
                             decoder_start_token_id=cfg.decoder_start_token_id,
                             pad_token_id=cfg.pad_token_id, eos_token_id=cfg.eos_token_id)
    return [clean(t) for t in tok.batch_decode(out, skip_special_tokens=True)]


def evaluate(model, tok, loader, max_len, beams, amp, device, show=0):
    """Character error rate and exact-line accuracy, counted like train.py does."""
    model.eval()
    errors = chars = exact = n = 0
    for pixels, _, texts in loader:
        for pred, gt in zip(generate(model, tok, pixels.to(device), max_len, beams, amp, device), texts):
            errors += edit_distance(pred, gt)
            chars += len(gt)
            exact += pred == gt
            n += 1
            if show > 0:
                print(f"    GT  : {gt}\n    PRED: {pred}")
                show -= 1
    return errors / max(1, chars), exact / max(1, n)


def make_optimizer(name, params, lr, weight_decay, device):
    """8-bit AdamW (bitsandbytes) keeps optimizer memory small on a 6 GB GPU; Adafactor needs no extra package."""
    if name == "auto":
        name = "adafactor"
        if device.type == "cuda":
            try:
                import bitsandbytes  # noqa: F401
                name = "adamw8bit"
            except ImportError:
                pass
    if name == "adamw8bit":
        import bitsandbytes as bnb
        return name, bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return name, Adafactor(params, lr=lr, weight_decay=weight_decay, scale_parameter=False,
                               relative_step=False, warmup_init=False)
    return "adamw", torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def save(model, tok, path: Path, meta: dict):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "khmer_ocr.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="lkhapple/Khmer-TrOCR-OCR", help="Hugging Face model id or local folder")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4, help="lines per step; lower it if the GPU runs out of memory")
    p.add_argument("--accum", type=int, default=4, help="steps per optimizer update (effective batch = batch x accum)")
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--optim", default="auto", choices=["auto", "adamw8bit", "adafactor", "adamw"])
    p.add_argument("--synth", type=float, default=0.2, help="synthetic lines per real line each epoch (0 disables)")
    p.add_argument("--no-fold", action="store_true", help="squeeze whole lines into 384x384 instead of folding them")
    p.add_argument("--max-target", type=int, default=256, help="longest label in tokens; longer training lines are skipped")
    p.add_argument("--beams", type=int, default=1, help="beam search width for evaluation")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 mixed precision on CUDA")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/trocr/last")
    p.add_argument("--eval-only", action="store_true", help="test-set CER of checkpoints/trocr/best")
    p.add_argument("--predict", nargs="+", type=Path, metavar="IMAGE", help="read line images with checkpoints/trocr/best")
    args = p.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported() and not args.no_amp
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    if args.predict or args.eval_only:
        best = OUT_DIR / "best"
        if not best.exists():
            raise SystemExit(f"{best} not found. Fine-tune first.")
        meta = json.loads((best / "khmer_ocr.json").read_text())
        tok, model = load_model(best, device)
        model.eval()
        if args.predict:
            for path in args.predict:
                x = to_pixels(Image.open(path), meta["fold"])[None].to(device)
                print(f"{path}\t{generate(model, tok, x, meta['max_target'], args.beams, amp, device)[0]}")
            return
        test = TrOCRLines(read_tsv(DATA_DIR / "test.tsv"), load_boxes(), tok, meta["max_target"], False, meta["fold"])
        dl = DataLoader(test, batch_size=16, collate_fn=collate, num_workers=args.workers)
        cer, acc = evaluate(model, tok, dl, meta["max_target"], args.beams, amp, device, show=5)
        print(f"Test CER {cer:.4f} | line accuracy {acc:.3f} | {meta['base_model']} epoch {meta['epoch']}")
        return

    start = OUT_DIR / "last" if args.resume else args.model
    tok, model = load_model(start, device)
    folded = not args.no_fold
    base_model = args.model
    if args.resume:
        meta = json.loads((OUT_DIR / "last" / "khmer_ocr.json").read_text())
        folded, base_model = meta["fold"], meta["base_model"]
    model.config.use_cache = False  # required with gradient checkpointing
    model.gradient_checkpointing_enable()
    print(f"Device {device}{' (bf16)' if amp else ''} | {base_model} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters | fold {folded}")

    splits = {s: read_tsv(DATA_DIR / f"{s}.tsv") for s in ("train", "val", "test")}
    too_long = [r for r in splits["train"] if len(tok(r[1]).input_ids) > args.max_target]
    splits["train"] = [r for r in splits["train"] if len(tok(r[1]).input_ids) <= args.max_target]
    if too_long:
        print(f"Skipping {len(too_long)} training lines longer than {args.max_target} tokens")
    boxes = load_boxes()

    synth = None
    if args.synth > 0:
        from synth import SynthLines, shaping_available
        if shaping_available():
            synth = SynthLines([t for _, t in splits["train"]])
        else:
            print("WARNING: Pillow has no libraqm, so Khmer can't be rendered. Training without synthetic lines.")

    train_ds = TrOCRLines(splits["train"], boxes, tok, args.max_target, True, folded, synth, args.synth)
    loader = lambda ds, bs, shuffle: DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                                                num_workers=args.workers, persistent_workers=args.workers > 0,
                                                pin_memory=device.type == "cuda", drop_last=shuffle)
    train_dl = loader(train_ds, args.batch, True)
    val_dl = loader(TrOCRLines(splits["val"], boxes, tok, args.max_target, False, folded), 16, False)
    test_dl = loader(TrOCRLines(splits["test"], boxes, tok, args.max_target, False, folded), 16, False)
    print(f"train {len(splits['train'])} real + {train_ds.n_synth} synthetic lines per epoch | "
          f"val {len(splits['val'])} | test {len(splits['test'])}")

    optim_name, opt = make_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay, device)
    updates = args.epochs * math.ceil(len(train_dl) / args.accum)
    warmup = max(1, updates // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / updates))))
    print(f"Optimizer {optim_name} | lr {args.lr:g} | effective batch {args.batch * args.accum} | {updates} updates")

    start_epoch, best_cer = 1, float("inf")
    if args.resume:
        state = torch.load(OUT_DIR / "last" / "train_state.pt", map_location=device)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, best_cer = state["epoch"] + 1, state["best_cer"]
        print(f"Resumed after epoch {state['epoch']} (best val CER {best_cer:.4f})")

    meta = {"base_model": base_model, "fold": folded, "max_target": args.max_target, "image_size": IMAGE_SIZE}
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for step, (pixels, labels, _) in enumerate(train_dl, 1):
            pixels, labels = pixels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(pixel_values=pixels, labels=labels).logits
            # logits line up with labels (the model shifts them right internally)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                                   ignore_index=-100, label_smoothing=0.1)
            (loss / args.accum).backward()
            if step % args.accum == 0 or step == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            total += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} | loss {total / step:.4f} | "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)

        print("  evaluating on val...", flush=True)
        cer, acc = evaluate(model, tok, val_dl, args.max_target, 1, amp, device, show=2)
        msg = (f"epoch {epoch:3d} | loss {total / len(train_dl):.4f} | lr {sched.get_last_lr()[0]:.2e} | "
               f"{time.time() - t0:.0f}s | val CER {cer:.4f} | line acc {acc:.3f}")
        if cer < best_cer:
            best_cer = cer
            save(model, tok, OUT_DIR / "best", {**meta, "epoch": epoch, "val_cer": cer})
            msg += "  * saved best"
        print(msg, flush=True)
        save(model, tok, OUT_DIR / "last", {**meta, "epoch": epoch, "val_cer": cer})
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "best_cer": best_cer},
                   OUT_DIR / "last" / "train_state.pt")

    tok, model = load_model(OUT_DIR / "best", device)
    cer, acc = evaluate(model, tok, test_dl, args.max_target, args.beams, amp, device, show=5)
    print(f"Best model test CER {cer:.4f} | line accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
