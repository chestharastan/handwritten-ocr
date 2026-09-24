#!/usr/bin/env python3
"""Train a Khmer handwriting line recognizer from scratch.

Model: a small CNN stem turns the line image into a sequence of feature
frames, a Transformer encoder contextualises them, and a Transformer decoder
generates the text one character at a time. An auxiliary CTC head on the
encoder speeds up and stabilises training on small datasets.

Usage:
    python train.py                     # train with defaults
    python train.py --epochs 200 --batch-size 16
    python train.py --resume            # continue from checkpoints/last.pt
    python train.py --eval-only         # evaluate checkpoints/best.pt on test

Input comes from prepare_dataset.py (dataset/train.tsv, val.tsv, test.tsv).
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CTC loss has no MPS kernel

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
CKPT_DIR = ROOT / "checkpoints"

PAD, SOS, EOS, UNK = 0, 1, 2, 3  # PAD doubles as the CTC blank
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


# ----------------------------------------------------------------- tokenizer

class Tokenizer:
    def __init__(self, charset: str):
        self.itos = SPECIALS + list(charset)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in text]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i >= len(SPECIALS):
                out.append(self.itos[i])
        return "".join(out)


# ------------------------------------------------------------------- dataset

def read_tsv(path: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        img, text = line.split("\t", 1)
        rows.append((DATA_DIR / img, text))
    return rows


def load_line(img: Image.Image, height: int, max_width: int) -> torch.Tensor:
    """Resize to a fixed height and return a 1xHxW tensor with ink ~1, paper ~0."""
    w = max(16, min(max_width, round(img.width * height / img.height)))
    img = img.resize((w, height), Image.BILINEAR)
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float()
    x = 1.0 - x.view(1, height, w) / 255.0
    # per-image contrast stretch so faint pencil and dark pen look alike
    lo, hi = x.quantile(0.05), x.quantile(0.995)
    return ((x - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-2, 2), resample=Image.BILINEAR, expand=True, fillcolor=255)
    if random.random() < 0.5:  # horizontal stretch / squeeze
        img = img.resize((max(8, int(img.width * random.uniform(0.8, 1.2))), img.height), Image.BILINEAR)
    if random.random() < 0.5:  # vertical crop / pad jitter
        dy = int(img.height * random.uniform(-0.08, 0.08))
        img = img.crop((0, dy, img.width, img.height + dy))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    if random.random() < 0.2:  # thicker or thinner strokes
        img = img.filter(ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3))
    return img


class LineDataset(Dataset):
    def __init__(self, rows, tokenizer, height, max_width, train):
        self.rows, self.tok = rows, tokenizer
        self.height, self.max_width, self.train = height, max_width, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        img = Image.open(path).convert("L")
        if self.train:
            img = augment(img)
        return load_line(img, self.height, self.max_width), self.tok.encode(text), text


class WidthBucketSampler(Sampler):
    """Batches lines of similar width together so little compute is wasted on padding."""

    def __init__(self, rows, batch_size, shuffle):
        self.batch_size, self.shuffle = batch_size, shuffle
        self.aspects = []
        for path, _ in rows:
            with Image.open(path) as im:
                self.aspects.append(im.width / im.height)

    def __iter__(self):
        idx = list(range(len(self.aspects)))
        if self.shuffle:
            random.shuffle(idx)
        chunk = self.batch_size * 50
        batches = []
        for i in range(0, len(idx), chunk):
            group = sorted(idx[i:i + chunk], key=lambda j: self.aspects[j])
            batches += [group[k:k + self.batch_size] for k in range(0, len(group), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.aspects) / self.batch_size)


def round_up(n, multiple):
    return (n + multiple - 1) // multiple * multiple


def collate(batch):
    images, ids, texts = zip(*batch)
    h = images[0].shape[1]
    # fixed size steps keep the number of distinct tensor shapes (and MPS recompiles) small
    max_w = round_up(max(x.shape[2] for x in images), 128)
    x = torch.zeros(len(images), 1, h, max_w)
    widths = torch.tensor([im.shape[2] for im in images])
    for i, im in enumerate(images):
        x[i, :, :, : im.shape[2]] = im
    max_len = round_up(max(len(t) for t in ids) + 2, 32)
    y = torch.full((len(ids), max_len), PAD, dtype=torch.long)
    for i, t in enumerate(ids):
        y[i, : len(t) + 2] = torch.tensor([SOS, *t, EOS])
    return x, widths, y, list(texts)


# --------------------------------------------------------------------- model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


def conv_block(cin, cout, pool):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU()]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


class OCRTransformer(nn.Module):
    DOWNSAMPLE = 4  # image width / encoder frames

    def __init__(self, vocab_size, height=64, d_model=256, nhead=8, enc_layers=4,
                 dec_layers=4, ff=1024, dropout=0.1):
        super().__init__()
        # 1 x H x W  ->  d_model x H/16 x W/4
        self.stem = nn.Sequential(
            conv_block(1, 64, (2, 2)),
            conv_block(64, 128, (2, 2)),
            conv_block(128, 192, None),
            conv_block(192, 192, (2, 1)),
            conv_block(192, d_model, None),
            conv_block(d_model, d_model, (2, 1)),
        )
        self.proj = nn.Linear(d_model * (height // 16), d_model)
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers, enable_nested_tensor=False)
        dec = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def encode(self, x, widths):
        f = self.stem(x)                                  # B, C, H', T
        b, c, h, t = f.shape
        f = self.proj(f.permute(0, 3, 1, 2).reshape(b, t, c * h))
        lengths = (widths // self.DOWNSAMPLE).clamp(min=1)  # max-pooling floors
        mask = torch.arange(t, device=x.device)[None] >= lengths.to(x.device)[:, None]
        mem = self.encoder(self.drop(self.pos(f)), src_key_padding_mask=mask)
        return self.enc_norm(mem), mask, lengths

    def decode(self, tgt_in, memory, mem_mask):
        L = tgt_in.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.drop(self.pos(self.embed(tgt_in) * math.sqrt(self.d_model)))
        h = self.decoder(h, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_in == PAD,
                         memory_key_padding_mask=mem_mask, tgt_is_causal=True)
        return self.out(self.dec_norm(h))

    @torch.no_grad()
    def greedy(self, x, widths, max_len=220):
        memory, mask, _ = self.encode(x, widths)
        # PAD-filled buffer grown in steps of 32: the causal mask hides the unfilled
        # tail, and a few fixed shapes avoid a recompile at every step on MPS
        ys = torch.full((x.size(0), 32), PAD, dtype=torch.long, device=x.device)
        ys[:, 0] = SOS
        done = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for i in range(max_len):
            if i + 1 >= ys.size(1):
                ys = F.pad(ys, (0, 32), value=PAD)
            nxt = self.decode(ys[:, : round_up(i + 1, 32)], memory, mask)[:, i].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys[:, i + 1] = nxt
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]


# ------------------------------------------------------------------- metrics

def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(model, loader, tok, device, show=0):
    model.eval()
    errors = chars = exact = n = 0
    samples = []
    for x, widths, _, texts in loader:
        preds = model.greedy(x.to(device), widths.to(device))
        for p, t in zip(preds.cpu(), texts):
            pred = tok.decode(p)
            errors += edit_distance(pred, t)
            chars += len(t)
            exact += pred == t
            n += 1
            if len(samples) < show:
                samples.append((t, pred))
    for t, p in samples:
        print(f"    GT  : {t}\n    PRED: {p}")
    return errors / max(1, chars), exact / max(1, n)


# ---------------------------------------------------------------------- main

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--max-width", type=int, default=2048)
    p.add_argument("--ctc-weight", type=float, default=0.3)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = pick_device(args.device)
    tok = Tokenizer((DATA_DIR / "charset.txt").read_text(encoding="utf-8"))
    splits = {s: read_tsv(DATA_DIR / f"{s}.tsv") for s in ("train", "val", "test")}
    print(f"Device {device} | vocab {len(tok)} | "
          + " | ".join(f"{s} {len(r)}" for s, r in splits.items()))

    def loader(split, train):
        ds = LineDataset(splits[split], tok, args.height, args.max_width, train)
        sampler = WidthBucketSampler(splits[split], args.batch_size, shuffle=train)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                          num_workers=args.workers, persistent_workers=args.workers > 0)

    train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)

    model = OCRTransformer(len(tok), height=args.height).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    CKPT_DIR.mkdir(exist_ok=True)

    if args.eval_only:
        ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        cer, acc = evaluate(model, test_dl, tok, device, show=10)
        print(f"Test CER {cer:.4f} | line accuracy {acc:.3f}")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_dl)
    warmup = min(1000, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    start_epoch, best_cer = 1, float("inf")

    if args.resume and (CKPT_DIR / "last.pt").exists():
        ckpt = torch.load(CKPT_DIR / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch, best_cer = ckpt["epoch"] + 1, ckpt["best_cer"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val CER {best_cer:.4f})")

    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, total_loss = time.time(), 0.0
        for step, (x, widths, y, _) in enumerate(train_dl, 1):
            x, widths, y = x.to(device), widths.to(device), y.to(device)
            memory, mask, frames = model.encode(x, widths)
            logits = model.decode(y[:, :-1], memory, mask)
            loss = ce(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            if args.ctc_weight > 0:
                targets = y[:, 1:]
                target_lens = (targets != PAD).sum(1) - 1  # drop EOS
                log_probs = model.ctc_head(memory).log_softmax(-1).transpose(0, 1)
                ctc = F.ctc_loss(log_probs.cpu(), targets.cpu(), frames.cpu(), target_lens.cpu(),
                                 blank=PAD, zero_infinity=True)
                loss = (1 - args.ctc_weight) * loss + args.ctc_weight * ctc.to(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} | loss {total_loss / step:.4f} | "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        msg = (f"epoch {epoch:3d} | loss {total_loss / len(train_dl):.4f} | "
               f"lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print("  evaluating on val...", flush=True)
            cer, acc = evaluate(model, val_dl, tok, device, show=2)
            msg += f" | val CER {cer:.4f} | line acc {acc:.3f}"
            if cer < best_cer:
                best_cer = cer
                torch.save({"model": model.state_dict(), "charset": tok.itos[len(SPECIALS):],
                            "args": vars(args), "epoch": epoch, "val_cer": cer}, CKPT_DIR / "best.pt")
                msg += "  * saved best"
        print(msg, flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best_cer": best_cer, "args": vars(args)}, CKPT_DIR / "last.pt")

    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    cer, acc = evaluate(model, test_dl, tok, device, show=5)
    print(f"Best model (epoch {ckpt['epoch']}) test CER {cer:.4f} | line accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
