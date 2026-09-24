#!/usr/bin/env python3
"""Train a Khmer handwriting line recognizer from scratch.

Model: a small CNN stem turns the line image into a sequence of feature
frames, a Transformer encoder contextualises them, and a Transformer decoder
generates the text one character at a time. An auxiliary CTC head on the
encoder speeds up and stabilises training on small datasets.

Training data: the real line crops, each shown with a randomly loose or tight
box (crops keep page context, so parts of neighbouring lines appear, as with
a line detector's boxes), plus synthetic lines rendered from Khmer fonts
(synth.py). Synthetic lines make up a large share of early epochs and a small
share later, so the model first learns the glyphs, then the handwriting.

Usage:
    python train.py                     # train with defaults
    python train.py --epochs 200 --batch-size 16
    python train.py --init checkpoints/best.pt   # start from a trained model
    python train.py --synth-start 0     # real lines only
    python train.py --resume            # continue from checkpoints/last.pt
    python train.py --eval-only         # evaluate checkpoints/best.pt on test

Input comes from prepare_dataset.py (dataset/train.tsv, val.tsv, test.tsv, boxes.json).
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CTC loss has no MPS kernel

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
CKPT_DIR = ROOT / "checkpoints"

PAD_RATIO = 0.08  # standard margin around a box (prepare_dataset.py, the web app's detector boxes)

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


def wave(img: Image.Image) -> Image.Image:
    """Bend the baseline with a gentle sine wave, like a line written without ruling."""
    a = np.asarray(img)
    h, w = a.shape
    amp = random.uniform(0.02, 0.06) * h
    period = random.uniform(2, 6) * h
    shift = np.round(amp * np.sin(2 * math.pi * np.arange(w) / period + random.uniform(0, 2 * math.pi)))
    rows = np.arange(h)[:, None] - shift[None, :].astype(int)
    inside = (rows >= 0) & (rows < h)
    return Image.fromarray(np.where(inside, a[rows.clip(0, h - 1), np.arange(w)[None]], 255).astype(np.uint8))


def elastic(img: Image.Image) -> Image.Image:
    """Warp the line with small random displacements on a coarse grid: shaky, uneven strokes."""
    w, h = img.size
    nx, ny = max(2, round(w / h * 2)), 2
    amp = h * random.uniform(0.02, 0.05)
    xs, ys = np.linspace(0, w, nx + 1), np.linspace(0, h, ny + 1)
    off = np.random.default_rng(random.getrandbits(32)).uniform(-amp, amp, (ny + 1, nx + 1, 2))
    off[[0, -1]] = 0  # keep the top and bottom edges straight so no paper is pulled in
    mesh = []
    for j in range(ny):
        for i in range(nx):
            box = (round(xs[i]), round(ys[j]), round(xs[i + 1]), round(ys[j + 1]))
            src = lambda jj, ii: (xs[ii] + off[jj, ii, 0], ys[jj] + off[jj, ii, 1])
            quad = (*src(j, i), *src(j + 1, i), *src(j + 1, i + 1), *src(j, i + 1))
            mesh.append((box, quad))
    return img.transform(img.size, Image.MESH, mesh, Image.BILINEAR, fillcolor=255)


def scribble(img: Image.Image) -> Image.Image:
    """Stray marks found on real pages: a ruled line, a teacher's tick or underline."""
    img = img.copy()
    d, w, h = ImageDraw.Draw(img), img.width, img.height
    if random.random() < 0.6:
        y = h * random.uniform(0.6, 1.0)
        d.line([(0, y), (w, y + random.uniform(-3, 3))], fill=random.randint(80, 170), width=random.randint(1, 3))
    else:
        x, y = random.uniform(0, w), random.uniform(0.2 * h, 0.9 * h)
        pts = [(x, y), (x + h * 0.15, y + h * 0.2), (x + h * random.uniform(0.3, 0.6), y - h * 0.3)]
        d.line(pts, fill=random.randint(90, 180), width=random.randint(2, 4), joint="curve")
    return img


def augment(img: Image.Image) -> Image.Image:
    # every geometric transform fills with white (255) so no fake ink appears at the edges
    if random.random() < 0.5:  # slant: writers lean their letters differently
        s = random.uniform(-0.3, 0.3)
        pad = int(abs(s) * img.height)
        img = img.transform((img.width + pad, img.height), Image.AFFINE,
                            (1, s, -pad if s > 0 else 0, 0, 1, 0), Image.BILINEAR, fillcolor=255)
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-2, 2), resample=Image.BILINEAR, expand=True, fillcolor=255)
    if random.random() < 0.3:
        img = elastic(img)
    if random.random() < 0.3:
        img = wave(img)
    if random.random() < 0.5:  # horizontal stretch / squeeze
        img = img.resize((max(8, int(img.width * random.uniform(0.8, 1.2))), img.height), Image.BILINEAR)
    if random.random() < 0.3:  # vertical shift and loose / tight horizontal margins
        dy = img.height * random.uniform(-0.05, 0.05)
        left, right = (int(img.height * random.uniform(-0.1, 0.2)) for _ in range(2))
        img = img.transform((max(8, img.width + left + right), img.height), Image.AFFINE,
                            (1, 0, -left, 0, 1, dy), Image.BILINEAR, fillcolor=255)
    if random.random() < 0.15:
        img = scribble(img)
    if random.random() < 0.3:  # uneven lighting across the line
        a = np.asarray(img, dtype=np.float32)
        ramp = np.linspace(0, 1, a.shape[1])[None, :] if random.random() < 0.7 else np.linspace(0, 1, a.shape[0])[:, None]
        a = a * (1 - random.uniform(0.1, 0.4) * (ramp if random.random() < 0.5 else 1 - ramp))
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    if random.random() < 0.2:  # thicker or thinner strokes
        img = img.filter(ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3))
    if random.random() < 0.3:  # scanner / paper noise
        a = np.asarray(img, dtype=np.float32)
        a = a + np.random.default_rng(random.getrandbits(32)).normal(0, random.uniform(3, 15), a.shape)
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if random.random() < 0.2:  # low-resolution photo
        f = random.uniform(0.35, 0.7)
        small = img.resize((max(8, int(img.width * f)), max(8, int(img.height * f))), Image.BILINEAR)
        img = small.resize(img.size, Image.BILINEAR)
    if random.random() < 0.2:  # JPEG artefacts
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(30, 85))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return img


def load_boxes() -> dict[str, list[float]]:
    """Where the labeled box sits inside each context crop (empty for old datasets without context)."""
    path = DATA_DIR / "boxes.json"
    return json.loads(path.read_text()) if path.exists() else {}


def crop_box(img: Image.Image, box, left, top, right, bottom) -> Image.Image:
    """Crop the box plus margins given as fractions of the box height (negative = tighter)."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    region = (max(0, x0 - left * h), max(0, y0 - top * h), min(img.width, x1 + right * h), min(img.height, y1 + bottom * h))
    return img.crop(tuple(round(v) for v in region))


class LineDataset(Dataset):
    """Real line crops; indices past the real lines are synthetic lines (see MixedBatchSampler)."""

    def __init__(self, rows, tokenizer, height, max_width, train, boxes=None, synth=None, batch_size=16):
        self.rows, self.tok = rows, tokenizer
        self.height, self.max_width, self.train = height, max_width, train
        self.boxes, self.synth, self.batch_size = boxes or {}, synth, batch_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if i >= len(self.rows):
            img, text = self.synthetic(i - len(self.rows))
        else:
            path, text = self.rows[i]
            img = Image.open(path).convert("L")
            box = self.boxes.get(path.relative_to(DATA_DIR).as_posix())
            if box:
                if self.train and random.random() < 0.75:  # loose or tight box, often showing neighbouring lines
                    img = crop_box(img, box, random.uniform(0, 0.4), random.uniform(-0.03, 0.25),
                                   random.uniform(0, 0.4), random.uniform(-0.03, 0.25))
                else:
                    img = crop_box(img, box, *[PAD_RATIO] * 4)
        if self.train:
            img = augment(img)
        return load_line(img, self.height, self.max_width), self.tok.encode(text), text

    def synthetic(self, k):
        # lines in one synthetic batch share a target length, so they have similar widths
        target = random.Random(k // self.batch_size).randint(4, 110)
        return self.synth.sample(random.Random(k), target)


class WidthBucketSampler(Sampler):
    """Batches lines of similar width together so little compute is wasted on padding."""

    def __init__(self, rows, batch_size, shuffle, boxes=None):
        self.batch_size, self.shuffle = batch_size, shuffle
        self.aspects = []
        for path, _ in rows:
            box = (boxes or {}).get(path.relative_to(DATA_DIR).as_posix())
            if box:
                h = box[3] - box[1]
                self.aspects.append((box[2] - box[0] + 2 * PAD_RATIO * h) / (h * (1 + 2 * PAD_RATIO)))
            else:
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


class MixedBatchSampler(Sampler):
    """Real batches plus a share of synthetic batches that changes per epoch (set_epoch)."""

    def __init__(self, real: WidthBucketSampler, n_real: int, batch_size: int, ratio):
        self.real, self.n_real, self.batch_size, self.ratio = real, n_real, batch_size, ratio
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def n_synth(self, epoch):
        return round(len(self.real) * self.ratio(epoch))

    def __iter__(self):
        batches = list(self.real)
        # a fresh random block of synthetic ids each epoch, so every epoch sees new lines
        base = self.n_real + random.randrange(10**9) * self.batch_size
        batches += [[base + (b * self.batch_size) + j for j in range(self.batch_size)]
                    for b in range(self.n_synth(self.epoch))]
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.real) + self.n_synth(self.epoch)


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
    def recognize(self, x, widths):
        """Return (attention decoder ids, CTC head ids) from one encoder pass."""
        memory, mask, lengths = self.encode(x, widths)
        return self.greedy(memory, mask, int(lengths.max())), self.ctc_greedy(memory, lengths)

    def ctc_greedy(self, memory, lengths):
        best = self.ctc_head(memory).argmax(-1).cpu()
        out = []
        for seq, n in zip(best, lengths.cpu()):
            s = seq[:n].tolist()
            out.append([c for i, c in enumerate(s) if c != PAD and (i == 0 or c != s[i - 1])])
        return out

    @torch.no_grad()
    def greedy(self, memory, mask, max_len):
        b, device = memory.size(0), memory.device
        # PAD-filled buffer grown in steps of 32: the causal mask hides the unfilled
        # tail, and a few fixed shapes avoid a recompile at every step on MPS
        ys = torch.full((b, 32), PAD, dtype=torch.long, device=device)
        ys[:, 0] = SOS
        done = torch.zeros(b, dtype=torch.bool, device=device)
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


def evaluate(model, loader, tok, device, amp, show=0):
    """Score both decoders. Returns {"attn": (cer, line_acc), "ctc": (cer, line_acc)}."""
    model.eval()
    errors, exact = {"attn": 0, "ctc": 0}, {"attn": 0, "ctc": 0}
    chars = n = 0
    samples = []
    for x, widths, _, texts in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            attn, ctc = model.recognize(x.to(device), widths.to(device))
        for a, c, t in zip(attn.cpu(), ctc, texts):
            preds = {"attn": tok.decode(a), "ctc": tok.decode(c)}
            for k, pred in preds.items():
                errors[k] += edit_distance(pred, t)
                exact[k] += pred == t
            chars += len(t)
            n += 1
            if len(samples) < show:
                samples.append((t, preds))
    for t, preds in samples:
        print(f"    GT  : {t}\n    ATTN: {preds['attn']}\n    CTC : {preds['ctc']}")
    return {k: (errors[k] / max(1, chars), exact[k] / max(1, n)) for k in errors}


def report(scores):
    return " | ".join(f"{k} CER {cer:.4f} acc {acc:.3f}" for k, (cer, acc) in scores.items())


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
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--height", type=int, default=64, help="multiple of 16")
    p.add_argument("--max-width", type=int, default=2048)
    p.add_argument("--ctc-weight", type=float, default=0.3)
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay of weights used for eval (0 disables)")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 mixed precision on CUDA")
    p.add_argument("--synth-start", type=float, default=1.0,
                   help="synthetic batches per real batch in epoch 1 (0 disables synthetic lines)")
    p.add_argument("--synth-end", type=float, default=0.2, help="synthetic batches per real batch at the end")
    p.add_argument("--synth-decay", type=int, default=0,
                   help="epochs to go from --synth-start to --synth-end (default: half of --epochs)")
    p.add_argument("--init", type=Path, help="start from this checkpoint's weights (e.g. checkpoints/best.pt)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    # checkpoints store the options; plain types only, so torch.load(weights_only=True) can read them
    saved_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    torch.manual_seed(0)
    random.seed(0)
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported() and not args.no_amp
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    tok = Tokenizer((DATA_DIR / "charset.txt").read_text(encoding="utf-8"))
    splits = {s: read_tsv(DATA_DIR / f"{s}.tsv") for s in ("train", "val", "test")}
    print(f"Device {device}{' (bf16)' if amp else ''} | vocab {len(tok)} | "
          + " | ".join(f"{s} {len(r)}" for s, r in splits.items()))

    boxes = load_boxes()
    if not boxes:
        print("dataset/boxes.json not found: crops have no page context. Re-run prepare_dataset.py for best results.")

    synth = None
    if args.synth_start > 0 or args.synth_end > 0:
        from synth import SynthLines, shaping_available
        if shaping_available():
            synth = SynthLines([t for _, t in splits["train"]])  # training text only: nothing from val/test
            print(f"Synthetic lines: {len(synth.fonts)} fonts, {args.synth_start:g} -> {args.synth_end:g} per real batch")
        else:
            print("WARNING: Pillow has no libraqm, so Khmer can't be rendered. Training without synthetic lines.")
    decay = args.synth_decay or max(1, args.epochs // 2)
    synth_ratio = lambda e: 0.0 if synth is None else (
        args.synth_start + (args.synth_end - args.synth_start) * min(1.0, (e - 1) / decay))

    def loader(split, train):
        ds = LineDataset(splits[split], tok, args.height, args.max_width, train, boxes, synth, args.batch_size)
        sampler = WidthBucketSampler(splits[split], args.batch_size, shuffle=train, boxes=boxes)
        if train:
            sampler = MixedBatchSampler(sampler, len(splits[split]), args.batch_size, synth_ratio)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
                          persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")

    train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)
    train_sampler = train_dl.batch_sampler

    model = OCRTransformer(len(tok), height=args.height, dropout=args.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    # an exponential moving average of the weights generalises better than the raw
    # weights on a small dataset; it is what gets evaluated and saved as best.pt
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema > 0 else None
    eval_model = ema.module if ema else model
    CKPT_DIR.mkdir(exist_ok=True)

    if args.init and not args.resume and not args.eval_only:
        ckpt = torch.load(args.init, map_location=device)
        if ckpt.get("charset") and ckpt["charset"] != tok.itos[len(SPECIALS):]:
            raise SystemExit(f"{args.init} was trained with a different charset.txt; it can't be used with --init.")
        model.load_state_dict(ckpt["model"])
        if ema:
            ema.module.load_state_dict(ckpt["model"])
        print(f"Initialised from {args.init} (epoch {ckpt.get('epoch')}, val CER {ckpt.get('val_cer', float('nan')):.4f})")

    if args.eval_only:
        ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        scores = evaluate(model, test_dl, tok, device, amp, show=10)
        print(f"Test {report(scores)} | best.pt uses the {ckpt.get('decoder', 'attn')} decoder")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # epochs get shorter as the synthetic share shrinks, so count the real number of steps
    total = sum(len(train_sampler.real) + train_sampler.n_synth(e) for e in range(1, args.epochs + 1))
    warmup = max(1, min(1000, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    start_epoch, best_cer = 1, float("inf")

    if args.resume and (CKPT_DIR / "last.pt").exists():
        ckpt = torch.load(CKPT_DIR / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch, best_cer = ckpt["epoch"] + 1, ckpt["best_cer"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val CER {best_cer:.4f})")

    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    ctc_device = torch.device("cpu") if device.type == "mps" else device  # no CTC kernel on MPS
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        t0, total_loss = time.time(), 0.0
        for step, (x, widths, y, _) in enumerate(train_dl, 1):
            x, widths, y = (t.to(device, non_blocking=True) for t in (x, widths, y))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                memory, mask, frames = model.encode(x, widths)
                logits = model.decode(y[:, :-1], memory, mask)
            loss = ce(logits.float().reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            if args.ctc_weight > 0:
                targets = y[:, 1:]
                target_lens = (targets != PAD).sum(1) - 1  # drop EOS
                log_probs = model.ctc_head(memory.float()).log_softmax(-1).transpose(0, 1)
                ctc = F.ctc_loss(log_probs.to(ctc_device), targets.to(ctc_device), frames.to(ctc_device),
                                 target_lens.to(ctc_device), blank=PAD, zero_infinity=True)
                loss = (1 - args.ctc_weight) * loss + args.ctc_weight * ctc.to(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema:
                ema.update_parameters(model)
            total_loss += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} | loss {total_loss / step:.4f} | "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        msg = (f"epoch {epoch:3d} | loss {total_loss / len(train_dl):.4f} | synth {synth_ratio(epoch):.2f} | "
               f"lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print("  evaluating on val...", flush=True)
            scores = evaluate(eval_model, val_dl, tok, device, amp, show=2)
            decoder = min(scores, key=lambda k: scores[k][0])
            cer = scores[decoder][0]
            msg += f" | val {report(scores)}"
            if cer < best_cer:
                best_cer = cer
                torch.save({"model": eval_model.state_dict(), "charset": tok.itos[len(SPECIALS):],
                            "args": saved_args, "epoch": epoch, "val_cer": cer, "decoder": decoder},
                           CKPT_DIR / "best.pt")
                msg += f"  * saved best ({decoder})"
        print(msg, flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict() if ema else None,
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best_cer": best_cer, "args": saved_args}, CKPT_DIR / "last.pt")

    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    scores = evaluate(model, test_dl, tok, device, amp, show=5)
    print(f"Best model (epoch {ckpt['epoch']}, {ckpt['decoder']} decoder) test {report(scores)}")


if __name__ == "__main__":
    main()
