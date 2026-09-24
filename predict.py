#!/usr/bin/env python3
"""Read text from line images with the trained model.

Usage:
    python predict.py path/to/line.jpg [more.jpg ...]
    python predict.py --decoder ctc path/to/line.jpg     # force the CTC head
"""

import argparse

import torch
from PIL import Image

from train import CKPT_DIR, OCRTransformer, Tokenizer, load_line, pick_device


def main():
    p = argparse.ArgumentParser(description="Read text from line images.")
    p.add_argument("images", nargs="+")
    p.add_argument("--decoder", choices=["attn", "ctc"], help="default: whichever scored better on val")
    opts = p.parse_args()
    device = pick_device("auto")
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    args = ckpt["args"]
    decoder = opts.decoder or ckpt.get("decoder", "attn")
    tok = Tokenizer("".join(ckpt["charset"]))
    model = OCRTransformer(len(tok), height=args["height"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for path in opts.images:
        x = load_line(Image.open(path).convert("L"), args["height"], args["max_width"])
        attn, ctc = model.recognize(x[None].to(device), torch.tensor([x.shape[2]], device=device))
        print(f"{path}\t{tok.decode(attn[0].cpu() if decoder == 'attn' else ctc[0])}")


if __name__ == "__main__":
    main()
