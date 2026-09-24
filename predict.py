#!/usr/bin/env python3
"""Read text from line images with the trained model.

Usage:
    python predict.py path/to/line.png [more.png ...]
"""

import sys

import torch
from PIL import Image

from train import CKPT_DIR, OCRTransformer, Tokenizer, load_line, pick_device


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    device = pick_device("auto")
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    args = ckpt["args"]
    decoder = ckpt.get("decoder", "attn")  # whichever scored better on val
    tok = Tokenizer("".join(ckpt["charset"]))
    model = OCRTransformer(len(tok), height=args["height"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for path in sys.argv[1:]:
        x = load_line(Image.open(path).convert("L"), args["height"], args["max_width"])
        attn, ctc = model.recognize(x[None].to(device), torch.tensor([x.shape[2]], device=device))
        print(f"{path}\t{tok.decode(attn[0].cpu() if decoder == 'attn' else ctc[0])}")


if __name__ == "__main__":
    main()
