"""Synthetic Khmer text lines for pre-training the recognizer.

Renders random word sequences, taken from the training labels only, in the
Khmer fonts in fonts/ (two of them handwriting styles). Each line is drawn on
paper with uneven lighting, often on a ruled line, often with parts of the
lines above and below showing, and then cropped with a random margin, much
like real line crops. train.py applies the usual augmentation on top.

Khmer needs complex text shaping (stacked subscripts, reordered vowels), which
Pillow only does with libraqm. SynthLines refuses to start without it rather
than produce wrongly shaped text.

Preview some samples:
    python synth.py            # writes synth_samples.png
"""

import random
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

FONT_DIR = Path(__file__).resolve().parent / "fonts"
SIZES = range(36, 64, 4)


def shaping_available() -> bool:
    return features.check("raqm")


@lru_cache(maxsize=256)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


def covered_chars(path: Path, chars: set[str]) -> set[str]:
    """Characters the font really has a glyph for (missing ones render as the .notdef box)."""
    font = load_font(str(path), 40)

    def render(c: str) -> bytes:
        img = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(img).text((20, 10), c, font=font, fill=255)
        return img.tobytes()

    notdef = render("")  # private-use code point: no Khmer font maps it
    return {c for c in chars if c.isspace() or render(c) != notdef}


class SynthLines:
    def __init__(self, texts: list[str], font_dir: Path = FONT_DIR):
        if not shaping_available():
            raise RuntimeError("Pillow was built without libraqm, so Khmer can't be shaped correctly")
        self.lines = [t for t in texts if t.strip()]
        chars = set("".join(self.lines))
        words = sorted({w for t in self.lines for w in t.split(" ") if w})
        self.fonts = []  # (path, words the font can draw, lines the font can draw)
        for path in sorted(font_dir.glob("*.ttf")):
            ok = covered_chars(path, chars)
            fw = [w for w in words if set(w) <= ok]
            fl = [t for t in self.lines if set(t) <= ok]
            if len(fw) >= 20:
                self.fonts.append((str(path), fw, fl))
        if not self.fonts:
            raise RuntimeError(f"No usable Khmer fonts in {font_dir}")

    def text(self, rng: random.Random, words: list[str], lines: list[str], target: int) -> str:
        if lines and rng.random() < 0.3:  # a window of a real line: natural word order
            t = rng.choice(lines)
            if len(t) > target:
                start = rng.randrange(len(t) - target + 1)
                t = t[start:start + target]
            text = t
        else:  # random words, with or without spaces between them as in handwriting
            parts, n = [], 0
            sep = " " if rng.random() < 0.6 else ""
            while n < target:
                w = rng.choice(words)
                parts.append(w)
                n += len(w) + len(sep)
            text = sep.join(parts)
        text = unicodedata.normalize("NFC", re.sub(r"\s+", " ", text)).strip()
        # a window can start on a combining mark, which can't be drawn alone
        while text and unicodedata.category(text[0]).startswith("M"):
            text = text[1:]
        return text or rng.choice(words)

    def sample(self, rng: random.Random, target_len: int) -> tuple[Image.Image, str]:
        path, words, lines = rng.choice(self.fonts)
        font = load_font(path, rng.choice(SIZES))
        text = self.text(rng, words, lines, target_len)

        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
        pitch = int(th * rng.uniform(0.95, 1.5))  # distance between neighbouring lines
        W, H = tw + 2 * th, th + 2 * pitch

        # paper with a lighting gradient
        paper = rng.randint(170, 250)
        img = Image.linear_gradient("L").rotate(rng.choice([0, 90, 180, 270])).resize((W, H))  # square, so no corners
        img = img.point(lambda v, s=rng.uniform(0, 0.25): int(paper * (1 - s * v / 255)))
        draw = ImageDraw.Draw(img)
        ink = rng.randint(15, 110)
        stroke = 1 if rng.random() < 0.3 else 0

        ox, oy = th - x0, pitch - y0  # main line's text origin
        ruled = rng.random() < 0.5
        rule_ink, rule_off = rng.randint(90, 170), rng.uniform(-0.05, 0.12) * th
        for dy in (-pitch, 0, pitch):
            if dy and rng.random() < 0.4:
                continue
            if ruled:
                y = oy + y1 + dy + rule_off
                draw.line([(0, y), (W, y)], fill=rule_ink, width=rng.randint(1, 2))
            t = text if dy == 0 else self.text(rng, words, lines, max(4, target_len + rng.randint(-10, 10)))
            dx = 0 if dy == 0 else rng.randint(-th, th)
            draw.text((ox + dx, oy + dy), t, font=font, fill=ink, stroke_width=stroke, stroke_fill=ink)

        # crop around the main line with a random margin, like a labeled or detected box
        bx0, by0, bx1, by1 = ox + x0, oy + y0, ox + x1, oy + y1
        m = lambda lo, hi: rng.uniform(lo, hi) * th
        box = (max(0, bx0 - m(0, 0.4)), max(0, by0 - m(-0.03, 0.25)), min(W, bx1 + m(0, 0.4)), min(H, by1 + m(-0.03, 0.25)))
        return img.crop(tuple(round(v) for v in box)), text


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import DATA_DIR, read_tsv

    gen = SynthLines([t for _, t in read_tsv(DATA_DIR / "train.tsv")])
    print(f"{len(gen.fonts)} fonts: " + ", ".join(Path(p).stem for p, _, _ in gen.fonts))
    rng = random.Random(0)
    samples = [gen.sample(rng, rng.randint(5, 60))[0] for _ in range(12)]
    samples = [s.resize((round(s.width * 64 / s.height), 64)) for s in samples]
    sheet = Image.new("L", (max(s.width for s in samples), 70 * len(samples)), 128)
    for i, s in enumerate(samples):
        sheet.paste(s, (0, i * 70))
    sheet.save("synth_samples.png")
    print("Wrote synth_samples.png")
