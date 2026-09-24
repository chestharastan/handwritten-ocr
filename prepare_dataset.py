#!/usr/bin/env python3
"""Build a line-level OCR dataset from the latest Supabase backup.

Reads the newest database/<timestamp>/ table backup, joins annotations with
their page images in database/storage/Document/images/, crops every
annotated text line and writes:

    dataset/lines/<annotation_id>.jpg   grayscale line crops with page context around the box
    dataset/boxes.json                   where the labeled box sits in each crop
    dataset/train.tsv, val.tsv, test.tsv  "<image path>\t<text>" per line
    dataset/charset.txt                  every character in the labels
    dataset/stats.json                   counts, for reference

Each crop keeps CONTEXT_RATIO of the box height of page around the box, so
training can vary how loose the box is and show parts of neighbouring lines,
like the boxes a line detector produces. Crops are scaled so that the box
plus the standard PAD_RATIO margin is CROP_HEIGHT pixels high.

Splits are made per page (not per line) so lines from one page never land in
both train and test.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --backup database/2026-09-24_12-07-54
"""

import argparse
import csv
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "database"
IMAGES_DIR = BACKUP_DIR / "storage" / "Document" / "images"
OUT_DIR = ROOT / "dataset"

CROP_HEIGHT = 128     # the box plus PAD_RATIO margin is resized to this height (training uses less)
PAD_RATIO = 0.08      # standard margin around each box, as a fraction of box height
CONTEXT_RATIO = 0.35  # page kept around each box in the saved crop, as a fraction of box height
MAX_TEXT_LEN = 200  # longer labels are usually multi-line paragraph boxes
SPLIT = (0.8, 0.1, 0.1)
SEED = 42

csv.field_size_limit(sys.maxsize)


def latest_backup() -> Path:
    backups = sorted(p for p in BACKUP_DIR.glob("20*") if (p / "public.annotations.csv").exists())
    if not backups:
        sys.exit("No table backup found. Run backup_supabase.py first.")
    return backups[-1]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "")  # zero-width space: invisible, can't be read from ink
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backup", type=Path, help="table backup folder (default: newest)")
    args = parser.parse_args()

    backup = args.backup or latest_backup()
    print(f"Using backup {backup}")
    images = {r["id"]: r for r in csv.DictReader(open(backup / "public.images.csv", encoding="utf-8"))}
    annotations = list(csv.DictReader(open(backup / "public.annotations.csv", encoding="utf-8")))

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR / "lines", ignore_errors=True)
    (OUT_DIR / "lines").mkdir(parents=True, exist_ok=True)

    dropped = Counter()
    by_page = defaultdict(list)
    for ann in annotations:
        raw = ann["text"].strip()
        text = clean_text(raw)
        if not text:
            dropped["empty text"] += 1
            continue
        if "\n" in raw or len(text) > MAX_TEXT_LEN:
            dropped["multi-line / too long"] += 1
            continue
        image = images.get(ann["image_id"])
        if image is None or not (IMAGES_DIR / image["filename"]).exists():
            dropped["image file missing"] += 1
            continue
        by_page[ann["image_id"]].append((ann, text))

    rows, boxes = {}, {}
    for page_id, anns in sorted(by_page.items(), key=lambda kv: int(kv[0])):
        page = Image.open(IMAGES_DIR / images[page_id]["filename"]).convert("L")
        pw, ph = page.size
        for ann, text in anns:
            x, y, w, h = (float(ann[k]) for k in ("x", "y", "width", "height"))
            x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(float(pw), x + w), min(float(ph), y + h)
            if x1 - x0 < 8 or y1 - y0 < 8:
                dropped["box too small"] += 1
                continue
            ctx = h * CONTEXT_RATIO
            region = [round(v) for v in (max(0, x0 - ctx), max(0, y0 - ctx), min(pw, x1 + ctx), min(ph, y1 + ctx))]
            scale = CROP_HEIGHT / (h * (1 + 2 * PAD_RATIO))
            crop = page.crop(region)
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.BILINEAR)
            rel = f"lines/{ann['id']}.jpg"
            crop.save(OUT_DIR / rel, quality=90)
            boxes[rel] = [round((x0 - region[0]) * scale, 1), round((y0 - region[1]) * scale, 1),
                          round((x1 - region[0]) * scale, 1), round((y1 - region[1]) * scale, 1)]
            rows.setdefault(page_id, []).append((rel, text))
        print(f"  page {page_id}: {len(anns)} lines")

    pages = sorted(rows, key=int)
    random.Random(SEED).shuffle(pages)
    n_train = round(len(pages) * SPLIT[0])
    n_val = round(len(pages) * SPLIT[1])
    splits = {
        "train": pages[:n_train],
        "val": pages[n_train:n_train + n_val],
        "test": pages[n_train + n_val:],
    }

    stats = {"backup": backup.name, "dropped": dict(dropped), "splits": {}}
    chars = Counter()
    for name, page_ids in splits.items():
        lines = [r for pid in page_ids for r in rows[pid]]
        with open(OUT_DIR / f"{name}.tsv", "w", encoding="utf-8") as f:
            for path, text in lines:
                f.write(f"{path}\t{text}\n")
                chars.update(text)
        stats["splits"][name] = {"pages": len(page_ids), "lines": len(lines)}

    (OUT_DIR / "boxes.json").write_text(json.dumps(boxes, separators=(",", ":")))
    (OUT_DIR / "charset.txt").write_text("".join(sorted(chars)), encoding="utf-8")
    stats["charset_size"] = len(chars)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
