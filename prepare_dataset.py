#!/usr/bin/env python3
"""Build a line-level OCR dataset from the latest Supabase backup.

Reads the newest database/<timestamp>/ table backup, joins annotations with
their page images in database/storage/Document/images/, crops every
annotated text line and writes:

    dataset/lines/<annotation_id>.png   grayscale line crops
    dataset/train.tsv, val.tsv, test.tsv  "<image path>\t<text>" per line
    dataset/charset.txt                  every character in the labels
    dataset/stats.json                   counts, for reference

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

CROP_HEIGHT = 128   # saved crops are resized to this height (training uses less)
PAD_RATIO = 0.08    # extra margin around each box, as a fraction of box height
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

    rows = {}
    for page_id, anns in sorted(by_page.items(), key=lambda kv: int(kv[0])):
        page = Image.open(IMAGES_DIR / images[page_id]["filename"]).convert("L")
        pw, ph = page.size
        for ann, text in anns:
            x, y, w, h = (float(ann[k]) for k in ("x", "y", "width", "height"))
            pad = h * PAD_RATIO
            box = (max(0, x - pad), max(0, y - pad), min(pw, x + w + pad), min(ph, y + h + pad))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                dropped["box too small"] += 1
                continue
            crop = page.crop(tuple(round(v) for v in box))
            new_w = max(1, round(crop.width * CROP_HEIGHT / crop.height))
            crop = crop.resize((new_w, CROP_HEIGHT), Image.BILINEAR)
            path = OUT_DIR / "lines" / f"{ann['id']}.png"
            crop.save(path)
            rows.setdefault(page_id, []).append((path.relative_to(OUT_DIR).as_posix(), text))
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

    (OUT_DIR / "charset.txt").write_text("".join(sorted(chars)), encoding="utf-8")
    stats["charset_size"] = len(chars)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
