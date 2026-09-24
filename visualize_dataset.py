#!/usr/bin/env python3
"""Build an HTML viewer and statistics report for the labeled Khmer dataset.

Reads the newest table backup in database/<timestamp>/, segments every
transcription into Khmer words with khmer-nltk, and writes
visualize/index.html with two tabs:

  Pages       each page image with its boxes drawn on it, and the text of
              every box (split into words) listed on the right
  Statistics  page / box / word counts, train-val-test split, most common
              words, words-per-box distribution and a searchable word list

Open the result in a browser (it loads page images from database/storage/):
    python visualize_dataset.py
    open visualize/index.html
"""

import csv
import json
import logging
import re
import sys
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)
from khmernltk import word_tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "database"
IMAGES_DIR = BACKUP_DIR / "storage" / "Document" / "images"
DATASET_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "visualize"

KHMER_LETTER = re.compile(r"[ក-ឳ]")
NUMBER = re.compile(r"^[0-9០-៩.,:/-]*[0-9០-៩][0-9០-៩.,:/-]*$")

csv.field_size_limit(sys.maxsize)


def latest_backup() -> Path:
    backups = sorted(p for p in BACKUP_DIR.glob("20*") if (p / "public.annotations.csv").exists())
    if not backups:
        sys.exit("No table backup found. Run backup_supabase.py first.")
    return backups[-1]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def segment(text: str) -> tuple[list[str], int]:
    """Return (Khmer words, count of number tokens) for one transcription."""
    words, numbers = [], 0
    for tok in word_tokenize(text):
        tok = tok.strip()
        if not tok:
            continue
        if KHMER_LETTER.search(tok):
            words.append(tok)
        elif NUMBER.match(tok):
            numbers += 1
    return words, numbers


def dataset_splits() -> dict[str, str]:
    """annotation id -> train / val / test, from prepare_dataset.py output."""
    splits = {}
    for split in ("train", "val", "test"):
        tsv = DATASET_DIR / f"{split}.tsv"
        if tsv.exists():
            for line in tsv.read_text(encoding="utf-8").splitlines():
                splits[Path(line.split("\t", 1)[0]).stem] = split
    return splits


def main() -> None:
    backup = latest_backup()
    print(f"Using backup {backup}")
    images = {r["id"]: r for r in csv.DictReader(open(backup / "public.images.csv", encoding="utf-8"))}
    projects = {r["id"]: r["name"] for r in csv.DictReader(open(backup / "public.projects.csv", encoding="utf-8"))}
    annotations = list(csv.DictReader(open(backup / "public.annotations.csv", encoding="utf-8")))
    splits = dataset_splits()

    print(f"Segmenting {len(annotations)} boxes into words...")
    pages: dict[str, dict] = {}
    word_freq: Counter = Counter()
    char_total = number_total = 0
    words_per_box = []
    split_stats = {s: {"pages": set(), "boxes": 0, "words": 0} for s in ("train", "val", "test", "excluded")}

    for ann in annotations:
        img = images.get(ann["image_id"])
        if img is None:
            continue
        text = clean_text(ann["text"])
        words, numbers = segment(text) if text else ([], 0)
        split = splits.get(ann["id"], "excluded")
        word_freq.update(words)
        char_total += len(text.replace(" ", ""))
        number_total += numbers
        words_per_box.append(len(words))
        st = split_stats[split]
        st["pages"].add(ann["image_id"])
        st["boxes"] += 1
        st["words"] += len(words)

        page = pages.setdefault(ann["image_id"], {
            "id": int(ann["image_id"]),
            "file": img["filename"],
            "original": img["original_filename"],
            "project": projects.get(img["project_id"], img["project_id"]),
            "status": img["status"],
            "w": int(img["width"]),
            "h": int(img["height"]),
            "exists": (IMAGES_DIR / img["filename"]).exists(),
            "boxes": [],
        })
        page["boxes"].append({
            "id": int(ann["id"]),
            "x": round(float(ann["x"]), 1), "y": round(float(ann["y"]), 1),
            "w": round(float(ann["width"]), 1), "h": round(float(ann["height"]), 1),
            "text": text, "words": words, "split": split,
        })

    for page in pages.values():
        page["boxes"].sort(key=lambda b: (b["y"], b["x"]))
        page["words"] = sum(len(b["words"]) for b in page["boxes"])

    status_counts = Counter(r["status"] for r in images.values())
    hist = Counter(words_per_box)
    total_words = sum(word_freq.values())
    data = {
        "backup": backup.name,
        "imageBase": "../database/storage/Document/images/",
        "summary": {
            "pagesTotal": len(images),
            "pagesLabeled": len(pages),
            "boxes": len(words_per_box),
            "boxesTrainable": sum(split_stats[s]["boxes"] for s in ("train", "val", "test")),
            "words": total_words,
            "uniqueWords": len(word_freq),
            "numbers": number_total,
            "chars": char_total,
            "avgWordsPerBox": round(total_words / max(1, len(words_per_box)), 1),
            "singletons": sum(1 for c in word_freq.values() if c == 1),
        },
        "status": dict(status_counts),
        "splits": {s: {"pages": len(v["pages"]), "boxes": v["boxes"], "words": v["words"]}
                   for s, v in split_stats.items()},
        "wordsPerBox": [[k, hist[k]] for k in range(0, max(hist) + 1)],
        "wordFreq": word_freq.most_common(),
        "pages": sorted(pages.values(), key=lambda p: p["id"]),
    }

    OUT_DIR.mkdir(exist_ok=True)
    html = (Path(__file__).with_name("visualize_template.html")).read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (OUT_DIR / "index.html").write_text(html.replace("__DATA__", payload), encoding="utf-8")
    s = data["summary"]
    print(f"Pages {s['pagesLabeled']}/{s['pagesTotal']} labeled | boxes {s['boxes']} | "
          f"words {s['words']} ({s['uniqueWords']} unique) | chars {s['chars']}")
    print(f"Wrote {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
