"""HTTP API around the trained Khmer line detector and recognizer.

Loads checkpoints/best.pt (recognizer) and, if present, checkpoints/best_det.pt
(YOLO line detector) once at startup.

    ../../.venv/bin/uvicorn app:app --port 8000        (run from system/backend)

POST /recognize   multipart "files": one or more line images (PNG/JPEG), already
                  cropped to a single line. Returns the text for each, from both
                  decoders, plus which decoder the checkpoint prefers.
POST /detect      multipart "file": a full page image. Returns the text-line
                  boxes in reading order, in the image's (EXIF-rotated) pixels.
POST /convert     multipart "file": any image, including HEIC/HEIF from iPhones
                  (which most browsers can't display). Returns it as a JPEG,
                  rotated upright.
GET  /health      model info (epoch, val CER, preferred decoder, device, detector).
"""

import io
import os
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # lets PIL open HEIC/HEIF everywhere below

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import OCRTransformer, Tokenizer, load_line, pick_device  # noqa: E402

CKPT = Path(os.environ.get("OCR_CHECKPOINT", ROOT / "checkpoints" / "best.pt"))
DET_CKPT = Path(os.environ.get("OCR_DET_CHECKPOINT", ROOT / "checkpoints" / "best_det.pt"))
DET_IMGSZ = 1024  # the size the detector was trained at
PAD_RATIO = 0.08  # recognizer training crops had this margin around each box, as a fraction of its height
MAX_FILES = 100

device = pick_device(os.environ.get("OCR_DEVICE", "auto"))
ckpt = torch.load(CKPT, map_location=device)
args = ckpt["args"]
tok = Tokenizer("".join(ckpt["charset"]))
model = OCRTransformer(len(tok), height=args["height"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
PREFERRED = ckpt.get("decoder", "attn")

detector = None
if DET_CKPT.exists():
    from ultralytics import YOLO

    detector = YOLO(DET_CKPT)
    DET_DEVICE = {"cuda": "0"}.get(device.type, device.type)

app = FastAPI(title="Khmer Handwriting OCR")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("OCR_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "checkpoint": CKPT.name,
        "epoch": ckpt.get("epoch"),
        "val_cer": ckpt.get("val_cer"),
        "decoder": PREFERRED,
        "device": str(device),
        "height": args["height"],
        "detector": DET_CKPT.name if detector else None,
    }


def recognize(img: Image.Image) -> dict:
    x = load_line(img, args["height"], args["max_width"])
    t0 = time.perf_counter()
    attn, ctc = model.recognize(x[None].to(device), torch.tensor([x.shape[2]], device=device))
    ms = (time.perf_counter() - t0) * 1000
    out = {"attn": tok.decode(attn[0].cpu()), "ctc": tok.decode(ctc[0])}
    return {**out, "text": out[PREFERRED], "ms": round(ms, 1)}


@app.post("/recognize")
async def recognize_files(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"At most {MAX_FILES} images per request")
    results = []
    for f in files:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(await f.read()))).convert("L")
        except Exception:
            raise HTTPException(400, f"{f.filename}: not a readable image")
        if img.width < 4 or img.height < 4:
            raise HTTPException(400, f"{f.filename}: crop is too small")
        results.append({"name": f.filename, **recognize(img)})
    return {"decoder": PREFERRED, "results": results}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    if detector is None:
        raise HTTPException(503, f"No detector checkpoint at {DET_CKPT}")
    try:
        page = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, f"{file.filename}: not a readable image")
    t0 = time.perf_counter()
    result = detector.predict(page, conf=conf, imgsz=DET_IMGSZ, device=DET_DEVICE, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    boxes = []
    for (x0, y0, x1, y1), c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        pad = (y1 - y0) * PAD_RATIO
        x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
        x1, y1 = min(float(page.width), x1 + pad), min(float(page.height), y1 + pad)
        boxes.append({"x": round(x0, 1), "y": round(y0, 1), "w": round(x1 - x0, 1), "h": round(y1 - y0, 1),
                      "conf": round(c, 3)})
    boxes.sort(key=lambda b: (b["y"] + b["h"] / 2, b["x"]))  # reading order: top to bottom
    return {"width": page.width, "height": page.height, "ms": round(ms, 1), "boxes": boxes}


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, f"{file.filename}: not a readable image")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return Response(buf.getvalue(), media_type="image/jpeg")
