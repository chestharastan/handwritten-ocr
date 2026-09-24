# OCR test system

A web app for trying the trained models on your own images. Open a page photo: the line detector boxes every handwritten line and the recognizer reads each one. You can then fix, add or remove boxes by hand, and every changed box is read again.

```
frontend/  Next.js app (port 3000): upload, crop, show results
backend/   FastAPI (port 8000): loads checkpoints/best.pt (recognizer) and
           checkpoints/best_det.pt (YOLO line detector, optional), detects and reads lines
```

## Run

Put the trained models in `checkpoints/`: `best.pt` from the recognizer and `best_det.pt` from the [line detector](https://github.com/chestharastan/handwritten-det-ocr) (its `weights/best.pt`, renamed). Without `best_det.pt` the app still works, but you draw every box yourself. Then:

```bash
cd system
./start.sh          # installs what's missing, starts both, then open http://localhost:3000
./start.sh --dev    # same, with the Next.js dev server (hot reload)
```

Or start them separately:

```bash
cd system/backend  && ../../.venv/bin/python -m uvicorn app:app --port 8000
cd system/frontend && npm install && npm run dev
```

Needs Node.js 20+ and the project's `.venv` (see the main README). On Windows use `..\..\.venv\Scripts\python` for the backend.

## Using it

- **Open image**: use the button, drag and drop, or paste from the clipboard (⌘V / Ctrl+V). JPEG, PNG and iPhone **HEIC/HEIF** photos work. HEIC is converted to JPEG by the backend, because most browsers can't display it.
- **Automatic**: when the detector is loaded, opening an image finds every line, in reading order (top to bottom), and reads them all. **Detect lines** runs it again and replaces your boxes.
- **Crop by hand**: drag on the image to add a box around a line the detector missed. Draw it around **one line** with a small margin, like the training crops. It is read when you release the mouse.
- **Adjust**: drag a box to move it, or drag its corners to resize it. It is read again when you let go. Delete / Backspace removes the selected box.
- **Whole image is one line**: use this when the image is already a single cropped line.
- **Copy all text** copies every box's text in reading order, top to bottom.
- Each card shows the text from the decoder the checkpoint prefers. "Decoders disagree" shows the CTC and attention outputs side by side.

## API

| Endpoint | |
|---|---|
| `GET /health` | Checkpoint epoch, val CER, preferred decoder, device |
| `POST /convert` | multipart `file` (any image, including HEIC) → upright JPEG |
| `POST /detect` | multipart `file` (a page image), optional `?conf=0.25` → `{"width", "height", "ms", "boxes": [{"x", "y", "w", "h", "conf"}]}` in reading order, padded like the recognizer's training crops |
| `POST /recognize` | multipart `files` (1–50 line images) → `{"results": [{"name", "text", "ctc", "attn", "ms"}]}` |

```bash
curl -F files=@dataset/lines/184.jpg http://localhost:8000/recognize
```

Environment variables: `OCR_CHECKPOINT` (default `checkpoints/best.pt`), `OCR_DET_CHECKPOINT` (default `checkpoints/best_det.pt`), `OCR_DEVICE` (`auto`, `cuda`, `mps`, `cpu`), `OCR_CORS_ORIGINS` (default `http://localhost:3000`). The frontend reads `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).
