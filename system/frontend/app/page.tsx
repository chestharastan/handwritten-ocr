"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const CROP_HEIGHT = 128; // training line crops are 128 px high; the model resizes further itself
const MIN_BOX = 6; // display px; smaller drags count as a click

type Box = { id: number; x: number; y: number; w: number; h: number };
type Result = {
  geom: string; // which box geometry this result belongs to
  status: "pending" | "done" | "error";
  thumb?: string;
  text?: string;
  attn?: string;
  ctc?: string;
  ms?: number;
  error?: string;
};
type Health = { epoch: number; val_cer: number; decoder: string; device: string; detector: string | null };
type Page = { url: string; bitmap: ImageBitmap; name: string; file: File };
type Drag =
  | { kind: "draw"; id: number; ax: number; ay: number }
  | { kind: "resize"; id: number; ax: number; ay: number; orig: Box }
  | { kind: "move"; id: number; sx: number; sy: number; orig: Box };

// iPhone photos: most browsers can't decode HEIC, and the file's type is often empty
const isHeic = (f: File) => /image\/hei[cf]/.test(f.type) || /\.hei[cf]$/i.test(f.name);

const geomOf = (b: Box) => `${Math.round(b.x)},${Math.round(b.y)},${Math.round(b.w)},${Math.round(b.h)}`;

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [image, setImage] = useState<Page | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [results, setResults] = useState<Record<number, Result>>({});
  const [selected, setSelected] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [scale, setScale] = useState(1); // image px per display px
  const [copied, setCopied] = useState<string | null>(null);
  const [detectState, setDetectState] = useState<{ busy: boolean; msg: string }>({ busy: false, msg: "" });

  const svgRef = useRef<SVGSVGElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const drag = useRef<Drag | null>(null);
  const nextId = useRef(1);
  const boxesRef = useRef<Box[]>([]);
  boxesRef.current = boxes;
  const imageRef = useRef<Page | null>(null);
  imageRef.current = image;
  const healthRef = useRef<Health | null>(null);
  healthRef.current = health;

  // ------------------------------------------------------------ backend status
  const checkHealth = useCallback(() => {
    fetch(`${API_URL}/health`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((h: Health) => { setHealth(h); setHealthError(false); })
      .catch(() => { setHealth(null); setHealthError(true); });
  }, []);
  useEffect(checkHealth, [checkHealth]);

  // keep handle and label sizes constant on screen whatever the image resolution
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !image) return;
    const ro = new ResizeObserver(() => setScale(image.bitmap.width / Math.max(1, svg.clientWidth)));
    ro.observe(svg);
    return () => ro.disconnect();
  }, [image]);

  // ------------------------------------------------------------ recognition
  // crop boxes from the page, send them in one request, and fill in each box's card
  const recognizeMany = useCallback(async (list: Box[], page?: Page) => {
    const img = page ?? imageRef.current;
    if (!img || !list.length) return;
    const jobs = list.map((box) => {
      const h = Math.min(CROP_HEIGHT, Math.round(box.h));
      const w = Math.max(1, Math.round((box.w * h) / box.h));
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img.bitmap, box.x, box.y, box.w, box.h, 0, 0, w, h);
      return { box, geom: geomOf(box), canvas, thumb: canvas.toDataURL("image/png") };
    });
    setResults((r) => {
      const next = { ...r };
      jobs.forEach(({ box, geom, thumb }) => { next[box.id] = { geom, status: "pending", thumb }; });
      return next;
    });

    // ignore answers for a box that was moved, resized or deleted in the meantime
    const update = (fill: (j: (typeof jobs)[number], i: number) => Result) =>
      setResults((r) => {
        const next = { ...r };
        jobs.forEach((j, i) => {
          if (boxesRef.current.some((b) => b.id === j.box.id && geomOf(b) === j.geom)) next[j.box.id] = fill(j, i);
        });
        return next;
      });
    try {
      const form = new FormData();
      for (const j of jobs) {
        const blob = await new Promise<Blob>((res, rej) => j.canvas.toBlob((b) => (b ? res(b) : rej()), "image/png"));
        form.append("files", blob, `box-${j.box.id}.png`);
      }
      const resp = await fetch(`${API_URL}/recognize`, { method: "POST", body: form });
      if (!resp.ok) throw new Error((await resp.json().catch(() => null))?.detail ?? `HTTP ${resp.status}`);
      const out = (await resp.json()).results;
      update((j, i) => ({ geom: j.geom, status: "done", thumb: j.thumb, ...out[i] }));
      setHealthError(false);
    } catch (e) {
      if (e instanceof TypeError) setHealthError(true); // network error: backend not running
      const error = e instanceof Error ? e.message : "Failed";
      update((j) => ({ geom: j.geom, status: "error", thumb: j.thumb, error }));
    }
  }, []);
  const recognize = useCallback((box: Box) => recognizeMany([box]), [recognizeMany]);

  // find every line with the detector, replacing the current boxes, then read them all
  const detect = useCallback(async (page?: Page) => {
    const img = page ?? imageRef.current;
    if (!img) return;
    setDetectState({ busy: true, msg: "Detecting lines…" });
    try {
      const form = new FormData();
      form.append("file", img.file, img.name);
      const resp = await fetch(`${API_URL}/detect`, { method: "POST", body: form });
      if (!resp.ok) throw new Error((await resp.json().catch(() => null))?.detail ?? `HTTP ${resp.status}`);
      const out: { ms: number; boxes: Omit<Box, "id">[] } = await resp.json();
      if (imageRef.current !== img) return; // another image was opened meanwhile
      const found = out.boxes.map(({ x, y, w, h }) => ({ id: nextId.current++, x, y, w, h }));
      boxesRef.current = found;
      setBoxes(found);
      setResults({});
      setSelected(null);
      setDetectState({ busy: false, msg: `Found ${found.length} line${found.length === 1 ? "" : "s"} in ${Math.round(out.ms)} ms` });
      recognizeMany(found, img);
    } catch (e) {
      if (e instanceof TypeError) setHealthError(true);
      setDetectState({ busy: false, msg: `Detection failed: ${e instanceof Error ? e.message : "error"}` });
    }
  }, [recognizeMany]);

  // ------------------------------------------------------------ loading images
  const loadFile = useCallback(async (input: File) => {
    let file = input;
    if (isHeic(file)) {
      // have the backend convert it to an upright JPEG, then continue as with any image
      setDetectState({ busy: true, msg: "Converting HEIC…" });
      try {
        const form = new FormData();
        form.append("file", file, file.name);
        const resp = await fetch(`${API_URL}/convert`, { method: "POST", body: form });
        if (!resp.ok) throw new Error((await resp.json().catch(() => null))?.detail ?? `HTTP ${resp.status}`);
        file = new File([await resp.blob()], file.name.replace(/\.hei[cf]$/i, "") + ".jpg", { type: "image/jpeg" });
      } catch (e) {
        if (e instanceof TypeError) setHealthError(true);
        setDetectState({ busy: false, msg: `Couldn't convert HEIC: ${e instanceof Error ? e.message : "error"}` });
        return;
      }
    } else if (!file.type.startsWith("image/")) {
      setDetectState({ busy: false, msg: `${file.name || "That file"} is not an image` });
      return;
    }
    // the bitmap is used for cropping; from-image applies the phone's EXIF rotation like <img> does
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const page: Page = { url: URL.createObjectURL(file), bitmap, name: file.name || "pasted image", file };
    const old = imageRef.current;
    if (old) { URL.revokeObjectURL(old.url); old.bitmap.close(); }
    imageRef.current = page;
    setImage(page);
    boxesRef.current = [];
    setBoxes([]);
    setResults({});
    setSelected(null);
    setDetectState({ busy: false, msg: "" });
    if (healthRef.current?.detector) detect(page);
  }, [detect]);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith("image/") || isHeic(f));
      if (file) loadFile(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [loadFile]);

  const dropProps = {
    onDragOver: (e: React.DragEvent) => { e.preventDefault(); setDragOver(true); },
    onDragLeave: () => setDragOver(false),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) loadFile(file);
    },
  };

  const addBox = (b: Omit<Box, "id">) => {
    const box = { ...b, id: nextId.current++ };
    setBoxes((bs) => [...bs, box]);
    setSelected(box.id);
    recognize(box);
  };

  const removeBox = useCallback((id: number) => {
    setBoxes((bs) => bs.filter((b) => b.id !== id));
    setResults((r) => { const { [id]: _, ...rest } = r; return rest; });
    setSelected((s) => (s === id ? null : s));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === "Delete" || e.key === "Backspace") && selected !== null && !(e.target instanceof HTMLInputElement)) {
        e.preventDefault();
        removeBox(selected);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, removeBox]);

  // ------------------------------------------------------------ drawing boxes
  const toImage = (e: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    const W = image!.bitmap.width, H = image!.bitmap.height;
    return {
      x: Math.min(W, Math.max(0, ((e.clientX - r.left) / r.width) * W)),
      y: Math.min(H, Math.max(0, ((e.clientY - r.top) / r.height) * H)),
    };
  };
  const rectFrom = (ax: number, ay: number, x: number, y: number) =>
    ({ x: Math.min(ax, x), y: Math.min(ay, y), w: Math.abs(x - ax), h: Math.abs(y - ay) });

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!image || e.button !== 0) return;
    const p = toImage(e);
    const target = e.target as SVGElement;
    const id = Number(target.dataset.id);
    const box = boxes.find((b) => b.id === id);
    if (target.dataset.corner && box) {
      // resizing = drawing again from the opposite corner
      const [cx, cy] = target.dataset.corner.split("");
      drag.current = { kind: "resize", id, ax: cx === "l" ? box.x + box.w : box.x, ay: cy === "t" ? box.y + box.h : box.y, orig: box };
    } else if (box) {
      drag.current = { kind: "move", id, sx: p.x, sy: p.y, orig: box };
      setSelected(id);
    } else {
      const newId = nextId.current++;
      drag.current = { kind: "draw", id: newId, ax: p.x, ay: p.y };
      setBoxes((bs) => [...bs, { id: newId, x: p.x, y: p.y, w: 0, h: 0 }]);
      setSelected(newId);
    }
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const p = toImage(e);
    setBoxes((bs) => bs.map((b) => {
      if (b.id !== d.id) return b;
      if (d.kind === "move") {
        const W = image!.bitmap.width, H = image!.bitmap.height;
        return {
          ...b,
          x: Math.min(W - b.w, Math.max(0, d.orig.x + p.x - d.sx)),
          y: Math.min(H - b.h, Math.max(0, d.orig.y + p.y - d.sy)),
        };
      }
      return { id: b.id, ...rectFrom(d.ax, d.ay, p.x, p.y) };
    }));
  };

  const onPointerUp = () => {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    const box = boxesRef.current.find((b) => b.id === d.id);
    if (!box) return;
    if (box.w < MIN_BOX * scale || box.h < MIN_BOX * scale) {
      // a click (or a resize squashed to nothing): drop the new box, restore a resized one
      if (d.kind === "draw") removeBox(d.id);
      if (d.kind === "resize") setBoxes((bs) => bs.map((b) => (b.id === d.id ? d.orig : b)));
      return;
    }
    if (results[box.id]?.geom !== geomOf(box)) recognize(box);
  };

  // ------------------------------------------------------------ actions
  const ordered = [...boxes].filter((b) => b.w > 0 && b.h > 0);
  const readingOrder = [...ordered].sort((a, b) => a.y - b.y || a.x - b.x);
  const copy = (text: string, key: string) => {
    navigator.clipboard.writeText(text).then(() => { setCopied(key); setTimeout(() => setCopied(null), 1200); });
  };
  const allText = readingOrder.map((b) => results[b.id]?.text ?? "").filter(Boolean).join("\n");

  const hs = 10 * scale; // handle size in image px
  const fs = 13 * scale;

  return (
    <>
      <header>
        <h1>Khmer Handwriting OCR</h1>
        <span className="status">
          <i className={`dot ${health ? "ok" : healthError ? "err" : ""}`} />
          {health
            ? `Recognizer epoch ${health.epoch} · val CER ${(health.val_cer * 100).toFixed(1)}% · ${health.decoder} decoder · ` +
              `${health.detector ? "line detector on" : "no line detector"} · ${health.device}`
            : healthError ? "Backend not reachable" : "Connecting…"}
        </span>
      </header>

      <main>
        {healthError && (
          <div className="banner">
            Can&apos;t reach the API at <code>{API_URL}</code>. Start it from <code>system/backend</code>:{" "}
            <code>../../.venv/bin/uvicorn app:app --port 8000</code>{" "}
            <button className="btn" onClick={checkHealth} style={{ marginLeft: 8 }}>Retry</button>
          </div>
        )}

        <div className="toolbar">
          <button className="btn primary" onClick={() => fileRef.current?.click()}>Open image</button>
          <input
            ref={fileRef} type="file" accept="image/*,.heic,.heif" hidden
            onChange={(e) => { const f = e.target.files?.[0]; if (f) loadFile(f); e.target.value = ""; }}
          />
          <button
            className="btn" disabled={!image}
            onClick={() => image && addBox({ x: 0, y: 0, w: image.bitmap.width, h: image.bitmap.height })}
          >
            Whole image is one line
          </button>
          <button
            className="btn" disabled={!image || !health?.detector || detectState.busy}
            title={health && !health.detector ? "No detector checkpoint (checkpoints/best_det.pt) on the server" : undefined}
            onClick={() => detect()}
          >
            {detectState.busy ? "Detecting…" : "Detect lines"}
          </button>
          <button className="btn" disabled={!boxes.length} onClick={() => recognizeMany(ordered)}>Re-run all</button>
          <button className="btn" disabled={!boxes.length} onClick={() => { setBoxes([]); setResults({}); setSelected(null); }}>
            Clear boxes
          </button>
          <button className="btn" disabled={!allText} onClick={() => copy(allText, "all")}>
            {copied === "all" ? "Copied" : "Copy all text"}
          </button>
          {detectState.msg && <span className="hint" role="status"><b>{detectState.msg}</b></span>}
          <span className="hint">
            {image ? "Lines are detected automatically. Drag on the image to add a missed line. Drag a box to move it, its corners to resize, Delete to remove." : "Open, drop or paste (⌘V / Ctrl+V) a page photo (JPEG, PNG or iPhone HEIC): every line is detected and read."}
          </span>
        </div>

        <div className="workspace">
          {image ? (
            <div className={`canvas-wrap ${dragOver ? "over" : ""}`} {...dropProps}>
              <div className="stage">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={image.url} alt={image.name} />
                <svg
                  ref={svgRef}
                  viewBox={`0 0 ${image.bitmap.width} ${image.bitmap.height}`}
                  preserveAspectRatio="none"
                  onPointerDown={onPointerDown}
                  onPointerMove={onPointerMove}
                  onPointerUp={onPointerUp}
                  onPointerCancel={onPointerUp}
                >
                  {ordered.map((b, i) => {
                    const sel = b.id === selected;
                    return (
                      <g key={b.id}>
                        <rect className={`box ${sel ? "sel" : ""}`} data-id={b.id} x={b.x} y={b.y} width={b.w} height={b.h} vectorEffect="non-scaling-stroke" />
                        <rect className={`tag ${sel ? "sel" : ""}`} x={b.x} y={b.y} width={fs * (String(i + 1).length * 0.7 + 1)} height={fs * 1.5} rx={3 * scale} pointerEvents="none" />
                        <text x={b.x + fs * 0.5} y={b.y + fs * 1.1} fontSize={fs}>{i + 1}</text>
                        {sel && ["lt", "rt", "lb", "rb"].map((c) => (
                          <rect
                            key={c} className="handle" data-id={b.id} data-corner={c}
                            x={(c[0] === "l" ? b.x : b.x + b.w) - hs / 2} y={(c[1] === "t" ? b.y : b.y + b.h) - hs / 2}
                            width={hs} height={hs} vectorEffect="non-scaling-stroke"
                            style={{ cursor: c === "lt" || c === "rb" ? "nwse-resize" : "nesw-resize" }}
                          />
                        ))}
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>
          ) : (
            <div className={`dropzone ${dragOver ? "over" : ""}`} onClick={() => fileRef.current?.click()} {...dropProps}>
              <div>
                <strong>Drop an image here</strong>
                or click to choose a file, or paste from the clipboard.
                <br />A page photo (then box each line) or an already cropped line.
              </div>
            </div>
          )}

          <div className="results">
            {!ordered.length && <div className="empty">Recognized text will appear here, one card per box.</div>}
            {ordered.map((b, i) => {
              const r = results[b.id];
              const stale = r && r.geom !== geomOf(b);
              return (
                <div key={b.id} className={`card ${b.id === selected ? "sel" : ""}`} onClick={() => setSelected(b.id)}>
                  <div className="card-head">
                    <span className="num">{i + 1}</span>
                    {r?.status === "done" && <span>{r.ms} ms · {health?.decoder ?? ""}</span>}
                    <span className="spacer" />
                    {r?.text && (
                      <button className="icon-btn" onClick={(e) => { e.stopPropagation(); copy(r.text!, `b${b.id}`); }}>
                        {copied === `b${b.id}` ? "Copied" : "Copy"}
                      </button>
                    )}
                    <button className="icon-btn" onClick={(e) => { e.stopPropagation(); removeBox(b.id); }} aria-label={`Remove box ${i + 1}`}>
                      Remove
                    </button>
                  </div>
                  {r?.thumb && <img className="thumb" src={r.thumb} alt={`Crop ${i + 1}`} />}
                  {!r || r.status === "pending" || stale ? (
                    <div className="text pending">{drag.current?.id === b.id ? "Release to read…" : "Reading…"}</div>
                  ) : r.status === "error" ? (
                    <div className="text error">Error: {r.error}</div>
                  ) : (
                    <>
                      <div className="text km">{r.text || <span className="text pending">(no text found)</span>}</div>
                      {r.attn !== r.ctc && (
                        <details className="alt" onClick={(e) => e.stopPropagation()}>
                          <summary>Decoders disagree: compare</summary>
                          <dl>
                            <dt>CTC</dt><dd className="km">{r.ctc || "—"}</dd>
                            <dt>Attention</dt><dd className="km">{r.attn || "—"}</dd>
                          </dl>
                        </details>
                      )}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </main>
    </>
  );
}
