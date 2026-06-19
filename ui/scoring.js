// 2-player live scoring for pic_1 (left/right halves scored independently).
//   Class score   = CLIP zero-shot on each half's silhouette (#maskL / #maskR) vs the target noun
//   Emotion score = face-api expression on each half's camera (#camL / #camR)
//   Total = class*2 + emotion*1   (class/emotion shown as 0..100, i.e. cls*200 + emo*100)
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

env.allowLocalModels = false;
env.useBrowserCache = true;

const FACE_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";
const $ = (id) => document.getElementById(id);

function pylog(m) {
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.log) {
      window.pywebview.api.log("[score] " + m);
    }
  } catch (e) { /* api not ready yet */ }
}

let clip = null;
let faceReady = false;
let modelsPromise = null;
let running = false;
let busy = false;
let paused = false;
let timer = null;

function setStatus(t) {
  const el = $("scoreStatus");
  if (el) el.textContent = t;
}

function ensureModels() {
  if (!modelsPromise) {
    modelsPromise = (async () => {
      setStatus("Loading CLIP model…");
      pylog("loading CLIP…");
      clip = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32", { quantized: true });
      pylog("CLIP loaded");
      if (window.faceapi) {
        setStatus("Loading face model…");
        pylog("loading face model…");
        await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_URL);
        await faceapi.nets.faceExpressionNet.loadFromUri(FACE_MODEL_URL);
        faceReady = true;
        pylog("face model loaded");
      }
      setStatus("");
    })();
  }
  return modelsPromise;
}

// Convert a white-on-black mask canvas into a black-silhouette-on-white square (for CLIP).
function silhouetteFromMask(maskCanvas, size) {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, size, size);
  const mw = maskCanvas.width || 1, mh = maskCanvas.height || 1;
  const scale = Math.max(size / mw, size / mh);
  const dw = mw * scale, dh = mh * scale;
  ctx.globalCompositeOperation = "difference"; // white - (white person) -> black silhouette
  ctx.drawImage(maskCanvas, (size - dw) / 2, (size - dh) / 2, dw, dh);
  ctx.globalCompositeOperation = "source-over";
  return c;
}

// Score one half: class on #mask{side} silhouette, emotion on #cam{side} face.
async function evaluate(side) {
  const noun = ($("nounTarget").value || "mushroom").trim();
  const emo = $("emoTarget").value;
  const maskCanvas = $("mask" + side);
  const camCanvas = $("cam" + side);

  let cls = 0;
  if (clip && maskCanvas && maskCanvas.width) {
    const sil = silhouetteFromMask(maskCanvas, 224);
    const res = await clip(sil.toDataURL("image/jpeg", 0.9), [
      `a black silhouette of a ${noun}`,
      "a black silhouette of a person",
      "a plain white image",
    ]);
    const hit = res.find((r) => r.label.toLowerCase().includes(noun.toLowerCase()));
    cls = hit ? hit.score : 0;
  }

  let emoScore = 0;
  if (faceReady && camCanvas && camCanvas.width) {
    const det = await faceapi
      .detectAllFaces(camCanvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 }))
      .withFaceExpressions();
    if (det.length) {
      const best = det.reduce((a, b) => {
        const aa = a.detection.box.width * a.detection.box.height;
        const bb = b.detection.box.width * b.detection.box.height;
        return aa >= bb ? a : b;
      });
      emoScore = best.expressions[emo] || 0;
    }
  }
  return { cls, emoScore };
}

function updatePanel(side, cls, emoScore) {
  const total = cls * 200 + emoScore * 100; // class*2 + emotion*1
  $("classScore" + side).textContent = (cls * 100).toFixed(1) + "%";
  $("emoScore" + side).textContent = (emoScore * 100).toFixed(1) + "%";
  $("totalScore" + side).textContent = total.toFixed(1);
}

async function scoreOnce() {
  if (busy || paused || !running || !clip) return;
  busy = true;
  try {
    for (const side of ["L", "R"]) {
      const { cls, emoScore } = await evaluate(side);
      updatePanel(side, cls, emoScore);
    }
  } catch (e) {
    console.warn("score error", e);
  } finally {
    busy = false;
  }
}

window.startScoring = async () => {
  running = true;
  try {
    await ensureModels();
  } catch (e) {
    setStatus("model load failed (see console)");
    console.warn(e);
    return;
  }
  if (!timer) timer = setInterval(scoreOnce, 1500);
};

window.stopScoring = () => { running = false; };

// Freeze/resume the displayed scores (e.g. while a still photo is shown).
window.setScoringPaused = (p) => { paused = !!p; };

function snapCanvas(c) {
  return (c && c.width) ? c.toDataURL("image/png") : null;
}

// Record for `seconds`; keep EACH player's highest-Total frame (+ its mask).
// -> { L: {dataUrl, maskUrl, total, cls, emoScore}|null, R: {...}|null }
window.recordBest2 = async (seconds) => {
  try { await ensureModels(); } catch (e) { console.warn(e); return null; }
  const prevPaused = paused;
  paused = true; // suspend the periodic loop while recording
  const best = { L: null, R: null };
  const endAt = performance.now() + seconds * 1000;
  try {
    do { // at least one pass, so Record = 0 still captures one frame per player
      for (const side of ["L", "R"]) {
        const { cls, emoScore } = await evaluate(side);
        updatePanel(side, cls, emoScore);
        const total = cls * 200 + emoScore * 100;
        if (!best[side] || total > best[side].total) {
          best[side] = {
            total, cls, emoScore,
            dataUrl: snapCanvas($("cam" + side)),
            maskUrl: snapCanvas($("mask" + side)),
          };
        }
      }
      setStatus(seconds > 0
        ? "Recording… " + Math.max(0, Math.ceil((endAt - performance.now()) / 1000)) + "s"
        : "");
    } while (performance.now() < endAt);
  } finally {
    paused = prevPaused;
    setStatus("");
  }
  return best;
};

// Preload models at app launch so they're ready before the user clicks Start.
ensureModels()
  .then(() => pylog("models preloaded"))
  .catch((e) => { console.warn("model preload failed", e); pylog("preload failed"); });
