// 2-player live scoring for pic_1 (left/right halves scored independently).
//   Class score   = CLIP zero-shot on each half's silhouette (#maskL / #maskR) vs the target noun
//   Emotion score = face-api expression on each half's camera (#camL / #camR)
//   Total = weighted score from 0..100. CLIP class is remapped from 0..40% to 0..100%.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

env.allowLocalModels = false;
env.useBrowserCache = true;

const FACE_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";
const CLASS_RAW_MAX = 0.40;
const CLASS_WEIGHT = 2;
const EMOTION_WEIGHT = 1;
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
const lastThreshold = { L: 0, R: 0 };
let scoreTarget = { noun: null, emotion: null };

function setStatus(t) {
  const el = $("scoreStatus");
  if (el) el.textContent = t;
}

function ensureModels() {
  if (!modelsPromise) {
    modelsPromise = (async () => {
      setStatus("Loading CLIP model...");
      pylog("loading CLIP...");
      clip = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32", { quantized: true });
      pylog("CLIP loaded");
      if (window.faceapi) {
        setStatus("Loading face model...");
        pylog("loading face model...");
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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function scoreBreakdown(cls, emoScore) {
  const classNorm = clamp(cls / CLASS_RAW_MAX, 0, 1);
  const classPct = lerp(0, 100, classNorm);
  const emoPct = clamp(emoScore, 0, 1) * 100;
  const total = (
    (classPct * CLASS_WEIGHT) +
    (emoPct * EMOTION_WEIGHT)
  ) / (CLASS_WEIGHT + EMOTION_WEIGHT);
  return {
    cls,
    classNorm,
    classPct,
    emoScore,
    emoPct,
    total: clamp(total, 0, 100),
  };
}

function crossedThresholds(side, total) {
  const current = Math.min(100, Math.floor(total / 10) * 10);
  const previous = lastThreshold[side] || 0;
  if (current <= previous) return [];

  const crossed = [];
  for (let threshold = previous + 10; threshold <= current; threshold += 10) {
    crossed.push(threshold);
  }
  lastThreshold[side] = current;
  return crossed;
}

function emitScoreEvent(side, breakdown) {
  const thresholds = crossedThresholds(side, breakdown.total);
  window.dispatchEvent(new CustomEvent("score-update", {
    detail: {
      side,
      thresholds,
      total: breakdown.total,
      classPct: breakdown.classPct,
      rawClassPct: breakdown.cls * 100,
      emoPct: breakdown.emoPct,
    },
  }));
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
  const noun = (scoreTarget.noun || $("nounTarget").value || "mushroom").trim();
  const emo = scoreTarget.emotion || $("emoTarget").value;
  const maskCanvas = $("mask" + side);
  const camCanvas = $("cam" + side);

  let cls = 0;
  if (clip && maskCanvas && maskCanvas.width) {
    const sil = silhouetteFromMask(maskCanvas, 224);
    const res = await clip(sil.toDataURL("image/jpeg", 0.9), [
      `a shadow puppet silhouette of a ${noun}`,
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
  const breakdown = scoreBreakdown(cls, emoScore);
  $("classScore" + side).textContent = (cls * 100).toFixed(0) + "%";
  $("classScore" + side).title = "Normalized: " + breakdown.classPct.toFixed(1) + "%";
  $("emoScore" + side).textContent = breakdown.emoPct.toFixed(0) + "%";
  $("totalScore" + side).textContent = breakdown.total.toFixed(0);
  emitScoreEvent(side, breakdown);
  return breakdown;
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

window.resetScoreEffects = () => {
  lastThreshold.L = 0;
  lastThreshold.R = 0;
};

window.setScoreTarget = (noun, emotion) => {
  scoreTarget = {
    noun: (noun || "object").trim(),
    emotion: emotion || "neutral",
  };
  pylog("target locked: " + scoreTarget.emotion + " " + scoreTarget.noun);
};

function snapCanvas(c) {
  return (c && c.width) ? c.toDataURL("image/png") : null;
}

async function captureCurrentBest() {
  const best = { L: null, R: null };
  for (const side of ["L", "R"]) {
    const { cls, emoScore } = await evaluate(side);
    const breakdown = updatePanel(side, cls, emoScore);
    best[side] = {
      ...breakdown,
      dataUrl: snapCanvas($("cam" + side)),
      maskUrl: snapCanvas($("mask" + side)),
    };
  }
  return best;
}

// Record for `seconds`; keep EACH player's highest-Total frame (+ its mask).
// -> { L: {dataUrl, maskUrl, total, cls, emoScore}|null, R: {...}|null }
window.recordBest2 = async (seconds) => {
  try { await ensureModels(); } catch (e) { console.warn(e); return null; }
  if (seconds <= 0) return captureCurrentBest();

  const prevPaused = paused;
  paused = true; // suspend the periodic loop while recording
  const best = { L: null, R: null };
  const endAt = performance.now() + seconds * 1000;
  try {
    while (performance.now() < endAt) {
      for (const side of ["L", "R"]) {
        const { cls, emoScore } = await evaluate(side);
        const breakdown = updatePanel(side, cls, emoScore);
        if (!best[side] || breakdown.total > best[side].total) {
          best[side] = {
            ...breakdown,
            dataUrl: snapCanvas($("cam" + side)),
            maskUrl: snapCanvas($("mask" + side)),
          };
        }
      }
      setStatus("Recording... " + Math.max(0, Math.ceil((endAt - performance.now()) / 1000)) + "s");
    }
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
