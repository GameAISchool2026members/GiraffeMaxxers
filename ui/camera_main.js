import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

env.allowLocalModels = false;
env.useBrowserCache = true;

const $ = (id) => document.getElementById(id);

const CAMERA_WIDTH = 1280;
const CAMERA_HEIGHT = 720;
const SEGMENTATION_MAX_WIDTH = 320;
const SEGMENTATION_INTERVAL_MS = 130;
const SCORE_INTERVAL_MS = 850;
const ROUND_DURATION_SECONDS = 15;
const FACE_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";
const CLASS_RAW_MAX = 0.40;
const CLASS_WEIGHT = 2;
const EMOTION_WEIGHT = 1;

const TARGET_OBJECTS = [
  "mushroom",
  "robot",
  "tree",
  "umbrella",
  "guitar",
  "teapot",
  "chair",
  "cactus",
  "airplane",
  "banana",
  "backpack",
  "statue",
  "sword",
  "trophy",
  "snowman",
  "rocket",
  "dragon",
  "wizard",
  "pirate",
  "dinosaur",
  "ghost",
  "alien",
  "knight",
  "penguin",
  "octopus",
  "shark",
  "butterfly",
  "spider",
  "elephant",
  "giraffe",
  "monkey",
  "frog",
  "snail",
  "rabbit",
  "cat",
  "dog",
  "horse",
  "bee",
  "fish",
  "crab",
  "lobster",
  "turtle",
  "flamingo",
  "peacock",
  "swan",
  "pumpkin",
  "castle",
  "car",
  "train",
  "bicycle",
  "motorbike",
  "boat",
  "submarine",
  "helicopter",
  "spaceship",
  "satellite",
  "camera",
  "phone",
  "laptop",
  "book",
  "pencil",
  "paintbrush",
  "hammer",
  "key",
  "crown",
  "hat",
  "shoe",
  "glove",
  "scarf",
  "balloon",
  "kite",
  "drum",
  "violin",
  "piano",
  "microphone",
  "pizza",
  "burger",
  "ice cream",
  "cupcake",
  "apple",
  "pineapple",
  "watermelon",
  "flower",
  "plant pot",
  "lamp",
  "clock",
  "mirror",
  "door",
  "bridge",
  "mountain",
  "volcano",
  "cloud",
  "lightning bolt",
  "moon",
  "sun",
];
const TARGET_EMOTIONS = ["angry", "happy", "sad", "surprised", "fearful", "disgusted", "neutral"];

let stream = null;
let clip = null;
let faceReady = false;
let mediaPipeSeg = null;
let latestMediaPipeMask = null;
let renderRaf = null;
let segBusy = false;
let scoringBusy = false;
let scoringActive = false;
let roundActive = false;
let roundEnding = false;
let roundEndAt = 0;
let lastSegAt = 0;
let lastScoreAt = 0;
let challenge = { object: "robot", emotion: "happy" };
let latestScores = { L: null, R: null };
let latestSegmentPairs = { L: null, R: null };
let pendingSegmentationFrame = null;
let pendingFullFrame = null;
let bestEntries = { L: null, R: null };

function pylog(message) {
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.log) {
      window.pywebview.api.log(message);
    }
  } catch (e) {
    // Python bridge may not be ready while the page is booting.
  }
}

function randomChoice(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function chooseChallenge() {
  challenge = {
    object: randomChoice(TARGET_OBJECTS),
    emotion: randomChoice(TARGET_EMOTIONS),
  };
  $("challengeEmotion").textContent = challenge.emotion;
  $("challengeObject").textContent = challenge.object;
  pylog("challenge: " + challenge.emotion + " " + challenge.object);
}

function setLoadingStatus(message) {
  $("loadStatus").textContent = message;
}

function setRuntimeStatus(message) {
  $("runtimeStatus").textContent = message || "";
}

function shortError(error) {
  const message = String((error && error.message) || error || "unknown error");
  return message.length > 260 ? message.slice(0, 220) + "...<redacted>" : message;
}

function setTimer(seconds) {
  $("roundTimer").textContent = String(Math.max(0, Math.ceil(seconds)));
}

async function openCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: CAMERA_WIDTH },
      height: { ideal: CAMERA_HEIGHT },
      frameRate: { ideal: 30, max: 30 },
      facingMode: "user",
    },
    audio: false,
  });
  $("preview").srcObject = stream;
  await $("preview").play();
}

async function setupMediaPipeSegmentation() {
  if (!window.SelfieSegmentation) {
    throw new Error("MediaPipe Selfie Segmentation did not load.");
  }
  if (mediaPipeSeg) return;

  mediaPipeSeg = new SelfieSegmentation({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
  });
  mediaPipeSeg.setOptions({ modelSelection: 1, selfieMode: false });
  mediaPipeSeg.onResults((results) => {
    latestMediaPipeMask = results.segmentationMask;
    latestSegmentPairs = buildSegmentPairs(results.segmentationMask, pendingSegmentationFrame);
  });
}

async function loadScoringModels() {
  clip = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32", { quantized: true });
  if (window.faceapi) {
    await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_URL);
    await faceapi.nets.faceExpressionNet.loadFromUri(FACE_MODEL_URL);
    faceReady = true;
  }
}

function fitCanvasToElement(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function drawSegmentationInputCanvas() {
  const video = $("preview");
  const sourceWidth = video.videoWidth || CAMERA_WIDTH;
  const sourceHeight = video.videoHeight || CAMERA_HEIGHT;
  const scale = Math.min(1, SEGMENTATION_MAX_WIDTH / Math.max(1, sourceWidth));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = $("segInput");
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.drawImage(video, 0, 0, width, height);

  pendingFullFrame = document.createElement("canvas");
  pendingFullFrame.width = sourceWidth;
  pendingFullFrame.height = sourceHeight;
  pendingFullFrame.getContext("2d", { alpha: false }).drawImage(video, 0, 0, sourceWidth, sourceHeight);

  pendingSegmentationFrame = document.createElement("canvas");
  pendingSegmentationFrame.width = width;
  pendingSegmentationFrame.height = height;
  pendingSegmentationFrame.getContext("2d", { alpha: false }).drawImage(canvas, 0, 0);
  return canvas;
}

async function segmentCurrentFrame() {
  if (!mediaPipeSeg || segBusy) return;
  segBusy = true;
  try {
    await mediaPipeSeg.send({ image: drawSegmentationInputCanvas() });
  } catch (e) {
    setRuntimeStatus("Segmentation error: " + e.message);
    pylog("segmentation error: " + e);
  } finally {
    segBusy = false;
    lastSegAt = performance.now();
  }
}

function drawRealHalf(canvas, side) {
  const video = $("preview");
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return;

  fitCanvasToElement(canvas);
  const ctx = canvas.getContext("2d", { alpha: false });
  const halfWidth = Math.floor(width / 2);
  const sx = side === "L" ? halfWidth : 0;
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
  ctx.drawImage(video, sx, 0, halfWidth, height, 0, 0, canvas.width, canvas.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function drawSilhouetteHalf(canvas, side) {
  fitCanvasToElement(canvas);
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.fillStyle = "#05070b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!latestMediaPipeMask) {
    ctx.fillStyle = "rgba(255, 255, 255, .10)";
    ctx.font = Math.floor(canvas.width * 0.045) + "px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Waiting for silhouette...", canvas.width / 2, canvas.height / 2);
    return;
  }

  const source = latestMediaPipeMask;
  const sourceWidth = source.width || $("segInput").width || SEGMENTATION_MAX_WIDTH;
  const sourceHeight = source.height || $("segInput").height || Math.round(SEGMENTATION_MAX_WIDTH * 0.75);
  const halfWidth = Math.floor(sourceWidth / 2);
  const sx = side === "L" ? halfWidth : 0;

  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
  ctx.drawImage(source, sx, 0, halfWidth, sourceHeight, 0, 0, canvas.width, canvas.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  thresholdMaskCanvas(canvas);
}

function thresholdMaskCanvas(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < imageData.data.length; i += 4) {
    const value = imageData.data[i] >= 128 ? 255 : 0;
    imageData.data[i] = value;
    imageData.data[i + 1] = value;
    imageData.data[i + 2] = value;
    imageData.data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function halfCanvasFromSource(source, side, threshold) {
  if (!source) return null;
  const sourceWidth = source.width || SEGMENTATION_MAX_WIDTH;
  const sourceHeight = source.height || Math.round(SEGMENTATION_MAX_WIDTH * 0.75);
  const halfWidth = Math.floor(sourceWidth / 2);
  if (!halfWidth || !sourceHeight) return null;

  const canvas = document.createElement("canvas");
  canvas.width = halfWidth;
  canvas.height = sourceHeight;
  const ctx = canvas.getContext("2d", { alpha: false });
  const sx = side === "L" ? halfWidth : 0;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, halfWidth, sourceHeight);
  ctx.setTransform(-1, 0, 0, 1, halfWidth, 0);
  ctx.drawImage(source, sx, 0, halfWidth, sourceHeight, 0, 0, halfWidth, sourceHeight);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (threshold) thresholdMaskCanvas(canvas);
  return canvas;
}

function buildSegmentPairs(maskSource, imageSource) {
  if (!maskSource || !imageSource) return latestSegmentPairs;
  const fullFrame = pendingFullFrame || imageSource;
  return {
    L: buildSegmentPairForSide("L", fullFrame, maskSource),
    R: buildSegmentPairForSide("R", fullFrame, maskSource),
  };
}

function buildSegmentPairForSide(side, imageSource, maskSource) {
  const imageCanvas = halfCanvasFromSource(imageSource, side, false);
  const maskCanvas = halfCanvasFromSource(maskSource, side, true);
  if (!imageCanvas || !maskCanvas) return null;

  const alignedMask = document.createElement("canvas");
  alignedMask.width = imageCanvas.width;
  alignedMask.height = imageCanvas.height;
  const ctx = alignedMask.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, alignedMask.width, alignedMask.height);
  ctx.drawImage(maskCanvas, 0, 0, alignedMask.width, alignedMask.height);
  thresholdMaskCanvas(alignedMask);

  return {
    dataUrl: imageCanvas.toDataURL("image/png"),
    maskUrl: alignedMask.toDataURL("image/png"),
  };
}

function silhouetteForClip(maskCanvas, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, size, size);

  const mw = maskCanvas.width || 1;
  const mh = maskCanvas.height || 1;
  const scale = Math.max(size / mw, size / mh);
  const dw = mw * scale;
  const dh = mh * scale;
  ctx.globalCompositeOperation = "difference";
  ctx.drawImage(maskCanvas, (size - dw) / 2, (size - dh) / 2, dw, dh);
  ctx.globalCompositeOperation = "source-over";
  return canvas;
}

function scoreBreakdown(cls, emoScore) {
  const classPct = lerp(0, 100, clamp(cls / CLASS_RAW_MAX, 0, 1));
  const emoPct = clamp(emoScore, 0, 1) * 100;
  const total = ((classPct * CLASS_WEIGHT) + (emoPct * EMOTION_WEIGHT)) / (CLASS_WEIGHT + EMOTION_WEIGHT);
  return {
    cls,
    classPct,
    emoScore,
    emoPct,
    total: clamp(total, 0, 100),
  };
}

async function evaluateSide(side) {
  let cls = 0;
  if (clip) {
    const sil = silhouetteForClip($("sil" + side), 224);
    const results = await clip(sil.toDataURL("image/jpeg", 0.9), [
      "a shadow puppet silhouette of a " + challenge.object,
      "a black silhouette of a person",
      "a plain white image",
    ]);
    const hit = results.find((result) => result.label.toLowerCase().includes(challenge.object.toLowerCase()));
    cls = hit ? hit.score : 0;
  }

  let emoScore = 0;
  if (faceReady) {
    const detections = await faceapi
      .detectAllFaces($("pip" + side), new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 }))
      .withFaceExpressions();
    if (detections.length) {
      const best = detections.reduce((a, b) => {
        const aa = a.detection.box.width * a.detection.box.height;
        const bb = b.detection.box.width * b.detection.box.height;
        return aa >= bb ? a : b;
      });
      emoScore = best.expressions[challenge.emotion] || 0;
    }
  }

  return scoreBreakdown(cls, emoScore);
}

function setScore(side, breakdown) {
  latestScores[side] = breakdown;
  const objPct = Math.round(breakdown.cls * 100);
  const emoPct = Math.round(breakdown.emoPct);
  const hueObj = Math.round(lerp(0, 125, objPct / 100));
  const hueEmo = Math.round(lerp(0, 125, emoPct / 100));
  const objEl = $("scoreObj" + side);
  const emoEl = $("scoreEmo" + side);
  objEl.textContent = objPct + "%";
  objEl.style.color = `hsl(${hueObj}, 92%, 58%)`;
  emoEl.textContent = emoPct + "%";
  emoEl.style.color = `hsl(${hueEmo}, 92%, 58%)`;
}

async function scoreLoopTick(now) {
  if (!scoringActive || scoringBusy || now - lastScoreAt < SCORE_INTERVAL_MS) return;
  scoringBusy = true;
  lastScoreAt = now;
  try {
    for (const side of ["L", "R"]) {
      const breakdown = await evaluateSide(side);
      setScore(side, breakdown);
      const pair = latestSegmentPairs[side];
      if (pair && pair.dataUrl && pair.maskUrl && (!bestEntries[side] || breakdown.total > bestEntries[side].score.total)) {
        bestEntries[side] = {
          dataUrl: pair.dataUrl,
          maskUrl: pair.maskUrl,
          score: breakdown,
        };
        pylog("best " + side + " = " + breakdown.total.toFixed(1));
      }
    }
    setRuntimeStatus("Scoring " + challenge.emotion + " " + challenge.object);
  } catch (e) {
    setRuntimeStatus("Scoring error: " + e.message);
    pylog("scoring error: " + e);
  } finally {
    scoringBusy = false;
  }
}

function snapshotImageSide(side) {
  const video = $("preview");
  const width = video.videoWidth;
  const height = video.videoHeight;
  const halfWidth = Math.floor(width / 2);
  if (!halfWidth || !height) return null;

  const canvas = document.createElement("canvas");
  canvas.width = halfWidth;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { alpha: false });
  const sx = side === "L" ? halfWidth : 0;
  ctx.setTransform(-1, 0, 0, 1, halfWidth, 0);
  ctx.drawImage(video, sx, 0, halfWidth, height, 0, 0, halfWidth, height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return canvas.toDataURL("image/png");
}

function snapshotMaskSide(side) {
  const source = latestMediaPipeMask;
  if (!source) return null;

  const sourceWidth = source.width || $("segInput").width || SEGMENTATION_MAX_WIDTH;
  const sourceHeight = source.height || $("segInput").height || Math.round(SEGMENTATION_MAX_WIDTH * 0.75);
  const halfWidth = Math.floor(sourceWidth / 2);
  if (!halfWidth || !sourceHeight) return null;

  const canvas = document.createElement("canvas");
  canvas.width = halfWidth;
  canvas.height = sourceHeight;
  const ctx = canvas.getContext("2d", { alpha: false });
  const sx = side === "L" ? halfWidth : 0;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, halfWidth, sourceHeight);
  ctx.setTransform(-1, 0, 0, 1, halfWidth, 0);
  ctx.drawImage(source, sx, 0, halfWidth, sourceHeight, 0, 0, halfWidth, sourceHeight);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  thresholdMaskCanvas(canvas);
  return canvas.toDataURL("image/png");
}

function snapshotFinalPairs() {
  return {
    L: bestEntries.L || {
      dataUrl: (latestSegmentPairs.L || {}).dataUrl || snapshotImageSide("L"),
      maskUrl: (latestSegmentPairs.L || {}).maskUrl || snapshotMaskSide("L"),
      score: latestScores.L,
    },
    R: bestEntries.R || {
      dataUrl: (latestSegmentPairs.R || {}).dataUrl || snapshotImageSide("R"),
      maskUrl: (latestSegmentPairs.R || {}).maskUrl || snapshotMaskSide("R"),
      score: latestScores.R,
    },
  };
}

function stopCamera() {
  if (renderRaf) {
    cancelAnimationFrame(renderRaf);
    renderRaf = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
}

function showResults(snapshot) {
  $("gameView").classList.add("hidden");
  $("resultView").classList.remove("hidden");
  $("resultEmotion").textContent = challenge.emotion;
  $("resultObject").textContent = challenge.object;

  ["L", "R"].forEach((side) => {
    $("resultCard" + side).classList.remove("winner");
    $("morphStage" + side).classList.remove("morph-ready");
    $("morphStage" + side).classList.remove("gif-ready");
    $("resultOriginal" + side).src = snapshot[side].dataUrl || "";
    $("resultImg" + side).removeAttribute("src");
    $("resultScoreObj" + side).textContent = Math.round(((snapshot[side].score || {}).cls || 0) * 100) + "%";
    $("resultScoreEmo" + side).textContent = Math.round((snapshot[side].score || {}).emoPct || 0) + "%";
  });

  const leftSum = ((snapshot.L.score || {}).cls || 0) * 100 + ((snapshot.L.score || {}).emoPct || 0);
  const rightSum = ((snapshot.R.score || {}).cls || 0) * 100 + ((snapshot.R.score || {}).emoPct || 0);
  if (leftSum >= rightSum) $("resultCardL").classList.add("winner");
  if (rightSum >= leftSum) $("resultCardR").classList.add("winner");
}

function generationPrompt() {
  return (
    "A " + challenge.emotion + " " + challenge.object + ". "
  );
}

async function generateFinalImages(snapshot) {
  if (!window.pywebview || !window.pywebview.api) {
    $("generationStatus").textContent = "BFL backend is not ready.";
    return;
  }

  $("generationStatus").textContent = "Generating infilled images...";
  try {
    const prompt = generationPrompt();
    if (!window.pywebview.api.generate_flux_people) {
      $("generationStatus").textContent = "BFL backend is not ready.";
      return;
    }
    const result = await window.pywebview.api.generate_flux_people(
      snapshot.L.dataUrl,
      snapshot.L.maskUrl,
      snapshot.R.dataUrl,
      snapshot.R.maskUrl,
      prompt,
      prompt,
      prompt,
    );

    if (!result || !result.ok) {
      $("generationStatus").textContent = "Generation skipped: " + ((result && result.error) || "unknown error");
      return;
    }

    (result.items || []).forEach((item) => {
      if (item.side === "L" || item.side === "R") {
        $("resultImg" + item.side).src = item.url;
        $("morphStage" + item.side).classList.add("morph-ready");
      }
    });
    $("generationStatus").textContent = "Generated infilled images.";
    if (result.error) $("generationStatus").textContent += " Some generations failed: " + result.error;
  } catch (e) {
    $("generationStatus").textContent = "Generation failed: " + shortError(e);
    pylog("generation error: " + shortError(e));
  }
}

async function endRound() {
  roundEnding = true;
  roundActive = false;
  scoringActive = false;
  setTimer(0);
  setRuntimeStatus("Time.");

  const snapshot = snapshotFinalPairs();
  stopCamera();
  showResults(snapshot);
  await generateFinalImages(snapshot);
}

function updateRoundTimer(now) {
  if (!roundActive) return;
  const remaining = Math.max(0, (roundEndAt - now) / 1000);
  setTimer(remaining);
  if (remaining <= 0 && !roundEnding) {
    endRound();
  }
}

function renderLoop(now) {
  renderRaf = requestAnimationFrame(renderLoop);
  const video = $("preview");
  if (!stream || !video.videoWidth) return;

  drawRealHalf($("pipL"), "L");
  drawRealHalf($("pipR"), "R");
  drawSilhouetteHalf($("silL"), "L");
  drawSilhouetteHalf($("silR"), "R");

  if (!segBusy && now - lastSegAt > SEGMENTATION_INTERVAL_MS) {
    segmentCurrentFrame();
  }
  updateRoundTimer(now);
  scoreLoopTick(now);
}

async function countdown() {
  const el = $("countdown");
  el.classList.remove("hidden");
  for (const value of ["3", "2", "1"]) {
    el.textContent = value;
    await new Promise((resolve) => setTimeout(resolve, 850));
  }
  el.classList.add("hidden");
}

async function startGame() {
  const startBtn = $("startBtn");
  startBtn.disabled = true;
  startBtn.textContent = "Loading";
  scoringActive = false;
  roundActive = false;
  roundEnding = false;
  latestScores = { L: null, R: null };
  latestSegmentPairs = { L: null, R: null };
  bestEntries = { L: null, R: null };
  pendingSegmentationFrame = null;
  pendingFullFrame = null;
  lastScoreAt = 0;
  lastSegAt = 0;
  latestMediaPipeMask = null;
  setTimer(ROUND_DURATION_SECONDS);
  chooseChallenge();

  try {
    $("resultView").classList.add("hidden");
    $("startView").classList.remove("hidden");
    $("gameView").classList.add("hidden");

    if (window.pywebview && window.pywebview.api && window.pywebview.api.new_session) {
      await window.pywebview.api.new_session();
    }

    setLoadingStatus("Opening camera...");
    await openCamera();

    setLoadingStatus("Loading silhouette model...");
    await setupMediaPipeSegmentation();

    setLoadingStatus("Loading scoring models...");
    await loadScoringModels();

    $("startView").classList.add("hidden");
    $("gameView").classList.remove("hidden");
    renderRaf = requestAnimationFrame(renderLoop);
    setRuntimeStatus("Get ready...");
    await countdown();
    scoringActive = true;
    roundActive = true;
    roundEndAt = performance.now() + ROUND_DURATION_SECONDS * 1000;
    setTimer(ROUND_DURATION_SECONDS);
    setRuntimeStatus("Scoring live");
  } catch (e) {
    startBtn.disabled = false;
    startBtn.textContent = "Start";
    setLoadingStatus("Could not start: " + shortError(e));
    pylog("start error: " + shortError(e));
  }
}

window.addEventListener("DOMContentLoaded", () => {
  $("startBtn").addEventListener("click", startGame);
  $("playAgainBtn").addEventListener("click", () => {
    $("resultView").classList.add("hidden");
    $("startView").classList.remove("hidden");
    $("startBtn").disabled = false;
    $("startBtn").textContent = "Start";
    setLoadingStatus("Press start when ready");
  });
  pylog("camera UI ready");
});

window.addEventListener("beforeunload", () => {
  if (renderRaf) cancelAnimationFrame(renderRaf);
  if (stream) stream.getTracks().forEach((track) => track.stop());
});
