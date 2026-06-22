const $ = (id) => document.getElementById(id);

const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 480;
const SEGMENTATION_MAX_WIDTH = 256;
const SEGMENTATION_INTERVAL_MS = 160;
const YOLO_INTERVAL_MS = 350;
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
const TARGET_POSES = [
  "angry",
  "happy",
  "sad",
  "surprised",
  "fearful",
  "disgusted",
  "neutral",
];

function pylog(msg) {
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.log) {
      window.pywebview.api.log(msg);
    }
  } catch (e) {
    // Python bridge is not ready yet.
  }
}

let stream = null;
let currentIndex = 0;
let retaking = false;
let currentDataUrl = null;
let liveActive = false;
let liveRaf = null;
let segBusy = false;
let segBackend = "mediapipe";
let lastSeg = 0;
let mediaPipeSeg = null;
let latestMediaPipeMask = null;
let endPopupPending = false;
let fluxRunId = 0;
let audioCtx = null;
let gameEffectsActive = false;
let currentChallenge = null;
let recordingChallenge = null;

function show(viewId) {
  document.querySelectorAll(".view").forEach((view) => view.classList.add("hidden"));
  $(viewId).classList.remove("hidden");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function scoreFromEntry(entry) {
  if (!entry) return { classPct: 0, emoPct: 0, total: 0 };
  const classPct = typeof entry.classPct === "number"
    ? entry.classPct
    : lerp(0, 100, clamp((entry.cls || 0) / CLASS_RAW_MAX, 0, 1));
  const emoPct = typeof entry.emoPct === "number"
    ? entry.emoPct
    : clamp(entry.emoScore || 0, 0, 1) * 100;
  const total = typeof entry.total === "number"
    ? entry.total
    : ((classPct * CLASS_WEIGHT) + (emoPct * EMOTION_WEIGHT)) / (CLASS_WEIGHT + EMOTION_WEIGHT);
  return { classPct, emoPct, total: clamp(total, 0, 100) };
}

function randomChoice(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function normalizeObject(value) {
  return (value || "").trim().replace(/\s+/g, " ");
}

function resetScoreDisplays() {
  ["L", "R"].forEach((side) => {
    $("classScore" + side).textContent = "--";
    $("emoScore" + side).textContent = "--";
    $("totalScore" + side).textContent = "--";
  });
}

function setChallenge(challenge, animate) {
  currentChallenge = {
    object: normalizeObject(challenge.object) || "object",
    emotion: challenge.emotion || "neutral",
  };
  $("nounTarget").value = currentChallenge.object;
  $("emoTarget").value = currentChallenge.emotion;
  $("challengeStatus").textContent = currentChallenge.emotion + " " + currentChallenge.object;
  if (animate) {
    $("vsTargets").classList.remove("challenge-pop");
    void $("vsTargets").offsetWidth;
    $("vsTargets").classList.add("challenge-pop");
  }
}

function updateChallengeFromInputs(animate) {
  const object = normalizeObject($("nounTarget").value) || "object";
  const emotion = $("emoTarget").value || "neutral";
  setChallenge({ object, emotion }, !!animate);
  resetScoreDisplays();
  if (window.resetScoreEffects) window.resetScoreEffects();
  return currentChallenge;
}

function randomizeChallenge(options) {
  const opts = options || {};
  const object = randomChoice(TARGET_OBJECTS);
  const emotion = randomChoice(TARGET_POSES);
  setChallenge({ object, emotion }, opts.animate !== false);
  resetScoreDisplays();
  if (window.resetScoreEffects) window.resetScoreEffects();
  return currentChallenge;
}

function prepareRoundIdle() {
  gameEffectsActive = false;
  recordingChallenge = null;
  randomizeChallenge({ animate: false });
  $("rerollTargetBtn").disabled = false;
  resetScoreDisplays();
  if (window.resetScoreEffects) window.resetScoreEffects();
}

function beginRecordingRound() {
  updateChallengeFromInputs(false);
  if (!currentChallenge) randomizeChallenge({ animate: false });
  recordingChallenge = { ...currentChallenge };
  setChallenge(recordingChallenge, true);
  $("rerollTargetBtn").disabled = true;
  $("nounTarget").disabled = true;
  $("emoTarget").disabled = true;
  if (window.setScoreTarget) {
    window.setScoreTarget(recordingChallenge.object, recordingChallenge.emotion);
  }
  gameEffectsActive = true;
  ensureAudio();
  playStartSound();
  pylog("recording challenge: " + recordingChallenge.emotion + " " + recordingChallenge.object);
}

function endRecordingRound() {
  gameEffectsActive = false;
  $("rerollTargetBtn").disabled = false;
  $("nounTarget").disabled = false;
  $("emoTarget").disabled = false;
}

function ensureAudio() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return null;
  if (!audioCtx) audioCtx = new AudioContextClass();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

function playTone(freq, start, duration, gainValue, type) {
  const ctx = ensureAudio();
  if (!ctx) return;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type || "sine";
  osc.frequency.setValueAtTime(freq, start);
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(gainValue, start + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(start);
  osc.stop(start + duration + 0.03);
}

function playStartSound() {
  const ctx = ensureAudio();
  if (!ctx) return;
  const start = ctx.currentTime + 0.01;
  playTone(196, start, 0.10, 0.04, "triangle");
  playTone(294, start + 0.08, 0.12, 0.05, "triangle");
  playTone(392, start + 0.16, 0.16, 0.05, "triangle");
}

function playFinishSound() {
  const ctx = ensureAudio();
  if (!ctx) return;
  const start = ctx.currentTime + 0.01;
  [392, 494, 587, 784].forEach((freq, index) => {
    playTone(freq, start + index * 0.08, 0.18, 0.045, "triangle");
  });
}

function playThresholdJingle(threshold, delayMs) {
  const ctx = ensureAudio();
  if (!ctx) return;
  const start = ctx.currentTime + (delayMs || 0) / 1000;
  const base = 220 + threshold * 6;
  [0, 4, 7].forEach((semi, index) => {
    const freq = base * Math.pow(2, semi / 12);
    playTone(freq, start + index * 0.055, 0.11, 0.035 + threshold / 5000, "sine");
  });
}

function burstStars(side, threshold) {
  const layer = $("effectsLayer");
  const stage = $("cam" + side).closest(".cam-stage");
  if (!layer || !stage) return;

  const rect = stage.getBoundingClientRect();
  const originX = rect.left + rect.width * 0.5;
  const originY = rect.top + rect.height * 0.35;
  const count = 8 + Math.floor(threshold / 10);

  for (let i = 0; i < count; i += 1) {
    const star = document.createElement("span");
    star.className = "star-pop";
    const angle = Math.random() * Math.PI * 2;
    const distance = 42 + Math.random() * 86;
    star.style.left = originX + "px";
    star.style.top = originY + "px";
    star.style.setProperty("--dx", Math.cos(angle) * distance + "px");
    star.style.setProperty("--dy", Math.sin(angle) * distance + "px");
    star.style.setProperty("--rot", (Math.random() * 220 - 110) + "deg");
    star.style.setProperty("--size", (8 + Math.random() * 13) + "px");
    layer.appendChild(star);
    setTimeout(() => star.remove(), 850);
  }
}

function pulseScore(side, threshold) {
  const score = $("totalScore" + side);
  score.classList.remove("score-pop");
  void score.offsetWidth;
  score.classList.add("score-pop");
  $("scoreStatus").textContent = "P" + (side === "L" ? "1" : "2") + " hit " + threshold + "%";
  setTimeout(() => score.classList.remove("score-pop"), 650);
}

function onScoreUpdate(event) {
  if (!gameEffectsActive) return;
  const detail = event.detail || {};
  const thresholds = detail.thresholds || [];
  thresholds.forEach((threshold, index) => {
    const delay = index * 115;
    setTimeout(() => {
      playThresholdJingle(threshold, 0);
      burstStars(detail.side, threshold);
      pulseScore(detail.side, threshold);
    }, delay);
  });
}

async function startCamera() {
  show("camera");
  currentIndex = 0;
  retaking = false;
  currentDataUrl = null;
  endPopupPending = false;
  fluxRunId += 1;
  resetToLive();
  prepareRoundIdle();

  try {
    const dir = await window.pywebview.api.new_session();
    pylog("session " + dir);
  } catch (e) {
    pylog("new_session error " + e);
  }

  $("camStatus").textContent = "Opening camera... (click Allow if prompted)";
  pylog("requesting camera (new session)");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: CAMERA_WIDTH },
        height: { ideal: CAMERA_HEIGHT },
        frameRate: { ideal: 24, max: 30 },
        facingMode: "user",
      },
      audio: false,
    });
    $("preview").srcObject = stream;
    await $("preview").play();

    segBackend = ($("segModel") && $("segModel").value) || "mediapipe";
    pylog("camera started");
    pylog("seg backend: " + segBackend);

    if (segBackend === "mediapipe") {
      await setupMediaPipeSegmentation();
    }

    liveActive = true;
    if (!liveRaf) renderHalves();
    $("camStatus").textContent = "Camera ready";
  } catch (e) {
    $("camStatus").textContent = "Cannot open camera: " + e.name + " - " + e.message;
    pylog("camera ERROR " + e.name + ": " + e.message);
  }
}

function stopCamera() {
  liveActive = false;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  $("preview").srcObject = null;
  if (window.stopScoring) window.stopScoring();
}

function resetToLive() {
  liveActive = true;
  if (window.setScoringPaused) window.setScoringPaused(false);
  $("photo").classList.add("hidden");
  $("singleStage").classList.add("hidden");
  $("camFull").classList.add("hidden");
  $("vsRow").classList.remove("hidden");
  $("vsTargets").classList.remove("hidden");
  $("shootBtn").classList.remove("hidden");
  $("uploadBtn").classList.remove("hidden");
  $("retakeBtn").classList.add("hidden");
  $("shootBtn").disabled = false;
}

function back() {
  stopCamera();
  resetToLive();
  show("home");
}

function showImage(dataUrl, statusText) {
  currentDataUrl = dataUrl;
  liveActive = false;
  if (window.setScoringPaused) window.setScoringPaused(true);
  $("vsRow").classList.add("hidden");
  $("vsTargets").classList.add("hidden");
  $("singleStage").classList.remove("hidden");
  $("camFull").classList.add("hidden");
  $("photo").src = dataUrl;
  $("photo").classList.remove("hidden");
  $("shootBtn").classList.add("hidden");
  $("uploadBtn").classList.add("hidden");
  $("retakeBtn").classList.remove("hidden");
  $("camStatus").textContent = statusText;
}

function onNewImage(dataUrl, label) {
  if (!retaking) currentIndex += 1;
  retaking = false;
  const name = "pic_" + currentIndex;
  showImage(dataUrl, label + " -> " + name);
  try {
    window.pywebview.api.save_image(dataUrl, name)
      .then((path) => {
        $("camStatus").textContent = name + " saved";
        pylog("saved " + path);
      })
      .catch((e) => pylog("save error " + e));
  } catch (e) {
    pylog("save error " + e);
  }
}

function capture() {
  const video = $("preview");
  const canvas = $("canvas");
  canvas.width = video.videoWidth || CAMERA_WIDTH;
  canvas.height = video.videoHeight || CAMERA_HEIGHT;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  pylog("photo captured " + canvas.width + "x" + canvas.height);
  onNewImage(canvas.toDataURL("image/png"), "Captured");
}

async function shoot() {
  const secs = Math.max(0, parseInt(($("delay") || {}).value, 10) || 0);
  if (window.recordBest2) {
    $("shootBtn").disabled = true;
    beginRecordingRound();
    $("camStatus").textContent = secs > 0
      ? "Recording " + secs + "s - keeping each player's best frame..."
      : "Scoring current frame...";
    let best = null;
    try {
      best = await window.recordBest2(secs);
      if (best && recordingChallenge) best.challenge = { ...recordingChallenge };
    } finally {
      $("shootBtn").disabled = false;
      if (!best) endRecordingRound();
    }
    if (best) finishRecord(best);
  } else {
    capture();
  }
}

function setScore(side, entry) {
  const score = scoreFromEntry(entry);
  $("classScore" + side).textContent = score.classPct.toFixed(0) + "%";
  $("emoScore" + side).textContent = score.emoPct.toFixed(0) + "%";
  $("totalScore" + side).textContent = score.total.toFixed(0);
}

function drawDataUrlToCanvas(canvas, dataUrl) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d").drawImage(img, 0, 0);
  };
  img.src = dataUrl;
}

function finishRecord(best) {
  liveActive = false;
  endRecordingRound();
  if (window.setScoringPaused) window.setScoringPaused(true);

  ["L", "R"].forEach((side) => {
    const entry = best[side];
    if (!entry || !entry.dataUrl) return;
    drawDataUrlToCanvas($("cam" + side), entry.dataUrl);
    if (entry.maskUrl) drawDataUrlToCanvas($("mask" + side), entry.maskUrl);
    setScore(side, entry);
    try {
      window.pywebview.api.save_image(entry.dataUrl, "pic_" + side);
      if (entry.maskUrl) window.pywebview.api.save_image(entry.maskUrl, "mask_" + side);
    } catch (e) {
      pylog("save best err " + e);
    }
  });

  const ft = (entry) => (entry ? scoreFromEntry(entry).total.toFixed(0) : "--");
  $("camStatus").textContent = "Best frames saved - P1 " + ft(best.L) + " - P2 " + ft(best.R);
  pylog("record2 best P1=" + ft(best.L) + " P2=" + ft(best.R));
  stopCamera();
  showEndScreen(best);
}

function onUpload(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    pylog("image uploaded: " + file.name);
    onNewImage(reader.result, "Uploaded");
  };
  reader.readAsDataURL(file);
  e.target.value = "";
}

function retake() {
  retaking = true;
  resetToLive();
  $("camStatus").textContent = "Retaking" + (stream ? "" : " (no camera)");
}

function drawCamHalf(canvas, video, side) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  const halfWidth = Math.floor(width / 2);
  if (!halfWidth) return;
  if (canvas.width !== halfWidth || canvas.height !== height) {
    canvas.width = halfWidth;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(-1, 0, 0, 1, halfWidth, 0);
  const sx = side === "L" ? halfWidth : 0;
  ctx.drawImage(video, sx, 0, halfWidth, height, 0, 0, halfWidth, height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function drawCamFull(video) {
  const canvas = $("camFull");
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width) return;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(-1, 0, 0, 1, width, 0);
  ctx.drawImage(video, 0, 0, width, height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

async function setupMediaPipeSegmentation() {
  if (mediaPipeSeg) return true;
  if (!window.SelfieSegmentation) {
    $("camStatus").textContent = "MediaPipe Selfie Segmentation did not load.";
    return false;
  }

  mediaPipeSeg = new SelfieSegmentation({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
  });
  mediaPipeSeg.setOptions({ modelSelection: 1, selfieMode: false });
  mediaPipeSeg.onResults((results) => {
    latestMediaPipeMask = results.segmentationMask;
    drawMediaPipeMasks();
  });
  return true;
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
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(video, 0, 0, width, height);
  return canvas;
}

async function segmentWithMediaPipe() {
  if (!mediaPipeSeg || segBusy) return;
  segBusy = true;
  try {
    await mediaPipeSeg.send({ image: drawSegmentationInputCanvas() });
  } catch (e) {
    pylog("mediapipe seg err " + e);
  } finally {
    segBusy = false;
    lastSeg = performance.now();
  }
}

function drawMediaPipeMasks() {
  if (!latestMediaPipeMask || !liveActive) return;
  drawMediaPipeMaskHalf($("maskL"), "L");
  drawMediaPipeMaskHalf($("maskR"), "R");
}

function drawMediaPipeMaskHalf(canvas, side) {
  const source = latestMediaPipeMask;
  const sourceWidth = source.width || $("segInput").width || SEGMENTATION_MAX_WIDTH;
  const sourceHeight = source.height || $("segInput").height || Math.round(SEGMENTATION_MAX_WIDTH * 0.75);
  const halfWidth = Math.floor(sourceWidth / 2);
  if (!halfWidth) return;
  if (canvas.width !== halfWidth || canvas.height !== sourceHeight) {
    canvas.width = halfWidth;
    canvas.height = sourceHeight;
  }

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, halfWidth, sourceHeight);
  ctx.setTransform(-1, 0, 0, 1, halfWidth, 0);
  const sx = side === "L" ? halfWidth : 0;
  ctx.drawImage(source, sx, 0, halfWidth, sourceHeight, 0, 0, halfWidth, sourceHeight);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  thresholdMask(canvas);
}

function thresholdMask(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < data.data.length; i += 4) {
    const keep = data.data[i] >= 128 ? 255 : 0;
    data.data[i] = keep;
    data.data[i + 1] = keep;
    data.data[i + 2] = keep;
    data.data[i + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
}

async function segHalfYolo(camCanvas, maskCanvas) {
  if (!camCanvas.width) return;
  try {
    const url = await window.pywebview.api.yolo_segment(camCanvas.toDataURL("image/jpeg", 0.85));
    await new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        if (liveActive) {
          if (maskCanvas.width !== img.width || maskCanvas.height !== img.height) {
            maskCanvas.width = img.width;
            maskCanvas.height = img.height;
          }
          const ctx = maskCanvas.getContext("2d");
          ctx.fillStyle = "black";
          ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
          ctx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);
        }
        resolve();
      };
      img.onerror = resolve;
      img.src = url;
    });
  } catch (e) {
    pylog("seg half err " + e);
  }
}

async function segBothHalvesYolo() {
  await segHalfYolo($("camL"), $("maskL"));
  await segHalfYolo($("camR"), $("maskR"));
}

function renderHalves() {
  liveRaf = requestAnimationFrame(renderHalves);
  const video = $("preview");
  if (!stream || !video.videoWidth || !liveActive) return;

  drawCamHalf($("camL"), video, "L");
  drawCamHalf($("camR"), video, "R");

  const now = performance.now();
  if (segBackend === "mediapipe") {
    if (!segBusy && now - lastSeg > SEGMENTATION_INTERVAL_MS) {
      segmentWithMediaPipe();
    }
  } else if (!segBusy && now - lastSeg > YOLO_INTERVAL_MS) {
    segBusy = true;
    segBothHalvesYolo().finally(() => {
      segBusy = false;
      lastSeg = performance.now();
    });
  }
}

function targetObject() {
  const challenge = recordingChallenge || currentChallenge;
  return challenge ? challenge.object : (($("nounTarget").value || "object").trim());
}

function targetEmotion() {
  const challenge = recordingChallenge || currentChallenge;
  return challenge ? challenge.emotion : ($("emoTarget").value || "emotion");
}

function bestEntry(best) {
  const entries = [
    { side: "L", player: "Player 1", data: best.L },
    { side: "R", player: "Player 2", data: best.R },
  ].filter((entry) => entry.data);
  if (!entries.length) return null;
  return entries.reduce((winner, entry) => (
    scoreFromEntry(entry.data).total > scoreFromEntry(winner.data).total ? entry : winner
  ));
}

function setRecapImage(id, dataUrl) {
  const image = $(id);
  if (dataUrl) {
    image.src = dataUrl;
    image.classList.remove("empty");
  } else {
    image.removeAttribute("src");
    image.classList.add("empty");
  }
}

function populateTeamRecap(best) {
  const winner = bestEntry(best);
  ["L", "R"].forEach((side) => {
    const entry = best[side];
    const score = scoreFromEntry(entry);
    const rawClassPct = entry && typeof entry.cls === "number" ? entry.cls * 100 : null;
    const isWinner = winner && winner.side === side;
    $("teamCard" + side).classList.toggle("winner", !!isWinner);
    $("teamBadge" + side).textContent = isWinner ? "Winner" : "";
    setRecapImage("finalImg" + side, entry && entry.dataUrl);
    setRecapImage("finalMask" + side, entry && entry.maskUrl);
    $("finalClass" + side).textContent = rawClassPct === null ? "--" : rawClassPct.toFixed(1) + "%";
    $("finalClass" + side).title = entry ? "Normalized for total: " + score.classPct.toFixed(0) + "%" : "";
    $("finalEmo" + side).textContent = entry ? score.emoPct.toFixed(1) + "%" : "--";
    $("finalTotal" + side).textContent = entry ? score.total.toFixed(0) : "--";
  });
}

function showEndScreen(best) {
  const winner = bestEntry(best);
  const score = winner ? scoreFromEntry(winner.data).total : 0;
  const challenge = best.challenge || recordingChallenge || currentChallenge || {
    emotion: targetEmotion(),
    object: targetObject(),
  };
  const emotion = challenge.emotion;
  const object = challenge.object;
  const message = "You got " + emotion + " " + object + " Maxed mogged.";
  const runId = resetFluxPanel();

  $("finalScore").textContent = score.toFixed(1);
  $("finalMessage").textContent = message;
  $("finalBreakdown").textContent = winner
    ? winner.player + " won. Player 1 " + (best.L ? scoreFromEntry(best.L).total.toFixed(1) : "--") +
      " / Player 2 " + (best.R ? scoreFromEntry(best.R).total.toFixed(1) : "--")
    : "No score recorded.";
  populateTeamRecap(best);

  show("end");
  playFinishSound();
  generateFluxPeople(best, emotion, object, runId);
  if (!endPopupPending) {
    endPopupPending = true;
    setTimeout(() => {
      window.alert((winner ? winner.player + " won.\n" : "") + message + "\nFinal score: " + score.toFixed(1));
      endPopupPending = false;
    }, 60);
  }
}

function resetFluxPanel() {
  fluxRunId += 1;
  $("fluxStatus").textContent = "Preparing BFL generation...";
  ["L", "R"].forEach((side) => {
    $("fluxCard" + side).classList.add("hidden");
    $("fluxImg" + side).removeAttribute("src");
    $("fluxCaption" + side).textContent = side === "L" ? "Player 1" : "Player 2";
  });
  return fluxRunId;
}

function fluxPromptForSide(emotion, object, isWinner) {
  const base = (
    "replace the masked object with a " + emotion + " " + object + ". " +
    "Use the original human in the input photo as the base. " +
    "The face is unmasked and must remain the original face, original identity, and original expression. " +
    "Do not invent a different person. Keep the face untouched. "
  );
  if (isWinner) {
    return base +
      "Winner version: giga chad, sigma, massive aura, aura farm, really cool kid, legendary confidence, " +
      "peak charisma, triumphant body language, premium drip, cinematic victory energy. " +
      "Make the object and emotion read clearly while preserving the original face.";
  }
  return base +
    "Loser version of the same concept: low aura, washed, goofy defeated energy, unlucky, awkward, " +
    "small sad aura, rumpled loser drip, rejected side character, comedic flop energy. " +
    "Make the object and emotion read clearly while preserving the original face.";
}

function fluxPrompts(best, emotion, object) {
  const winner = bestEntry(best);
  return {
    L: fluxPromptForSide(emotion, object, winner && winner.side === "L"),
    R: fluxPromptForSide(emotion, object, winner && winner.side === "R"),
  };
}

async function generateFluxPeople(best, emotion, object, runId) {
  const left = best.L || {};
  const right = best.R || {};
  if (!left.dataUrl && !right.dataUrl) {
    $("fluxStatus").textContent = "No final masks available for BFL.";
    return;
  }
  if (!window.pywebview || !window.pywebview.api || !window.pywebview.api.generate_flux_people) {
    $("fluxStatus").textContent = "BFL backend is not ready.";
    return;
  }

  $("fluxStatus").textContent = "Generating two BFL images...";
  try {
    const prompts = fluxPrompts(best, emotion, object);
    const result = await window.pywebview.api.generate_flux_people(
      left.dataUrl || null,
      left.maskUrl || null,
      right.dataUrl || null,
      right.maskUrl || null,
      null,
      prompts.L,
      prompts.R,
    );
    if (runId !== fluxRunId) return;

    if (!result || !result.ok) {
      $("fluxStatus").textContent = "BFL skipped: " + ((result && result.error) || "unknown error");
      return;
    }

    (result.items || []).forEach((item) => {
      const side = item.side;
      $("fluxImg" + side).src = item.url;
      $("fluxCaption" + side).textContent = item.label + " BFL";
      $("fluxCard" + side).classList.remove("hidden");
      pylog("flux " + side + " saved " + item.path);
    });

    const count = (result.items || []).length;
    $("fluxStatus").textContent = count + " BFL image" + (count === 1 ? "" : "s") + " generated.";
    if (result.error) $("fluxStatus").textContent += " Some generations failed: " + result.error;
  } catch (e) {
    if (runId === fluxRunId) {
      $("fluxStatus").textContent = "BFL failed: " + e;
    }
    pylog("flux error " + e);
  }
}

function playAgain() {
  stopCamera();
  startCamera();
}

window.addEventListener("DOMContentLoaded", () => {
  $("startBtn").addEventListener("click", startCamera);
  $("rerollTargetBtn").addEventListener("click", () => {
    if (gameEffectsActive) return;
    ensureAudio();
    randomizeChallenge({ animate: true });
  });
  $("nounTarget").addEventListener("input", () => updateChallengeFromInputs(false));
  $("nounTarget").addEventListener("blur", () => updateChallengeFromInputs(true));
  $("emoTarget").addEventListener("change", () => updateChallengeFromInputs(true));
  $("shootBtn").addEventListener("click", shoot);
  $("uploadBtn").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", onUpload);
  $("retakeBtn").addEventListener("click", retake);
  $("backBtn").addEventListener("click", back);
  $("endAgainBtn").addEventListener("click", playAgain);
  $("endHomeBtn").addEventListener("click", back);

  pylog("ui ready");
});

window.addEventListener("score-update", onScoreUpdate);
