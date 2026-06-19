#!/usr/bin/env python3
"""Browser-based CLIP charades prototype.

Run:
    python clip_charades_web.py

The page uses the webcam, MediaPipe Selfie Segmentation, and Transformers.js
CLIP to compare raw webcam frames vs silhouette frames for an AI-charades game.
"""

from __future__ import annotations

import argparse
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CLIP Charades Lab</title>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js"></script>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #f5f7fb;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    header {
      padding: 15px 18px;
      border-bottom: 1px solid #2a2f3a;
      background: #151923;
    }

    h1 {
      margin: 0 0 5px;
      font-size: 18px;
    }

    .subtitle {
      color: #aab2c0;
      font-size: 13px;
      line-height: 1.4;
    }

    main {
      display: grid;
      grid-template-columns: 370px 1fr;
      min-height: 0;
    }

    aside {
      padding: 16px;
      overflow: auto;
      border-right: 1px solid #2a2f3a;
      background: #121620;
    }

    .stage {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: 1fr auto;
    }

    .views {
      min-height: 0;
      overflow: auto;
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
      gap: 16px;
      align-content: start;
    }

    .card {
      min-width: 0;
      border: 1px solid #2a2f3a;
      border-radius: 12px;
      background: #151923;
      overflow: hidden;
    }

    .card h2 {
      margin: 0;
      padding: 10px 12px;
      font-size: 13px;
      border-bottom: 1px solid #2a2f3a;
      color: #dce3ee;
    }

    .wide {
      grid-column: 1 / -1;
    }

    .canvas-wrap {
      position: relative;
      background: #080a0f;
    }

    canvas, video {
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
    }

    video {
      transform: scaleX(-1);
    }

    .meter {
      padding: 12px;
    }

    .meter-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 12px;
      color: #aab2c0;
      margin-bottom: 7px;
    }

    .bar {
      width: 100%;
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      background: #232a38;
    }

    .bar > div {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, #79ffe1, #ffca85);
      transition: width 0.2s ease;
    }

    .top-labels {
      padding: 10px 12px 12px;
      color: #9ca6b7;
      font-size: 12px;
      line-height: 1.5;
      min-height: 62px;
      border-top: 1px solid #2a2f3a;
    }

    .score-board {
      display: grid;
      grid-template-columns: repeat(3, minmax(140px, 1fr));
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid #2a2f3a;
    }

    .score-tile {
      padding: 12px;
      border: 1px solid #344d45;
      border-radius: 10px;
      background: #12261f;
    }

    .score-tile strong {
      display: block;
      color: #9ca6b7;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }

    .score-tile span {
      display: block;
      color: #79ffe1;
      font-size: 24px;
      font-weight: 850;
    }

    #historyCanvas {
      width: 100%;
      height: 170px;
      aspect-ratio: auto;
      border-bottom: 1px solid #2a2f3a;
      background: #10141d;
    }

    .best-shots {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      padding: 12px;
      min-height: 92px;
    }

    .shot {
      border: 1px solid #2a2f3a;
      border-radius: 9px;
      overflow: hidden;
      background: #10141d;
    }

    .shot img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      background: #080a0f;
    }

    .shot div {
      padding: 7px 8px;
      color: #aab2c0;
      font-size: 11px;
      line-height: 1.3;
    }

    .control {
      margin-bottom: 14px;
    }

    label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
      color: #dce3ee;
      font-size: 12px;
      font-weight: 650;
    }

    input, textarea, select, button {
      width: 100%;
      border: 1px solid #343b4a;
      border-radius: 8px;
      background: #202532;
      color: #f5f7fb;
      padding: 8px 10px;
      font: inherit;
    }

    input[type="range"] {
      padding: 0;
      accent-color: #79ffe1;
    }

    textarea {
      min-height: 118px;
      resize: vertical;
      line-height: 1.35;
    }

    button {
      cursor: pointer;
      font-weight: 750;
    }

    button:hover {
      background: #293040;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .status {
      min-height: 42px;
      padding: 10px;
      border: 1px solid #2a2f3a;
      border-radius: 8px;
      background: #171c27;
      color: #aab2c0;
      font-size: 12px;
      line-height: 1.4;
      margin-bottom: 14px;
    }

    .target {
      padding: 12px;
      border: 1px solid #344d45;
      border-radius: 10px;
      background: #12261f;
      margin-bottom: 14px;
    }

    .target div:first-child {
      color: #9ca6b7;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 5px;
    }

    .target div:last-child {
      font-size: 24px;
      font-weight: 850;
      color: #79ffe1;
    }

    .hint {
      color: #9ca6b7;
      font-size: 12px;
      line-height: 1.4;
    }

    .footer {
      padding: 12px 18px;
      border-top: 1px solid #2a2f3a;
      background: #151923;
      color: #9ca6b7;
      font-size: 12px;
      line-height: 1.4;
    }

    @media (max-width: 980px) {
      main {
        grid-template-columns: 1fr;
      }

      aside {
        border-right: 0;
        border-bottom: 1px solid #2a2f3a;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>CLIP Charades Lab</h1>
    <div class="subtitle">
      Pose for the webcam and try to push CLIP toward the target label. Compare raw webcam scoring against a segmented silhouette/shadow-puppet version.
    </div>
  </header>

  <main>
    <aside>
      <div class="target">
        <div>Target</div>
        <div id="targetDisplay">giraffe</div>
      </div>

      <div class="status" id="status">Click Start Webcam, then Load CLIP Model. Scores appear after Score Once or Start Scoring.</div>

      <div class="row control">
        <button id="startWebcam">Start Webcam</button>
        <button id="loadModel">Load CLIP Model</button>
      </div>

      <div class="control">
        <label for="target">Target Label</label>
        <input id="target" value="giraffe" />
      </div>

      <div class="control">
        <label for="labels">Comparison Labels</label>
        <textarea id="labels">person
giraffe
elephant
dog
cat
horse
bird
crab
snake
umbrella
tree
airplane</textarea>
        <div class="hint">CLIP score is relative to this list. Keep the target in the list. Add pose-friendly labels to make the game more interesting.</div>
      </div>

      <div class="control">
        <label for="template">Raw Prompt Template</label>
        <input id="template" value="a photo of a {label}" />
      </div>

      <div class="control">
        <label for="silhouetteTemplate">Silhouette Prompt Template</label>
        <input id="silhouetteTemplate" value="a black silhouette of a {label}" />
      </div>

      <div class="row">
        <div class="control">
          <label for="interval">Score Every <span id="intervalValue">2500ms</span></label>
          <input id="interval" type="range" min="1000" max="7000" step="250" value="2500" />
        </div>
        <div class="control">
          <label for="scoreSize">Score Size <span id="scoreSizeValue">224px</span></label>
          <input id="scoreSize" type="range" min="224" max="512" step="32" value="224" />
        </div>
      </div>

      <div class="control">
        <label for="scoreMode">Scoring Mode</label>
        <select id="scoreMode">
          <option value="alternate">Alternate raw/silhouette (fastest)</option>
          <option value="both">Raw + silhouette every pass</option>
          <option value="raw">Raw only</option>
          <option value="silhouette">Silhouette only</option>
        </select>
      </div>

      <div class="row">
        <div class="control">
          <label for="maskThreshold">Mask Threshold <span id="maskThresholdValue">55%</span></label>
          <input id="maskThreshold" type="range" min="5" max="95" value="55" />
        </div>
        <div class="control">
          <label for="maskBlur">Mask Blur <span id="maskBlurValue">4px</span></label>
          <input id="maskBlur" type="range" min="0" max="20" value="4" />
        </div>
      </div>

      <div class="control">
        <label for="silhouetteStyle">Silhouette Style</label>
        <select id="silhouetteStyle">
          <option value="black-white">Black person on white background</option>
          <option value="white-black">White person on black background</option>
          <option value="cutout">Person cutout on plain background</option>
          <option value="edge">Outline / edge only</option>
        </select>
      </div>

      <div class="row control">
        <button id="scoreOnce">Score Once</button>
        <button id="toggleLoop">Start Scoring</button>
      </div>

      <div class="control">
        <button id="resetTracking">Reset Score History</button>
      </div>

      <div class="hint">
        First model load may take a while because the browser downloads a quantized CLIP model. After that, it should cache locally.
      </div>
    </aside>

    <section class="stage">
      <div class="views">
        <div class="card">
          <h2>Raw Webcam</h2>
          <div class="canvas-wrap">
            <video id="video" autoplay playsinline muted></video>
          </div>
          <div class="meter">
            <div class="meter-row"><span>Target score</span><strong id="rawScore">--</strong></div>
            <div class="bar"><div id="rawBar"></div></div>
          </div>
          <div class="top-labels" id="rawTop">Top labels will appear here.</div>
        </div>

        <div class="card">
          <h2>Segmented Silhouette</h2>
          <div class="canvas-wrap">
            <canvas id="silhouetteCanvas"></canvas>
          </div>
          <div class="meter">
            <div class="meter-row"><span>Target score</span><strong id="silhouetteScore">--</strong></div>
            <div class="bar"><div id="silhouetteBar"></div></div>
          </div>
          <div class="top-labels" id="silhouetteTop">Top labels will appear here.</div>
        </div>

        <div class="card wide">
          <h2>Target Chase</h2>
          <div class="score-board">
            <div class="score-tile">
              <strong>Current Target Score</strong>
              <span id="targetScore">--</span>
            </div>
            <div class="score-tile">
              <strong>Best Target Score</strong>
              <span id="bestScore">--</span>
            </div>
            <div class="score-tile">
              <strong>Best Mode</strong>
              <span id="bestMode">--</span>
            </div>
          </div>
          <canvas id="historyCanvas"></canvas>
          <div class="best-shots" id="bestShots">
            <div class="hint">New best target-score frames will appear here.</div>
          </div>
        </div>
      </div>

      <div class="footer">
        Tip: this is a game probe, not a guaranteed classifier jailbreak. Watch which mode responds more to pose changes, then tune the label set and prompt template around the fun failure cases.
      </div>
    </section>
  </main>

  <canvas id="rawScoreCanvas" style="display:none;"></canvas>
  <canvas id="maskCanvas" style="display:none;"></canvas>
  <canvas id="workCanvas" style="display:none;"></canvas>

  <script type="module">
    import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

    env.allowLocalModels = false;
    env.useBrowserCache = true;

    const elements = {
      status: byId("status"),
      startWebcam: byId("startWebcam"),
      loadModel: byId("loadModel"),
      target: byId("target"),
      targetDisplay: byId("targetDisplay"),
      labels: byId("labels"),
      template: byId("template"),
      silhouetteTemplate: byId("silhouetteTemplate"),
      interval: byId("interval"),
      intervalValue: byId("intervalValue"),
      scoreSize: byId("scoreSize"),
      scoreSizeValue: byId("scoreSizeValue"),
      scoreMode: byId("scoreMode"),
      maskThreshold: byId("maskThreshold"),
      maskThresholdValue: byId("maskThresholdValue"),
      maskBlur: byId("maskBlur"),
      maskBlurValue: byId("maskBlurValue"),
      silhouetteStyle: byId("silhouetteStyle"),
      scoreOnce: byId("scoreOnce"),
      toggleLoop: byId("toggleLoop"),
      resetTracking: byId("resetTracking"),
      video: byId("video"),
      silhouetteCanvas: byId("silhouetteCanvas"),
      rawScoreCanvas: byId("rawScoreCanvas"),
      maskCanvas: byId("maskCanvas"),
      workCanvas: byId("workCanvas"),
      rawScore: byId("rawScore"),
      rawBar: byId("rawBar"),
      rawTop: byId("rawTop"),
      silhouetteScore: byId("silhouetteScore"),
      silhouetteBar: byId("silhouetteBar"),
      silhouetteTop: byId("silhouetteTop"),
      targetScore: byId("targetScore"),
      bestScore: byId("bestScore"),
      bestMode: byId("bestMode"),
      historyCanvas: byId("historyCanvas"),
      bestShots: byId("bestShots"),
    };

    const state = {
      classifier: null,
      webcamReady: false,
      segmenterReady: false,
      scoring: false,
      scoringNow: false,
      latestMask: null,
      latestResults: null,
      segmentationBusy: false,
      scoreTimer: null,
      segmentationTimer: null,
      history: [],
      bestScore: -1,
      bestMode: "",
      bestShotCount: 0,
      trackingKey: "",
      scorePassCount: 0,
      lastRawScore: null,
      lastSilhouetteScore: null,
      nextAlternateMode: "raw",
    };

    let segmenter = null;

    elements.startWebcam.addEventListener("click", startWebcam);
    elements.loadModel.addEventListener("click", loadModel);
    elements.scoreOnce.addEventListener("click", scoreBoth);
    elements.toggleLoop.addEventListener("click", toggleScoring);
    elements.resetTracking.addEventListener("click", resetTracking);

    for (const input of [
      elements.target,
      elements.labels,
      elements.template,
      elements.silhouetteTemplate,
      elements.interval,
      elements.scoreSize,
      elements.scoreMode,
      elements.maskThreshold,
      elements.maskBlur,
      elements.silhouetteStyle,
    ]) {
      input.addEventListener("input", () => {
        updateLabels();
        drawSilhouette();
      });
    }

    updateLabels();
    drawHistoryChart();
    window.addEventListener("resize", drawHistoryChart);

    function byId(id) {
      return document.getElementById(id);
    }

    function updateLabels() {
      elements.targetDisplay.textContent = normalizedTarget();
      elements.intervalValue.textContent = `${elements.interval.value}ms`;
      elements.scoreSizeValue.textContent = `${elements.scoreSize.value}px`;
      elements.maskThresholdValue.textContent = `${elements.maskThreshold.value}%`;
      elements.maskBlurValue.textContent = `${elements.maskBlur.value}px`;

      const key = trackingKey();
      if (state.trackingKey && state.trackingKey !== key) {
        resetTracking();
      }
      state.trackingKey = key;
    }

    async function startWebcam() {
      try {
        setStatus("Requesting webcam...");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false,
        });
        elements.video.srcObject = stream;
        await elements.video.play();
        state.webcamReady = true;
        resizeCanvases();
        await setupSegmentation();
        startSegmentationLoop();
        setStatus(state.classifier ? "Webcam ready. Running first score..." : "Webcam ready. Load CLIP, then score once or start scoring.");
        if (state.classifier) {
          await scoreBoth();
        }
      } catch (error) {
        setStatus(`Webcam failed: ${error.message}`);
      }
    }

    async function setupSegmentation() {
      if (segmenter) return;
      if (!window.SelfieSegmentation) {
        setStatus("MediaPipe Selfie Segmentation did not load.");
        return;
      }

      segmenter = new SelfieSegmentation({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
      });
      segmenter.setOptions({ modelSelection: 1, selfieMode: true });
      segmenter.onResults((results) => {
        state.latestResults = results;
        state.latestMask = results.segmentationMask;
        drawSilhouette();
      });
      state.segmenterReady = true;
    }

    function startSegmentationLoop() {
      if (state.segmentationTimer) return;
      state.segmentationTimer = setInterval(async () => {
        if (!segmenter || !state.webcamReady || state.segmentationBusy || state.scoringNow) return;
        state.segmentationBusy = true;
        try {
          await segmenter.send({ image: elements.video });
        } catch (error) {
          console.warn(error);
        } finally {
          state.segmentationBusy = false;
        }
      }, 250);
    }

    async function loadModel() {
      if (state.classifier) return;

      try {
        elements.loadModel.disabled = true;
        setStatus("Loading CLIP model. First load can take a minute...");
        state.classifier = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32", {
          quantized: true,
          progress_callback: (progress) => {
            if (progress.status === "progress") {
              const pct = progress.progress ? `${Math.round(progress.progress)}%` : "";
              setStatus(`Loading ${progress.file || "model"} ${pct}`);
            }
          },
        });
        setStatus(state.webcamReady ? "CLIP ready. Running first score..." : "CLIP ready. Start webcam, then score once or start scoring.");
        if (state.webcamReady) {
          await scoreBoth();
        }
      } catch (error) {
        elements.loadModel.disabled = false;
        setStatus(`CLIP load failed: ${error.message}`);
      }
    }

    function toggleScoring() {
      state.scoring = !state.scoring;
      elements.toggleLoop.textContent = state.scoring ? "Stop Scoring" : "Start Scoring";

      if (state.scoring) {
        scheduleNextScore(0);
      } else if (state.scoreTimer) {
        clearTimeout(state.scoreTimer);
        state.scoreTimer = null;
      }
    }

    function scheduleNextScore(delay = Number(elements.interval.value)) {
      if (!state.scoring) return;
      clearTimeout(state.scoreTimer);
      state.scoreTimer = setTimeout(async () => {
        await scoreBoth();
        scheduleNextScore();
      }, delay);
    }

    async function scoreBoth() {
      if (state.scoringNow) return;
      if (!state.webcamReady) {
        setStatus("Start the webcam first.");
        return;
      }
      if (!state.classifier) {
        setStatus("Load the CLIP model first.");
        return;
      }

      state.scoringNow = true;
      elements.scoreOnce.disabled = true;
      const started = performance.now();

      try {
        const rawLabels = candidatePrompts(elements.template.value);
        const silhouetteLabels = candidatePrompts(elements.silhouetteTemplate.value);
        const rawTargetPrompt = promptForLabel(normalizedTarget(), elements.template.value);
        const silhouetteTargetPrompt = promptForLabel(normalizedTarget(), elements.silhouetteTemplate.value);
        if (!rawLabels.includes(rawTargetPrompt)) rawLabels.unshift(rawTargetPrompt);
        if (!silhouetteLabels.includes(silhouetteTargetPrompt)) silhouetteLabels.unshift(silhouetteTargetPrompt);

        const modes = modesForThisPass();

        if (modes.includes("raw")) {
          const rawCanvas = drawRawScoreCanvas();
          setStatus(`Scoring raw webcam frame with CLIP... pass ${state.scorePassCount + 1}`);
          const rawResults = await classifyCanvas(rawCanvas, rawLabels);
          const rawScore = updateScore("raw", rawResults, rawTargetPrompt);
          updateTargetProgress("Raw", rawScore, rawCanvas);
        }

        if (modes.includes("silhouette")) {
          const silhouetteCanvas = drawSilhouetteScoreCanvas();
          setStatus(`Scoring segmented silhouette with CLIP... pass ${state.scorePassCount + 1}`);
          const silhouetteResults = await classifyCanvas(silhouetteCanvas, silhouetteLabels);
          const silhouetteScore = updateScore("silhouette", silhouetteResults, silhouetteTargetPrompt);
          updateTargetProgress("Silhouette", silhouetteScore, silhouetteCanvas);
        }

        state.scorePassCount += 1;

        const elapsed = Math.round(performance.now() - started);
        setStatus(`Scored ${modes.join(" + ")} in ${elapsed}ms. History points: ${state.history.length}. Target: "${normalizedTarget()}"`);
      } catch (error) {
        console.error(error);
        setStatus(`Scoring failed: ${error.message || error}`);
      } finally {
        state.scoringNow = false;
        elements.scoreOnce.disabled = false;
      }
    }

    async function classifyCanvas(canvas, labels) {
      const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
      const results = await state.classifier(dataUrl, labels);
      if (!Array.isArray(results)) {
        throw new Error(`CLIP returned unexpected output: ${JSON.stringify(results).slice(0, 180)}`);
      }
      if (results.length === 0) {
        throw new Error("CLIP returned no labels.");
      }
      return results;
    }

    function updateScore(kind, results, targetPrompt) {
      const target = findTargetResult(results, targetPrompt);
      const score = target ? target.score : 0;
      const pct = Math.round(score * 1000) / 10;
      const scoreElement = kind === "raw" ? elements.rawScore : elements.silhouetteScore;
      const barElement = kind === "raw" ? elements.rawBar : elements.silhouetteBar;
      const topElement = kind === "raw" ? elements.rawTop : elements.silhouetteTop;

      scoreElement.textContent = `${pct.toFixed(1)}%`;
      barElement.style.width = `${Math.max(1, pct)}%`;
      topElement.innerHTML = results
        .slice(0, 5)
        .map((result) => `${escapeHtml(result.label)}: ${(result.score * 100).toFixed(1)}%`)
        .join("<br>");
      return score;
    }

    function modesForThisPass() {
      const mode = elements.scoreMode.value;
      if (mode === "both") return ["raw", "silhouette"];
      if (mode === "raw") return ["raw"];
      if (mode === "silhouette") return ["silhouette"];

      const next = state.nextAlternateMode;
      state.nextAlternateMode = next === "raw" ? "silhouette" : "raw";
      return [next];
    }

    function updateTargetProgress(mode, newScore, canvas) {
      if (mode === "Raw") state.lastRawScore = newScore;
      if (mode === "Silhouette") state.lastSilhouetteScore = newScore;

      const rawScore = state.lastRawScore ?? 0;
      const silhouetteScore = state.lastSilhouetteScore ?? 0;
      const rawWins = rawScore >= silhouetteScore;
      const score = rawWins ? rawScore : silhouetteScore;
      const bestKnownMode = rawWins ? "Raw" : "Silhouette";

      state.history.push({ score, rawScore, silhouetteScore, mode, time: Date.now() });
      if (state.history.length > 120) state.history.shift();

      elements.targetScore.textContent = formatScore(score);

      if (score > state.bestScore) {
        state.bestScore = score;
        state.bestMode = bestKnownMode;
        elements.bestScore.textContent = formatScore(score);
        elements.bestMode.textContent = bestKnownMode;
        saveBestShot(canvas, score, mode);
      }

      drawHistoryChart();
    }

    function saveBestShot(canvas, score, mode) {
      if (state.bestShotCount === 0) {
        elements.bestShots.innerHTML = "";
      }
      state.bestShotCount += 1;

      const shot = document.createElement("div");
      shot.className = "shot";

      const image = document.createElement("img");
      image.src = canvas.toDataURL("image/jpeg", 0.92);
      image.alt = `${mode} best score ${formatScore(score)}`;

      const caption = document.createElement("div");
      const target = escapeHtml(normalizedTarget());
      const stamp = new Date().toLocaleTimeString();
      caption.innerHTML = `<strong>#${state.bestShotCount} ${formatScore(score)}</strong><br>${escapeHtml(mode)} · ${target}<br>${stamp}`;

      shot.append(image, caption);
      elements.bestShots.prepend(shot);

      while (elements.bestShots.children.length > 12) {
        elements.bestShots.lastElementChild.remove();
      }
    }

    function drawHistoryChart() {
      const canvas = elements.historyCanvas;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width * window.devicePixelRatio));
      const height = Math.max(170, Math.round(170 * window.devicePixelRatio));
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      const pad = 28 * window.devicePixelRatio;
      const chartWidth = width - pad * 1.5;
      const chartHeight = height - pad * 1.6;
      const left = pad;
      const top = pad * 0.55;
      const bottom = top + chartHeight;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#10141d";
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = "#283142";
      ctx.lineWidth = window.devicePixelRatio;
      ctx.fillStyle = "#8290a5";
      ctx.font = `${11 * window.devicePixelRatio}px system-ui, sans-serif`;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";

      for (let i = 0; i <= 4; i++) {
        const y = top + (chartHeight * i) / 4;
        const value = 100 - i * 25;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(left + chartWidth, y);
        ctx.stroke();
        ctx.fillText(`${value}%`, left - 7 * window.devicePixelRatio, y);
      }

      if (state.history.length < 2) {
        ctx.textAlign = "center";
        ctx.fillText("Score history will appear after a few scoring passes.", width / 2, height / 2);
        return;
      }

      drawScoreLine(ctx, state.history.map((point) => point.rawScore), "#4ea1ff", left, top, chartWidth, chartHeight);
      drawScoreLine(ctx, state.history.map((point) => point.silhouetteScore), "#ffca85", left, top, chartWidth, chartHeight);
      drawScoreLine(ctx, state.history.map((point) => point.score), "#79ffe1", left, top, chartWidth, chartHeight, 2.4);

      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillStyle = "#79ffe1";
      ctx.fillText("best-of-both", left, bottom + 20 * window.devicePixelRatio);
      ctx.fillStyle = "#4ea1ff";
      ctx.fillText("raw", left + 110 * window.devicePixelRatio, bottom + 20 * window.devicePixelRatio);
      ctx.fillStyle = "#ffca85";
      ctx.fillText("silhouette", left + 155 * window.devicePixelRatio, bottom + 20 * window.devicePixelRatio);
    }

    function drawScoreLine(ctx, values, color, left, top, width, height, lineScale = 1.6) {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineScale * window.devicePixelRatio;
      ctx.beginPath();

      values.forEach((score, index) => {
        const x = left + (values.length === 1 ? width : (width * index) / (values.length - 1));
        const y = top + height * (1 - score);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.stroke();
    }

    function resetTracking() {
      state.history = [];
      state.bestScore = -1;
      state.bestMode = "";
      state.bestShotCount = 0;
      state.scorePassCount = 0;
      state.lastRawScore = null;
      state.lastSilhouetteScore = null;
      state.nextAlternateMode = "raw";
      state.trackingKey = trackingKey();
      elements.targetScore.textContent = "--";
      elements.bestScore.textContent = "--";
      elements.bestMode.textContent = "--";
      elements.bestShots.innerHTML = '<div class="hint">New best target-score frames will appear here.</div>';
      drawHistoryChart();
    }

    function trackingKey() {
      return JSON.stringify({
        target: normalizedTarget(),
        labels: elements.labels.value,
        rawTemplate: elements.template.value,
        silhouetteTemplate: elements.silhouetteTemplate.value,
      });
    }

    function formatScore(score) {
      if (!Number.isFinite(score) || score < 0) return "--";
      return `${(score * 100).toFixed(1)}%`;
    }

    function findTargetResult(results, targetPrompt) {
      const exact = results.find((result) => result.label === targetPrompt);
      if (exact) return exact;

      const targetWords = targetPrompt.toLowerCase().split(/\s+/);
      const targetLabel = normalizedTarget().toLowerCase();
      return results.find((result) => {
        const label = String(result.label).toLowerCase();
        return label === targetLabel || label.includes(targetLabel) || targetWords.every((word) => label.includes(word));
      });
    }

    function resizeCanvases() {
      const width = elements.video.videoWidth || 640;
      const height = elements.video.videoHeight || 480;
      for (const canvas of [elements.silhouetteCanvas, elements.maskCanvas, elements.workCanvas]) {
        if (canvas.width === width && canvas.height === height) continue;
        canvas.width = width;
        canvas.height = height;
      }
    }

    function drawRawScoreCanvas() {
      const size = Number(elements.scoreSize.value);
      const canvas = elements.rawScoreCanvas;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      drawMirroredCover(ctx, elements.video, size, size);
      return canvas;
    }

    function drawSilhouetteScoreCanvas() {
      const size = Number(elements.scoreSize.value);
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(elements.silhouetteCanvas, 0, 0, size, size);
      return canvas;
    }

    function drawSilhouette() {
      if (!state.webcamReady) return;
      resizeCanvases();

      const canvas = elements.silhouetteCanvas;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      const width = canvas.width;
      const height = canvas.height;
      const threshold = Number(elements.maskThreshold.value) / 100;
      const blur = Number(elements.maskBlur.value);
      const style = elements.silhouetteStyle.value;

      ctx.save();
      ctx.clearRect(0, 0, width, height);

      if (!state.latestMask) {
        drawMirroredCover(ctx, elements.video, width, height);
        ctx.restore();
        return;
      }

      const maskCanvas = elements.maskCanvas;
      const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
      maskCtx.clearRect(0, 0, width, height);
      maskCtx.save();
      maskCtx.filter = blur > 0 ? `blur(${blur}px)` : "none";
      maskCtx.drawImage(state.latestMask, 0, 0, width, height);
      maskCtx.restore();

      const maskData = maskCtx.getImageData(0, 0, width, height);
      for (let i = 0; i < maskData.data.length; i += 4) {
        const keep = maskData.data[i] / 255 >= threshold ? 255 : 0;
        maskData.data[i] = keep;
        maskData.data[i + 1] = keep;
        maskData.data[i + 2] = keep;
        maskData.data[i + 3] = keep;
      }
      maskCtx.putImageData(maskData, 0, 0);

      if (style === "black-white" || style === "white-black") {
        const backgroundColor = style === "black-white" ? "white" : "black";
        const personColor = style === "black-white" ? "black" : "white";
        const work = elements.workCanvas;
        const workCtx = work.getContext("2d");

        ctx.fillStyle = backgroundColor;
        ctx.fillRect(0, 0, width, height);

        workCtx.clearRect(0, 0, width, height);
        workCtx.fillStyle = personColor;
        workCtx.fillRect(0, 0, width, height);
        workCtx.globalCompositeOperation = "destination-in";
        workCtx.drawImage(maskCanvas, 0, 0);
        workCtx.globalCompositeOperation = "source-over";
        ctx.drawImage(work, 0, 0);
      } else if (style === "cutout") {
        ctx.fillStyle = "#eeeeee";
        ctx.fillRect(0, 0, width, height);
        const work = elements.workCanvas;
        const workCtx = work.getContext("2d");
        workCtx.clearRect(0, 0, width, height);
        drawMirroredCover(workCtx, elements.video, width, height);
        workCtx.globalCompositeOperation = "destination-in";
        workCtx.drawImage(maskCanvas, 0, 0);
        workCtx.globalCompositeOperation = "source-over";
        ctx.drawImage(work, 0, 0);
      } else {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);
        ctx.filter = "blur(2px)";
        ctx.drawImage(maskCanvas, 0, 0);
        ctx.filter = "none";
        ctx.globalCompositeOperation = "difference";
        ctx.drawImage(maskCanvas, 2, 0);
        ctx.drawImage(maskCanvas, -2, 0);
        ctx.drawImage(maskCanvas, 0, 2);
        ctx.drawImage(maskCanvas, 0, -2);
        ctx.globalCompositeOperation = "source-over";
      }

      ctx.restore();
    }

    function drawMirroredCover(ctx, source, width, height) {
      const sourceWidth = source.videoWidth || source.naturalWidth || source.width;
      const sourceHeight = source.videoHeight || source.naturalHeight || source.height;
      const scale = Math.max(width / sourceWidth, height / sourceHeight);
      const drawWidth = sourceWidth * scale;
      const drawHeight = sourceHeight * scale;
      ctx.save();
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(source, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
      ctx.restore();
    }

    function candidatePrompts(template) {
      const labels = elements.labels.value
        .split(/\n|,/)
        .map((label) => label.trim())
        .filter(Boolean);
      return [...new Set(labels.map((label) => promptForLabel(label, template)))];
    }

    function promptForLabel(label, templateValue) {
      const template = templateValue.trim() || "a photo of a {label}";
      return template.includes("{label}") ? template.replaceAll("{label}", label) : `${template} ${label}`;
    }

    function normalizedTarget() {
      return elements.target.value.trim() || "giraffe";
    }

    function setStatus(message) {
      elements.status.textContent = message;
    }

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the CLIP charades browser prototype.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=0, help="Port to bind. Defaults to a free port.")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser.")
    return parser.parse_args()


def pick_port(host: str, requested_port: int) -> int:
    if requested_port:
        return requested_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main() -> int:
    args = parse_args()
    port = pick_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), Handler)
    url = f"http://{args.host}:{port}/"

    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    print(f"Opening {url}")
    print("Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
