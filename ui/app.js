const $ = (id) => document.getElementById(id);

// Report a status line back to Python (shows in stdout) — handy for debugging.
function pylog(msg) {
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.log) {
      window.pywebview.api.log(msg);
    }
  } catch (e) { /* api not ready yet */ }
}

let stream = null;
let liveActive = false;     // only drive the live mask + scoring while the camera is live
let segBusy = false;
let liveRaf = null;
let lastYolo = 0;           // throttle YOLO calls (CPU)

function show(viewId) {
  document.querySelectorAll('.view').forEach((v) => v.classList.add('hidden'));
  $(viewId).classList.remove('hidden');
}

async function startCamera() {
  show('camera');
  resetToLive();
  try { const dir = await window.pywebview.api.new_session(); pylog('session ' + dir); }
  catch (e) { pylog('new_session error ' + e); }
  $('camStatus').textContent = 'Opening camera… (click Allow if prompted)';
  pylog('requesting camera (new session)');
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    $('preview').srcObject = stream;
    $('camStatus').textContent = 'Camera ready';
    pylog('camera started');
    liveActive = true;
    if (!liveRaf) renderHalves();
    if (window.startScoring) window.startScoring(); // per-half (2-player) scoring
  } catch (e) {
    $('camStatus').textContent = 'Cannot open camera: ' + e.name + ' — ' + e.message;
    pylog('camera ERROR ' + e.name + ': ' + e.message);
  }
}

// Capture: record for N seconds and keep each player's best-Total frame (N=0 → one snapshot).
async function shoot() {
  const secs = Math.max(0, parseInt(($('delay') || {}).value, 10) || 0);
  if (!window.recordBest2) return;
  $('shootBtn').disabled = true;
  $('camStatus').textContent = secs > 0
    ? 'Recording ' + secs + "s — keeping each player's best frame…"
    : "Capturing each player's frame…";
  const best = await window.recordBest2(secs);
  $('shootBtn').disabled = false;
  if (best) finishRecord(best);
}

function setScore(side, cls, emo) {
  $('classScore' + side).textContent = (cls * 100).toFixed(1) + '%';
  $('emoScore' + side).textContent = (emo * 100).toFixed(1) + '%';
  $('totalScore' + side).textContent = (cls * 200 + emo * 100).toFixed(1);
}

function drawDataUrlToCanvas(canvas, dataUrl) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width; canvas.height = img.height;
    canvas.getContext('2d').drawImage(img, 0, 0);
  };
  img.src = dataUrl;
}

// Freeze each player's panel to their best frame; show their best score; save pic_/mask_ per side.
function finishRecord(best) {
  liveActive = false; // stop live render so the frozen best frames stay
  if (window.setScoringPaused) window.setScoringPaused(true);
  ['L', 'R'].forEach((side) => {
    const b = best[side];
    if (!b || !b.dataUrl) return;
    drawDataUrlToCanvas($('cam' + side), b.dataUrl);
    if (b.maskUrl) drawDataUrlToCanvas($('mask' + side), b.maskUrl);
    setScore(side, b.cls, b.emoScore);
    try {
      window.pywebview.api.save_image(b.dataUrl, 'pic_' + side);
      if (b.maskUrl) window.pywebview.api.save_image(b.maskUrl, 'mask_' + side);
    } catch (e) { pylog('save best err ' + e); }
  });
  $('shootBtn').classList.add('hidden');
  $('retakeBtn').classList.remove('hidden');
  const ft = (b) => (b ? b.total.toFixed(0) : '--');
  $('camStatus').textContent = 'Best frames saved · P1 ' + ft(best.L) + ' · P2 ' + ft(best.R);
  pylog('record2 best P1=' + ft(best.L) + ' P2=' + ft(best.R));
}

/* ---------- live: 2-player split (left/right halves), each YOLO-segmented ---------- */
// Draw the screen-`side` half of the mirrored camera into a half-canvas.
function drawCamHalf(canvas, v, side) {
  const w = v.videoWidth, h = v.videoHeight, hw = Math.floor(w / 2);
  if (!hw) return;
  if (canvas.width !== hw || canvas.height !== h) { canvas.width = hw; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(-1, 0, 0, 1, hw, 0);   // mirror within the half-canvas
  const sx = side === 'L' ? hw : 0;        // L = mirror of raw right half; R = mirror of raw left half
  ctx.drawImage(v, sx, 0, hw, h, 0, 0, hw, h);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

// YOLO-segment a cam-half canvas; draw the white-on-black mask into its corner thumbnail.
async function segHalf(camCanvas, maskCanvas) {
  if (!camCanvas.width) return;
  try {
    const url = await window.pywebview.api.yolo_segment(camCanvas.toDataURL('image/jpeg', 0.85));
    await new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        if (liveActive) {
          if (maskCanvas.width !== img.width || maskCanvas.height !== img.height) {
            maskCanvas.width = img.width; maskCanvas.height = img.height;
          }
          const ctx = maskCanvas.getContext('2d');
          ctx.fillStyle = 'black';
          ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
          ctx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);
        }
        resolve();
      };
      img.onerror = resolve;
      img.src = url;
    });
  } catch (e) { pylog('seg half err ' + e); }
}

async function segBothHalves() {
  await segHalf($('camL'), $('maskL'));
  await segHalf($('camR'), $('maskR'));
}

function renderHalves() {
  liveRaf = requestAnimationFrame(renderHalves);
  const v = $('preview');
  if (!stream || !v.videoWidth || !liveActive) return;
  drawCamHalf($('camL'), v, 'L');
  drawCamHalf($('camR'), v, 'R');
  if (!segBusy && performance.now() - lastYolo > 300) {  // throttle per-half YOLO (CPU)
    segBusy = true;
    segBothHalves().finally(() => { segBusy = false; lastYolo = performance.now(); });
  }
}

function resetToLive() {
  liveActive = true; // camera is live again
  if (window.setScoringPaused) window.setScoringPaused(false);
  $('retakeBtn').classList.add('hidden');
  $('shootBtn').classList.remove('hidden');
}

function retake() {
  resetToLive();
  $('camStatus').textContent = stream ? 'Camera ready' : 'No camera';
}

function back() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  $('preview').srcObject = null;
  resetToLive();
  liveActive = false; // stop the live mask on home
  if (window.stopScoring) window.stopScoring();
  show('home');
}

window.addEventListener('DOMContentLoaded', () => {
  $('startBtn').addEventListener('click', startCamera);
  $('shootBtn').addEventListener('click', shoot);
  $('retakeBtn').addEventListener('click', retake);
  $('backBtn').addEventListener('click', back);
  pylog('ui ready');
});
