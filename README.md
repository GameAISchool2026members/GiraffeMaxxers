# Game Jam Charades

Desktop pywebview app for a two-player CLIP/emotion charades round.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe actual_game\jam\main.py
```

Or double-click/run:

```powershell
actual_game\jam\run.bat
```

## What It Does

- Opens a webcam at 640x480.
- Uses MediaPipe Selfie Segmentation by default for live masks.
- Runs MediaPipe on a hidden downsampled canvas capped at 256px wide.
- Randomly picks an object and expression challenge each round.
- Starts the challenge reveal, score effects, and sounds only when recording begins.
- Scores each player from 0-100 with browser-side CLIP plus face-api emotion
  detection. Raw CLIP class confidence is remapped from 0-40% into 0-100%
  before combining class x2 plus expression x1.
- Plays threshold jingles and star effects as players cross each 10% score mark.
- Defaults to a 10 second recording window.
- After capture, shows a final score screen and popup:
  `You got <emotion> <object> Maxed mogged.`
- The final recap shows each player's original best frame, mask, raw CLIP class
  confidence, expression confidence, normalized total, and winner.
- If `BFL_API_KEY` is set before launch, generates one BFL inpainted image
  for each team's final best frame using that team's final best mask. The prompt
  uses the original script style: `replace the masked object with a <emotion> <object>`.
- BFL uses a winner prompt with giga chad/sigma/aura keywords and a loser prompt
  with low-aura loser keywords.

YOLO remains available as an optional fallback in the start-screen selector.
