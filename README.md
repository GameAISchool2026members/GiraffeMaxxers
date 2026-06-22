# Game Jam Charades

Desktop pywebview app for a two-player CLIP/emotion charades round.

## Run

Install requirements.txt to an environment of your choice, then from the main directory run  ```camera_ui_main.py```



## What It Does

- Opens a webcam at 1280×720.
- Uses MediaPipe Selfie Segmentation for live masks.
- Runs MediaPipe on a hidden downsampled canvas capped at 320px wide.
- Randomly picks an object and emotion challenge each round.
- Scoring starts after a 3-second countdown and runs for 15 seconds.
- Scores each player from 0-100 with browser-side CLIP plus face-api emotion
  detection. Raw CLIP class confidence is remapped from 0-40% into 0-100%
  before combining class x2 plus expression x1.
- After the round, shows a final result screen with each player's original best
  frame, shape score, emotion score, and a winner badge.
- If `BFL_API_KEY` is set before launch, generates one BFL inpainted image
  for each player's final best frame using that player's best mask. The prompt
  is `A <emotion> <object>.`

The legacy UI (`main.py` / `ui/index.html`) retains: threshold jingles and star
effects as players cross each 10% score mark; a final-score popup
(`You got <emotion> <object> Maxed mogged.`); a full recap with masks, raw CLIP
class confidence, and normalized totals; winner vs. loser BFL prompt variants with
giga-chad/low-aura keywords; and a YOLO segmentation fallback in the start-screen
selector.

## Team

Panagiotis Tsakalakis, Eric Wang, James van der Pol, Tim Merino, Maria Edwards, and dical_gwt
