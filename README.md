# Driver Drowsiness and Distraction Detection System — Modular Build

Your original single-file `app.py` has been split into focused modules and
gained a new **night vision** capability, so detection keeps working after
dark instead of just going blind.

## ▶ How to run

**`main.py` is the file to run** (it replaces `app.py` as the entry point):

```bash
pip install -r requirements.txt
python main.py
```

`yolov8n.pt` and `face_landmarker.task` are already included in this folder
(carried over from your repo), so the first run won't need to download
anything. Press **Q** to quit, same as before.

## File layout

| File | Responsibility |
|---|---|
| `config.py` | All tunable constants (thresholds, camera, HUD, **night vision settings**) |
| `geometry.py` | Distance/EAR/MAR math, bounding-box IoU & containment |
| `night_vision.py` | **New.** Brightness detection, low-light image enhancement, camera gain/exposure control, cosmetic green-tint display |
| `audio_alert.py` | Non-blocking cross-platform beep |
| `event_logger.py` | CSV event logging |
| `camera_utils.py` | Camera open/reconnect + day/night hardware switch |
| `yolo_detector.py` | YOLO model load + scaled inference (person/phone boxes) |
| `face_analyzer.py` | MediaPipe FaceLandmarker setup + per-frame EAR/MAR/head-pose extraction |
| `hud.py` | On-screen metrics panel, alert banner, fullscreen letterboxing |
| `main.py` | Orchestrates everything — **run this file** |

## Two bugs fixed while modularizing

Your uploaded `app.py` had two issues that were carried forward into the
new modules as fixes (not new behavior, just corrected):

1. **Phone detection almost never fired.** It used `box_iou()` (intersection
   over *union*) between the phone box and driver box. A phone box is tiny
   next to a person box, so the union is dominated by the person box and
   IoU stays near-zero even when the phone sits fully inside the driver's
   region. `geometry.py` now also has `box_containment_ratio()`
   (intersection ÷ phone-box area), and `main.py` uses that for the phone
   check — it correctly asks "how much of the phone is inside the driver
   box" regardless of the size mismatch.
2. **Driver-side-after-flip logic was subtle/easy to get backwards.** It's
   now a single, commented calculation in `main.py`'s
   `resolve_driver_side_in_image()` that matches how the camera actually
   sees the cabin (facing the occupants, so left/right is naturally
   mirrored versus their physical seat — flipping the frame cancels that
   mirroring). If the driver box lands on the wrong side for your camera
   mount, flip `DRIVER_SIDE` in `config.py`.

## How night vision works

A normal RGB webcam isn't a true infrared sensor, but in realistic night
driving conditions (streetlights, dashboard glow, dim cabin lighting) it
still picks up a faint, noisy image rather than pure black. The new
pipeline in `night_vision.py` makes that faint signal usable by the exact
same YOLO + MediaPipe models you already had, instead of running a
separate detection path:

1. **Brightness sensing** — every frame, `frame_mean_luma()` measures
   average luminance on a cheap downsampled copy.
2. **Hysteresis switch** — `NightModeState` flips into night mode below
   `NIGHT_MODE_ENTER_LUMA` and back to day mode above
   `NIGHT_MODE_EXIT_LUMA` (two different thresholds), so the system
   doesn't flicker between modes when brightness hovers near the edge,
   e.g. driving under intermittent streetlights.
3. **Image enhancement** — in night mode, `enhance_for_low_light()`:
   - Converts to LAB and applies **CLAHE** (adaptive local contrast) to
     the luminance channel only, so color cues (skin tone, phone color)
     aren't distorted.
   - Applies a **gamma lift** to brighten shadows without blowing out any
     remaining highlights.
   - **Denoises** (sensor noise dominates in the dark).
   - **Unsharp-masks** afterward to restore edge detail that denoising
     softens — edges are what MediaPipe's landmarker and YOLO rely on.
4. **Same detection, better input** — YOLO and the face landmarker run on
   this enhanced frame at night (with a slightly lower YOLO confidence
   threshold, since low-light detections score lower even when correct).
   Every alert — sleep, drowsiness, yawning, distraction, phone usage —
   uses the identical logic as during the day.
5. **Camera hardware nudge** — `camera_utils.sync_camera_to_light_mode()`
   best-effort raises the camera's gain and opens up exposure when night
   mode starts, and restores auto-exposure when it ends. Wrapped in
   try/except everywhere since many USB webcams ignore some of these
   properties — the software enhancement pipeline is the real workhorse,
   the hardware nudge is a bonus when supported.
6. **Display** — what's shown on screen is optionally tinted in a classic
   green-phosphor "night vision" look (`NIGHT_VISION_DISPLAY_TINT` in
   `config.py`) for driver familiarity; this tint is display-only and
   never touches the frame fed into the detectors. The HUD also shows a
   live "Light: NN luma" reading and a "NIGHT VISION ACTIVE" badge.

### Tuning it

All in `config.py`:

- `ENABLE_NIGHT_VISION` — master on/off switch.
- `NIGHT_MODE_ENTER_LUMA` / `NIGHT_MODE_EXIT_LUMA` — sensitivity of the
  day/night switch.
- `CLAHE_CLIP_LIMIT`, `GAMMA_NIGHT`, `DENOISE_H_NIGHT` — enhancement
  strength; raise `GAMMA_NIGHT` for darker cabins, raise
  `DENOISE_H_NIGHT` if the enhanced frame looks too grainy (at some cost
  to fine edge detail).
- `NIGHT_CAM_GAIN` / `NIGHT_CAM_EXPOSURE` — hardware knobs, if your camera
  driver honors them.

## What stayed the same

Driver-side determination, EAR/MAR thresholds, alert priority order
(sleep > phone > drowsy > yawn > distraction), CSV logging, camera
auto-reconnect, and the fullscreen HUD are unchanged in behavior — just
relocated into the appropriate module (with the two bug fixes above).
