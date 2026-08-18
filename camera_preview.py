"""
Simple camera preview and diagnostics.
Run this to verify the webcam works and to see live frame luma/size output.
Usage:
  python camera_preview.py --index 0 --backend dshow

Press 'q' to quit.
"""

import argparse
import time
import platform
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--backend", type=str, default=None, choices=['dshow','msmf','default',None])
args = parser.parse_args()

backend_map = {
    'dshow': cv2.CAP_DSHOW,
    'msmf': cv2.CAP_MSMF,
    'default': None,
    None: None,
}
backend = backend_map.get(args.backend, None)

print(f"Camera preview: index={args.index}, backend={args.backend}")

# ensure no leftover windows
try:
    cv2.destroyAllWindows()
except Exception:
    pass

if backend is None:
    cap = cv2.VideoCapture(args.index)
else:
    cap = cv2.VideoCapture(args.index, backend)

if not cap.isOpened():
    print("[ERROR] Could not open camera")
    cap.release()
    raise SystemExit(1)

print("[INFO] Camera opened. Press 'q' in the preview window to quit.")

frame_count = 0
start = time.time()

while True:
    ret, frame = cap.read()
    frame_count += 1
    if not ret or frame is None or getattr(frame, 'size', 0) == 0:
        print(f"[WARN] Bad frame #{frame_count}: ret={ret}, size={0 if frame is None else frame.size}")
        # show a placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow('Camera Preview', placeholder)
    else:
        # compute mean luma of the downsampled frame for quick brightness metric
        small = cv2.resize(frame, (160, 120))
        luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()
        text = f"Frame #{frame_count} - {frame.shape[1]}x{frame.shape[0]} - luma={luma:.1f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Camera Preview', frame)

    # print status occasionally
    if frame_count % 30 == 0:
        elapsed = time.time() - start
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"[INFO] {frame_count} frames, {fps:.1f} fps, last_luma={luma if 'luma' in locals() else 'N/A'}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Preview ended")
