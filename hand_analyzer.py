"""
hand_analyzer.py
----------------
Basic MediaPipe Hand Landmarker wrapper.

Provides:
- create_hand_landmarker(): create the task-landmarker (VIDEO mode, num_hands=2)
- analyze_hands(): run detect_for_video and return pixel landmarks
- draw_hands(): draw landmarks and hand connections onto an OpenCV frame

This module is intentionally minimal: it does not implement any application logic
about hands (timers, alerts, driver/passenger classification). It only detects
and draws up to 2 hands and reports the count so the main loop can display
"Hands detected: N".
"""

import os
import cv2
import time

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from geometry import box_containment_ratio, box_iou, landmarks_to_px


class HandsFrameResult:
    __slots__ = ("hands_detected", "landmarks_list")

    def __init__(self):
        self.hands_detected = 0
        self.landmarks_list = []  # list of lists of (x_px, y_px)


def ensure_model_exists():
    if not os.path.exists(config.HAND_MODEL_PATH):
        msg = (
            f"[ERROR] Missing MediaPipe hand model: {config.HAND_MODEL_PATH}.\n"
            f"Place the hand_landmarker.task file next to config.py or set config.HAND_MODEL_PATH accordingly."
        )
        print(msg)
        raise FileNotFoundError(msg)


def create_hand_landmarker():
    # Ensure the model is present; do not attempt automatic download here.
    ensure_model_exists()

    # Force CPU delegate for stability
    base_options = mp_python.BaseOptions(
        model_asset_path=config.HAND_MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return mp_vision.HandLandmarker.create_from_options(hand_options)


def analyze_hands(hand_landmarker, mp_image, timestamp_ms, w_max, h_max):
    """Run hand landmarker in VIDEO mode and return pixel landmarks.

    Returns a HandsFrameResult with hands_detected and landmarks_list where each
    hand is a list of (x_px, y_px) tuples.
    """
    result = HandsFrameResult()
    try:
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as exc:
        print(f"[ERROR] MediaPipe hand analysis failed: {exc}")
        return result

    # The tasks API returns lists like hand_result.hand_landmarks
    landmarks_container = None
    if hasattr(hand_result, 'hand_landmarks'):
        landmarks_container = hand_result.hand_landmarks
    elif hasattr(hand_result, 'hand_world_landmarks'):
        # fallback (unlikely) — prefer normalized 2D landmarks
        landmarks_container = hand_result.hand_world_landmarks

    if not landmarks_container:
        return result

    for hand_landmarks in landmarks_container:
        # convert to pixel coordinates
        try:
            pts_px = landmarks_to_px(hand_landmarks, w_max, h_max)
        except Exception:
            # hand_landmarks may not be the expected iterable; skip if problematic
            continue
        result.landmarks_list.append(pts_px)

    result.hands_detected = len(result.landmarks_list)
    return result


def hand_bbox_from_landmarks(pts_px):
    xs = [p[0] for p in pts_px]
    ys = [p[1] for p in pts_px]
    return [min(xs), min(ys), max(xs), max(ys)]


def classify_driver_hands(hand_result, driver_box, overlap_threshold=0.05):
    """Return (driver_hands, passenger_hands), each as a list of (pts, bbox)."""
    if hand_result is None or driver_box is None:
        return [], [] if hand_result is None else [(pts, hand_bbox_from_landmarks(pts)) for pts in hand_result.landmarks_list]

    driver_hands = []
    passenger_hands = []

    for pts in hand_result.landmarks_list:
        bbox = hand_bbox_from_landmarks(pts)
        iou = box_iou(bbox, driver_box)
        containment = box_containment_ratio(bbox, driver_box)
        if iou > overlap_threshold or containment > 0.15:
            driver_hands.append((pts, bbox))
        else:
            passenger_hands.append((pts, bbox))

    return driver_hands, passenger_hands


def draw_hands(frame, landmarks_list, color=(0, 255, 0), draw_box=True):
    """Draw landmarks and connections for each detected hand onto frame.

    Uses mediapipe.solutions.hands.HAND_CONNECTIONS for canonical hand edges.
    """
    if not landmarks_list:
        return

    connections = mp.solutions.hands.HAND_CONNECTIONS
    for pts in landmarks_list:
        # Draw landmarks
        for (x, y) in pts:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        # Draw connections
        for (a, b) in connections:
            if a < len(pts) and b < len(pts):
                x1, y1 = pts[a]
                x2, y2 = pts[b]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if draw_box:
            x1, y1, x2, y2 = hand_bbox_from_landmarks(pts)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
