"""
face_analyzer.py
-----------------
Wraps MediaPipe's FaceLandmarker (Tasks API): model download, setup, and
turning raw landmarks into the features the alert logic needs (EAR, MAR,
head pose, face bounding box).
"""

import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from geometry import (
    landmarks_to_px, eye_aspect_ratio, mouth_aspect_ratio, face_bbox_from_landmarks,
)


def ensure_model_downloaded():
    if not os.path.exists(config.MODEL_PATH):
        print("[INFO] Downloading face_landmarker.task...")
        try:
            urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
            print("[INFO] Model downloaded.")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}\nManually place {config.MODEL_PATH} next to script.")
            raise


def create_face_landmarker():
    ensure_model_downloaded()
    
    # Force CPU delegate to avoid XNNPACK crash on Python 3.14
    base_options = mp_python.BaseOptions(
        model_asset_path=config.MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU
    )
    
    landmarker_options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(landmarker_options)


class FaceFrameResult:
    __slots__ = ("face_detected", "bbox", "head_pose_x_ratio", "avg_ear", "mar")

    def __init__(self):
        self.face_detected = False
        self.bbox = None            # (x, y, w, h)
        self.head_pose_x_ratio = None
        self.avg_ear = None
        self.mar = None


def analyze_frame(face_landmarker, mp_image, timestamp_ms, w_max, h_max):
    """
    Runs the landmarker on `mp_image` (already an mp.Image, RGB) and
    extracts the raw features. Threshold/state logic (drowsy counters,
    "Looking Left" strings, etc.) stays in main.py so this stays reusable.
    """
    result = FaceFrameResult()
    try:
        mesh_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as exc:
        print(f"[ERROR] MediaPipe face analysis failed: {exc}")
        return result

    if not mesh_result.face_landmarks:
        return result

    face_landmarks = mesh_result.face_landmarks[0]
    result.face_detected = True

    pts_px = landmarks_to_px(face_landmarks, w_max, h_max)
    fx, fy, fw, fh = face_bbox_from_landmarks(pts_px, w_max, h_max)
    result.bbox = (fx, fy, fw, fh)

    nose_x = pts_px[config.NOSE_TIP_IDX][0]
    result.head_pose_x_ratio = (nose_x - fx) / fw if fw > 0 else 0.5

    left_ear = eye_aspect_ratio(pts_px, config.LEFT_EYE_EAR_IDX)
    right_ear = eye_aspect_ratio(pts_px, config.RIGHT_EYE_EAR_IDX)
    result.avg_ear = (left_ear + right_ear) / 2.0

    result.mar = mouth_aspect_ratio(pts_px)

    return result


def new_mp_image(rgb_frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)