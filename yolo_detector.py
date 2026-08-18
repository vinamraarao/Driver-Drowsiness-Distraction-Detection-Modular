"""
yolo_detector.py
-----------------
Loads the YOLO model once and runs scaled-down inference, returning
person boxes (COCO class 0) and cell-phone boxes (COCO class 67).
"""

import cv2
from ultralytics import YOLO

import config


def load_yolo_model():
    print("[INFO] Loading YOLO...")
    try:
        return YOLO(config.YOLO_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] YOLO load failed: {e}")
        raise


def run_yolo_scaled(yolo_model, frame, scale, conf_threshold):
    if scale < 0.999:
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        inv_scale = 1.0 / scale
    else:
        small = frame
        inv_scale = 1.0
    try:
        result = yolo_model(small, verbose=False)[0]
    except Exception:
        return [], []
    person_boxes = []
    phone_boxes = []
    if result.boxes is not None:
        for item in result.boxes:
            conf = float(item.conf)
            if conf < conf_threshold:
                continue
            label = int(item.cls)
            coords = item.xyxy.flatten().tolist()[:4]
            coords = [c * inv_scale for c in coords]
            if label == 0:
                person_boxes.append(coords)
            elif label == 67:
                phone_boxes.append(coords)
    return person_boxes, phone_boxes


def confidence_threshold_for_mode(night_active):
    """At night the enhanced frame is noisier and objects are lower-contrast,
    so we accept slightly lower-confidence detections rather than lose the
    driver/phone box entirely."""
    return (
        config.YOLO_CONFIDENCE_THRESHOLD_NIGHT
        if night_active
        else config.YOLO_CONFIDENCE_THRESHOLD
    )
