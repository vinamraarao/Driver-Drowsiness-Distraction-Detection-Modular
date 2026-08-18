"""
geometry.py
-----------
Pure math helpers: point distances, eye/mouth aspect ratios, bounding-box
utilities. No OpenCV drawing, no I/O -> easy to unit test.
"""

import numpy as np

from config import (
    MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX,
)


def _dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def landmarks_to_px(face_landmarks, w, h):
    return [(lm.x * w, lm.y * h) for lm in face_landmarks]


def eye_aspect_ratio(pts_px, eye_idx):
    p1, p2, p3, p4, p5, p6 = [pts_px[i] for i in eye_idx]
    vertical1 = _dist(p2, p6)
    vertical2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(pts_px):
    verticals = [_dist(pts_px[t], pts_px[b]) for t, b in zip(MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX)]
    horizontal = _dist(pts_px[MOUTH_LEFT_IDX], pts_px[MOUTH_RIGHT_IDX])
    if horizontal == 0:
        return 0.0
    return sum(verticals) / (3.0 * horizontal)


def face_bbox_from_landmarks(pts_px, frame_w, frame_h, pad_ratio=0.08):
    xs = [p[0] for p in pts_px]
    ys = [p[1] for p in pts_px]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = (x_max - x_min) * pad_ratio
    pad_y = (y_max - y_min) * pad_ratio
    x_min = max(0, int(x_min - pad_x))
    y_min = max(0, int(y_min - pad_y))
    x_max = min(frame_w, int(x_max + pad_x))
    y_max = min(frame_h, int(y_max + pad_y))
    return x_min, y_min, x_max - x_min, y_max - y_min


def box_iou(box1, box2):
    """ box = [x1, y1, x2, y2] """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def box_containment_ratio(inner_box, outer_box):
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    return inter / inner_area if inner_area > 0 else 0.0


def is_driver_box(box, w_max, driver_side):
    """Return True if the box centre is on the driver's side."""
    cx = (box[0] + box[2]) / 2.0
    if driver_side == 'right':
        return cx > w_max * 0.55
    else:
        return cx < w_max * 0.45