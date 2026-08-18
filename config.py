"""
config.py
---------
All tunable constants for the driver monitoring system.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(BASE_DIR / "face_landmarker.task")
YOLO_MODEL_PATH = str(BASE_DIR / "yolov8n.pt")
LOG_FILE = str(BASE_DIR / "event_history_log.csv")

# Path to optional MediaPipe hand landmarker task file (place next to this script)
HAND_MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")

# =====================================================================
# ALERT THRESHOLDS (frame counters)
# =====================================================================
DROWSY_FRAME_LIMIT = 20
SLEEP_FRAME_LIMIT = 45
YAWN_FRAME_LIMIT = 20
DISTRACTION_FRAME_LIMIT = 30

# =====================================================================
# EYE / MOUTH GEOMETRY
# =====================================================================
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.55

# =====================================================================
# CAMERA
# =====================================================================
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
MAX_CAMERA_RETRIES = 5

# =====================================================================
# YOLO
# =====================================================================
YOLO_INFER_SCALE = 0.4
YOLO_SKIP_FRAMES = 1                 # run every 2nd frame
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_CONFIDENCE_THRESHOLD_NIGHT = 0.35

# =====================================================================
# DRIVER SIDE
# =====================================================================
DRIVER_SIDE = 'right'
FLIP_FRAME = True

# =====================================================================
# PHONE FALSE-POSITIVE REDUCTION
# =====================================================================
PHONE_CONTAINMENT_THRESHOLD = 0.3
# How many consecutive YOLO confirmations required before treating a
# detected phone as belonging to the driver and raising a violation.
PHONE_CONFIRMATION_LIMIT = 3
# If the YOLO worker hasn't produced a new result within this many seconds,
# treat detections as stale and reset confirmation counters.
PHONE_STALE_TIMEOUT = 1.0

# =====================================================================
# HAND CONTROL RULES
# =====================================================================
ONE_HAND_MAX_SECONDS = 60.0
TWO_HAND_MAX_SECONDS = 5.0

# =====================================================================
# HUD  -  compact horizontal top bar
# =====================================================================
HUD_WIDTH = 600
HUD_HEIGHT = 55
HUD_X_OFFSET = 10
HUD_Y_OFFSET = 10

# =====================================================================
# LOGGING
# =====================================================================

# =====================================================================
# AUDIO
# =====================================================================
BEEP_COOLDOWN_SEC = 1.5

# =====================================================================
# MEDIAPIPE MODEL (auto-download)
# =====================================================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# =====================================================================
# STANDARD LANDMARK INDICES
# =====================================================================
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_TOP_IDX = [82, 13, 312]
MOUTH_BOTTOM_IDX = [87, 14, 317]
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308
NOSE_TIP_IDX = 1

# =====================================================================
# NIGHT VISION / LOW-LIGHT ENHANCEMENT
# =====================================================================
ENABLE_NIGHT_VISION = True

NIGHT_MODE_ENTER_LUMA = 70
NIGHT_MODE_EXIT_LUMA = 85
BRIGHTNESS_SAMPLE_DOWNSCALE = 0.25

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

GAMMA_NIGHT = 1.8

DENOISE_H_NIGHT = 7
SHARPEN_NIGHT = True

NIGHT_VISION_DISPLAY_TINT = True   # green-phosphor cosmetic tint on screen at night

# Hardware gain/exposure knobs are OFF by default. Many Windows UVC webcams
# (esp. with CAP_DSHOW) mishandle a forced manual-exposure request and get
# DARKER instead of brighter, which traps NightModeState in night mode
# forever (frame never climbs back above NIGHT_MODE_EXIT_LUMA) -> permanent
# green tint. The software pipeline (CLAHE + gamma in night_vision.py) does
# the real work and doesn't have this failure mode. Set these back to a
# number only if you've confirmed your specific camera driver honors them.
NIGHT_CAM_GAIN = None
NIGHT_CAM_EXPOSURE = None
DAY_CAM_GAIN = None
DAY_CAM_EXPOSURE = None
