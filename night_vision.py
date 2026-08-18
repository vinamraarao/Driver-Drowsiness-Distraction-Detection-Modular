"""
night_vision.py
----------------
Low-light detection + enhancement so the SAME face-landmark and YOLO
pipelines used during the day keep working after dark, instead of the
system going blind once ambient light drops.

Design:
1. `frame_mean_luma()` cheaply measures how dark the frame currently is
   (on a downsampled copy, so it's fast enough to run every frame).
2. `NightModeState` turns that single number into a stable day/night flag
   using hysteresis, so the system doesn't flicker in and out of night
   mode when brightness hovers near the boundary (e.g. passing under a
   streetlight).
3. `enhance_for_low_light()` is the actual image pipeline: CLAHE local
   contrast + gamma lift on the LAB luminance channel, denoise, then a
   light unsharp-mask to restore edges that denoising softens. Detection
   (YOLO + MediaPipe) is run on this enhanced frame at night, so eye/mouth
   landmarks, head pose and phone detection all keep working.
4. `apply_night_vision_tint()` is purely cosmetic (green-phosphor look)
   for what's shown on screen; it never touches what the detectors see.

Works with both a normal RGB webcam picking up scraps of light at night,
and with a true IR/NIR camera (whose frames are effectively grayscale --
CLAHE/gamma degrade gracefully there too).
"""

import cv2
import numpy as np

import config


def frame_mean_luma(frame_bgr):
    """Fast approximate brightness of a frame, 0 (black) - 255 (white)."""
    small = cv2.resize(
        frame_bgr, None,
        fx=config.BRIGHTNESS_SAMPLE_DOWNSCALE,
        fy=config.BRIGHTNESS_SAMPLE_DOWNSCALE,
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _gamma_correct(gray_u8, gamma):
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(gray_u8, table)


class NightModeState:
    """Hysteresis state machine: enter night mode below ENTER, leave above EXIT."""

    def __init__(self):
        self.active = False
        self.last_luma = 255.0

    def update(self, frame_bgr):
        self.last_luma = frame_mean_luma(frame_bgr)
        if not config.ENABLE_NIGHT_VISION:
            self.active = False
            return self.active
        if not self.active and self.last_luma < config.NIGHT_MODE_ENTER_LUMA:
            self.active = True
        elif self.active and self.last_luma > config.NIGHT_MODE_EXIT_LUMA:
            self.active = False
        return self.active


def enhance_for_low_light(frame_bgr):
    """
    Returns a brightened / contrast-enhanced BGR frame suitable for both
    YOLO and MediaPipe face-landmark detection in low light. Color
    channels (a, b) are left alone so skin tone / phone-color cues survive;
    only luminance is boosted.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    l = clahe.apply(l)
    l = _gamma_correct(l, config.GAMMA_NIGHT)

    if config.DENOISE_H_NIGHT > 0:
        l = cv2.fastNlMeansDenoising(
            l, None,
            h=config.DENOISE_H_NIGHT,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    if config.SHARPEN_NIGHT:
        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=1.2)
        l = cv2.addWeighted(l, 1.5, blur, -0.5, 0)

    lab_enhanced = cv2.merge((l, a, b))
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def apply_night_vision_tint(frame_bgr):
    """
    Cosmetic-only classic green-on-black night-vision look, for the
    on-screen display copy. Never used as the frame fed to the detectors.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    tinted = np.zeros_like(frame_bgr)
    tinted[:, :, 1] = gray  # green channel only
    return tinted


def apply_night_camera_settings(cap):
    """
    Best-effort attempt to raise gain / open exposure on the physical
    camera at night. Many UVC webcams ignore some/all of these; every
    call is wrapped so a rejected property never crashes the app.
    """
    if config.NIGHT_CAM_GAIN is not None:
        try:
            cap.set(cv2.CAP_PROP_GAIN, config.NIGHT_CAM_GAIN)
        except Exception:
            pass
    if config.NIGHT_CAM_EXPOSURE is not None:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode on many drivers
            cap.set(cv2.CAP_PROP_EXPOSURE, config.NIGHT_CAM_EXPOSURE)
        except Exception:
            pass


def apply_day_camera_settings(cap):
    """Restore auto exposure / default gain for daytime."""
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto mode on many drivers
    except Exception:
        pass
    if config.DAY_CAM_GAIN is not None:
        try:
            cap.set(cv2.CAP_PROP_GAIN, config.DAY_CAM_GAIN)
        except Exception:
            pass
