"""
camera_utils.py
----------------
Reliable webcam handling for Windows/Linux.
 
The program tries multiple camera backends and camera indices.
It validates that actual frames are being received before accepting
a camera device.
"""
 
import platform
import time
 
import cv2
 
import config
from night_vision import (
    apply_night_camera_settings,
    apply_day_camera_settings,
)
 
 
import threading
 
 
class CameraStream:
    """Background camera reader that keeps the latest frame in memory.
 
    Provides a simple drop-in replacement for cv2.VideoCapture with methods:
    - read() -> (ret, frame)
    - release()
    - isOpened()
    - set(prop, val)
    - get(prop)
 
    The background thread continuously reads frames so the consumer can
    always fetch the most recent frame even when the main thread is blocked.
    """
 
    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._stopped = False
        self._ret = False
        self._frame = None
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
 
    def _reader(self):
        while not self._stopped:
            try:
                if not self._cap.isOpened():
                    time.sleep(0.05)
                    continue
                ret, frame = self._cap.read()
                with self._lock:
                    self._ret = ret
                    self._frame = frame
                # small sleep so this thread doesn't spin too hard
                time.sleep(0.001)
            except Exception:
                # keep attempting until released
                time.sleep(0.05)
 
    def read(self):
        with self._lock:
            return (self._ret, None if self._frame is None else self._frame.copy())
 
    def release(self):
        self._stopped = True
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass
 
    def isOpened(self):
        try:
            return self._cap.isOpened()
        except Exception:
            return False
 
    def set(self, prop, val):
        try:
            return self._cap.set(prop, val)
        except Exception:
            return False
 
    def get(self, prop):
        try:
            return self._cap.get(prop)
        except Exception:
            return 0
 
 
def _try_open_camera(index, backend):
    """Try opening one camera index with one OpenCV backend."""
 
    try:
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
 
        if not cap.isOpened():
            cap.release()
            return None
 
        # Give Windows/UVC camera time to initialize.
        time.sleep(0.3)
 
        def _count_valid_frames(attempts, delay):
            valid = 0
            for _ in range(attempts):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    valid += 1
                time.sleep(delay)
            return valid
 
        # STEP 1: confirm the camera actually delivers frames at whatever
        # NATIVE mode it opened in, before touching any properties. This is
        # exactly what a plain `cv2.VideoCapture(index).read()` does, so if
        # this fails the camera itself is the problem, not our settings.
        native_valid = _count_valid_frames(attempts=8, delay=0.1)
 
        if native_valid == 0:
            print(
                f"[WARN] index={index} backend={backend}: camera reported "
                f"open but delivered 0 frames even at native settings. "
                f"This usually means Windows camera privacy settings are "
                f"blocking access, or another app is holding the camera."
            )
            cap.release()
            return None
 
        # STEP 2: camera works natively. NOW try to request the desired
        # resolution/FPS. Some DirectShow/MSMF drivers stall or blank out
        # when forced into an unsupported width/height/fps combo, so we
        # re-validate afterward and revert if that happened.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
 
        time.sleep(0.3)
        forced_valid = _count_valid_frames(attempts=8, delay=0.1)
 
        if forced_valid == 0:
            # Forcing our resolution broke it -> reopen a fresh capture with
            # no properties forced, since a stale VideoCapture object may
            # stay wedged even after re-setting properties on some drivers.
            print(
                f"[WARN] index={index} backend={backend}: requested "
                f"{config.CAM_WIDTH}x{config.CAM_HEIGHT}@{config.CAM_FPS} "
                f"broke frame delivery; reopening at the camera's native "
                f"mode instead."
            )
            cap.release()
            time.sleep(0.3)
            cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                return None
            time.sleep(0.3)
            valid_frames = _count_valid_frames(attempts=8, delay=0.1)
        else:
            valid_frames = forced_valid
 
        if valid_frames >= 3:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
            print(
                f"[INFO] Camera opened successfully: "
                f"index={index}, backend={backend}, "
                f"resolution={width}x{height}"
            )
 
            return cap
 
        cap.release()
 
    except Exception as exc:
        print(
            f"[WARN] Camera test failed "
            f"(index={index}, backend={backend}): {exc}"
        )
 
    return None
 
 
def open_camera(preferred_index=None, preferred_backend=None):
    """
    Find and open an available webcam.
 
    preferred_index: int or None — if provided, try this index first
    preferred_backend: one of ("dshow", "msmf", "default") or None
    """
 
    system = platform.system()
 
    # Build backend list; allow a preferred_backend override
    if system == "Windows":
        default_backends = [
            ("DirectShow", cv2.CAP_DSHOW, 'dshow'),
            ("Media Foundation", cv2.CAP_MSMF, 'msmf'),
            ("Default", None, 'default'),
        ]
    else:
        default_backends = [
            ("Default", None, 'default'),
        ]
 
    # Reorder backends so preferred_backend (string) comes first when provided
    backends = []
    if preferred_backend is not None:
        key = str(preferred_backend).lower()
        for name, backend, tag in default_backends:
            if tag == key:
                backends.append((name, backend))
        for name, backend, tag in default_backends:
            if tag != key:
                backends.append((name, backend))
    else:
        backends = [(name, backend) for name, backend, _ in default_backends]
 
    # Build camera index list, trying preferred_index first if given
    camera_indices = []
    if preferred_index is not None:
        try:
            pref = int(preferred_index)
            camera_indices.append(pref)
        except Exception:
            pass
    # then fall back to 0..4 (avoid duplicates)
    for i in range(5):
        if i not in camera_indices:
            camera_indices.append(i)
 
    print("[INFO] Searching for webcam...")
 
    for backend_name, backend in backends:
        for index in camera_indices:
            print(
                f"[INFO] Testing camera index {index} "
                f"using {backend_name}..."
            )
 
            cap = _try_open_camera(index, backend)
 
            if cap is not None:
                return cap
 
    print("[ERROR] No working webcam was found.")
    print()
    print("Please check:")
    print("1. Windows Camera permission is enabled.")
    print("2. No other application is using the webcam.")
    print("3. The webcam is connected.")
    print("4. The webcam appears in Windows Camera app.")
    print()
 
    return None
 
 
def sync_camera_to_light_mode(
    cap,
    night_active,
    current_mode_is_night,
):
    """
    Change camera hardware settings only when day/night mode changes.
    """
 
    if cap is None:
        return current_mode_is_night
 
    if night_active and not current_mode_is_night:
 
        try:
            apply_night_camera_settings(cap)
        except Exception as exc:
            print(f"[WARN] Night camera settings failed: {exc}")
 
        return True
 
    if not night_active and current_mode_is_night:
 
        try:
            apply_day_camera_settings(cap)
        except Exception as exc:
            print(f"[WARN] Day camera settings failed: {exc}")
 
        return False
 
    return current_mode_is_night