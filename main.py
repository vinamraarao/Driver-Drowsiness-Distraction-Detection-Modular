"""
main.py
-------
Module 8: Unified Dashboard System - entry point.
Run with: python main.py
"""

import os
# Fix OpenBLAS memory crash on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import cv2
import numpy as np

import config
from audio_alert import alert_beep
from camera_utils import open_camera, sync_camera_to_light_mode
from event_logger import EventLogger
from geometry import box_containment_ratio, is_driver_box
from hud import draw_hud_panel, draw_alert_banner, draw_zone_gridlines, letterbox_to_fullscreen, get_screen_size
from night_vision import (
    NightModeState, enhance_for_low_light, apply_night_vision_tint,
)

WINDOW_NAME = f"Module 8: Unified Dashboard System - PID {os.getpid()}"


def resolve_driver_side_in_image():
    if config.FLIP_FRAME:
        return config.DRIVER_SIDE
    return 'left' if config.DRIVER_SIDE == 'right' else 'right'


def main():
    # Parse CLI args early so camera selection can be forced for debugging
    import argparse

    parser = argparse.ArgumentParser(description="Driver monitoring main")
    parser.add_argument("--camera-index", type=int, default=None, help="Force camera index (e.g., 0 or 1)")
    parser.add_argument("--backend", type=str, default=None, choices=["dshow", "msmf", "default"], help="Force video backend on Windows")
    parser.add_argument("--no-fullscreen", action="store_true", help="Do not create a fullscreen window (for debugging)")
    parser.add_argument("--skip-models", action="store_true", help="Run without loading YOLO/MediaPipe (debug camera only)")
    args = parser.parse_args()

    # Set sensible defaults on Windows so common webcams are tried first
    import platform
    if platform.system() == 'Windows':
        if args.camera_index is None:
            args.camera_index = 1
        if args.backend is None:
            args.backend = 'dshow'

    print(f"[INFO] Using camera_index={args.camera_index}, backend={args.backend}")

    # Open camera early to fail fast and surface permission/connection errors
    cap = open_camera(preferred_index=args.camera_index, preferred_backend=args.backend)
    if cap is None:
        print("[ERROR] Could not open camera after retries.")
        return

    # Quick live test preview (1 second) when debugging / no-fullscreen so the user
    # immediately sees whether the camera feed is producing frames before heavy imports.
    if args.no_fullscreen:
        try:
            test_start = time.time()
            cv2.namedWindow('Camera Test')
            while time.time() - test_start < 1.0:
                ret_test, frame_test = cap.read()
                if ret_test and frame_test is not None and getattr(frame_test, 'size', 0) > 0:
                    cv2.imshow('Camera Test', frame_test)
                else:
                    placeholder = (np.zeros((480, 640, 3), dtype='uint8'))
                    cv2.putText(placeholder, 'No Frame', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow('Camera Test', placeholder)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception:
            pass
        finally:
            try:
                cv2.destroyWindow('Camera Test')
            except Exception:
                pass

    # If requested, skip loading heavy models and run a simple camera-only loop
    if args.skip_models:
        print('[INFO] Running in --skip-models mode: camera-only display')
        try:
            if not args.no_fullscreen:
                cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            screen_w, screen_h = get_screen_size()
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or getattr(frame, 'size', 0) == 0:
                    placeholder = (np.zeros((480, 640, 3), dtype='uint8'))
                    cv2.putText(placeholder, 'No Frame', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    display = letterbox_to_fullscreen(placeholder, screen_w, screen_h)
                else:
                    display = letterbox_to_fullscreen(frame, screen_w, screen_h)
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            print('[INFO] Camera-only mode exited')
        return

    # Import heavy modules after camera is confirmed so import-time initializers don't hide camera errors
    from yolo_detector import load_yolo_model, run_yolo_scaled, confidence_threshold_for_mode
    from face_analyzer import create_face_landmarker, analyze_frame, new_mp_image
    from hand_analyzer import create_hand_landmarker, analyze_hands, classify_driver_hands, draw_hands

    # Load models
    yolo_model = load_yolo_model()
    face_landmarker = create_face_landmarker()
    # Create hand landmarker (basic detector, num_hands=2)
    try:
        hand_landmarker = create_hand_landmarker()
    except Exception as exc:
        # Bubble up the error so the user can place the model file if missing
        print(f"[ERROR] Could not create hand landmarker: {exc}")
        raise

    logger = EventLogger()

    # Background YOLO worker to keep detection off the main/UI thread
    import threading

    class YoloWorker:
        def __init__(self, model, scale, conf_fn, interval_s=0.25):
            self.model = model
            self.scale = scale
            self.conf_fn = conf_fn
            self.interval = interval_s
            self._lock = threading.Lock()
            self._frame = None
            self._night_active = False
            self._person_boxes = []
            self._phone_boxes = []
            self._last_result_ts = 0.0
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def set_frame(self, frame, night_active=False):
            with self._lock:
                # store a small copy to avoid holding references to large arrays
                self._frame = frame.copy() if frame is not None else None
                self._night_active = bool(night_active)

        def get_results(self):
            with self._lock:
                return list(self._person_boxes), list(self._phone_boxes)

        def last_result_age(self):
            try:
                return time.monotonic() - self._last_result_ts
            except Exception:
                return float('inf')

        def _run(self):
            while not self._stop.is_set():
                frame = None
                night = False
                with self._lock:
                    if self._frame is not None:
                        frame = self._frame
                        night = self._night_active
                        # clear stored frame so we don't re-run the same image repeatedly
                        self._frame = None
                if frame is not None:
                    try:
                        conf_threshold = self.conf_fn(night)
                        persons, phones = run_yolo_scaled(self.model, frame, self.scale, conf_threshold)
                        with self._lock:
                            self._person_boxes = persons
                            self._phone_boxes = phones
                            self._last_result_ts = time.monotonic()
                    except Exception as exc:
                        print(f"[ERROR] YOLO worker inference failed: {exc}")

                        with self._lock:
                            self._person_boxes = []
                            self._phone_boxes = []
                            self._last_result_ts = 0.0
                self._stop.wait(self.interval)

        def stop(self):
            self._stop.set()
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass

    yolo_worker = YoloWorker(yolo_model, config.YOLO_INFER_SCALE, confidence_threshold_for_mode)

    # Store no-fullscreen flag in config for later use
    no_fullscreen = args.no_fullscreen

    # Ensure previous OpenCV windows are closed (handle lingering windows from prior runs)
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # Create window (optionally avoid fullscreen when debugging)
    if not no_fullscreen:
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w, screen_h = get_screen_size()

    driver_side_in_image = resolve_driver_side_in_image()
    print(f"[INFO] Driver side in image: {driver_side_in_image}")

    # ---- state ----
    closed_eye_counter = 0
    yawn_counter = 0
    distraction_counter = 0
    consecutive_face_loss = 0
    last_valid_head_pose = "Center Focus"

    cached_person_boxes = []
    cached_phone_boxes = []
    frame_counter = 0
    bad_frame_streak = 0
    video_timestamp_ms = 0
    last_driver_hands_count = 0
    one_hand_start_ts = None
    two_hand_start_ts = None

    # Phone confirmation state
    phone_confirmation_counter = 0
    phone_present = False

    night_state = NightModeState()
    camera_is_in_night_mode = False

    try:
        while cap is not None:
            ret, frame = cap.read()

            # ---- Frame validation -------------------------------------------------
            if not ret or frame is None or frame.size == 0:
                bad_frame_streak += 1
                print(f"[WARN] Bad frame #{bad_frame_streak} (ret={ret}, frame={'None' if frame is None else frame.size})")
                if bad_frame_streak > 10:
                    print("[WARN] Camera lost. Reconnecting...")
                    cap.release()
                    cap = open_camera(
                        preferred_index=args.camera_index,
                        preferred_backend=args.backend,
                    )
                    if cap is None:
                        print("[ERROR] Reconnection failed. Exiting.")
                        break
                    bad_frame_streak = 0
                    continue
                continue
            bad_frame_streak = 0
            # -----------------------------------------------------------------------

            if config.FLIP_FRAME:
                frame = cv2.flip(frame, 1)
            h_max, w_max, _ = frame.shape

            # ---------------------------------------------------------
            # NIGHT VISION
            # ---------------------------------------------------------
            night_active = night_state.update(frame)
            camera_is_in_night_mode = sync_camera_to_light_mode(
                cap, night_active, camera_is_in_night_mode
            )

            if night_active:
                detection_frame = enhance_for_low_light(frame)
                display_frame = (
                    apply_night_vision_tint(detection_frame)
                    if config.NIGHT_VISION_DISPLAY_TINT
                    else detection_frame
                )
            else:
                detection_frame = frame
                display_frame = frame

            # -----------------------------------------------------
            # YOLO detection (throttled)
            # -----------------------------------------------------
            # Feed frames to the background YOLO worker at the same throttle as before
            if frame_counter % (config.YOLO_SKIP_FRAMES + 1) == 0:
                try:
                    yolo_worker.set_frame(detection_frame, night_active)
                except Exception:
                    pass
            frame_counter += 1

            # Read latest YOLO results from the worker (non-blocking)
            try:
                cached_person_boxes, cached_phone_boxes = yolo_worker.get_results()
            except Exception:
                cached_person_boxes, cached_phone_boxes = [], []

            # -----------------------------------------------------
            # Determine driver and occupancy
            # -----------------------------------------------------
            driver_box = None
            cabin_occupancy = "Empty Seat"
            driver_classification = "N/A"

            for box in cached_person_boxes:
                if is_driver_box(box, w_max, driver_side_in_image):
                    driver_box = box
                    break

            if driver_box is not None:
                x1, y1, x2, y2 = [int(c) for c in driver_box]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                driver_classification = "Driver (Adult)"
                cabin_occupancy = "Driver Active"
            elif cached_person_boxes:
                driver_classification = "Passenger"
                cabin_occupancy = "Passenger Present"

            # -----------------------------------------------------
            # Phone detection (driver-associated with temporal confirmation)
            # -----------------------------------------------------
            # Evaluate freshness of YOLO results
            try:
                age = yolo_worker.last_result_age()
            except Exception:
                age = float('inf')

            if age > config.PHONE_STALE_TIMEOUT:
                # stale results: reset confirmation and clear any confirmed state
                if phone_present:
                    print("[PHONE] Driver phone cleared (stale YOLO results)")
                phone_confirmation_counter = 0
                phone_present = False
            else:
                # Determine if any detected phone belongs to the driver by containment
                driver_phone_candidate = False
                if driver_box is not None:
                    for pbox in cached_phone_boxes:
                        if box_containment_ratio(pbox, driver_box) > config.PHONE_CONTAINMENT_THRESHOLD:
                            driver_phone_candidate = True
                            # draw candidate phone box on display (debug)
                            cv2.rectangle(display_frame, (int(pbox[0]), int(pbox[1])),
                                          (int(pbox[2]), int(pbox[3])), (0, 0, 255), 3)
                            break

                if driver_phone_candidate:
                    # Increment confirmation counter until limit
                    print("[PHONE] Driver phone candidate detected")
                    if not phone_present:
                        phone_confirmation_counter = min(config.PHONE_CONFIRMATION_LIMIT, phone_confirmation_counter + 1)
                        print(f"[PHONE] Confirmation: {phone_confirmation_counter}/{config.PHONE_CONFIRMATION_LIMIT}")
                        if phone_confirmation_counter >= config.PHONE_CONFIRMATION_LIMIT:
                            phone_present = True
                            print("[PHONE] DRIVER PHONE CONFIRMED")
                    else:
                        # already confirmed, keep counter at limit
                        phone_confirmation_counter = config.PHONE_CONFIRMATION_LIMIT
                else:
                    # No candidate this frame: decay counter so brief misses don't immediately clear
                    if phone_confirmation_counter > 0:
                        phone_confirmation_counter = max(0, phone_confirmation_counter - 1)
                    if phone_confirmation_counter == 0 and phone_present:
                        phone_present = False
                        print("[PHONE] Driver phone cleared")

            # -----------------------------------------------------
            # MediaPipe Face Landmarker
            # -----------------------------------------------------
            rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            mp_image = new_mp_image(rgb)
            video_timestamp_ms = int(time.monotonic() * 1000)

            face_result = analyze_frame(
                face_landmarker, mp_image, video_timestamp_ms, w_max, h_max
            )

            # Basic hand detection (MediaPipe Hand Landmarker)
            try:
                hand_result = analyze_hands(hand_landmarker, mp_image, video_timestamp_ms, w_max, h_max)
            except Exception as exc:
                print(f"[ERROR] Hand analysis failed: {exc}")
                hand_result = None

            # Driver-only hand classification based on overlap with the existing driver box.
            driver_hands = []
            passenger_hands = []
            driver_hands_count = 0
            one_hand_elapsed = 0.0
            two_hand_elapsed = 0.0
            if hand_result is not None:
                try:
                    driver_hands, passenger_hands = classify_driver_hands(hand_result, driver_box)
                except Exception:
                    driver_hands = []
                    passenger_hands = []

                driver_hands_count = len(driver_hands)

                if driver_hands_count == 1:
                    if last_driver_hands_count != 1:
                        one_hand_start_ts = time.monotonic()
                    one_hand_elapsed = time.monotonic() - one_hand_start_ts if one_hand_start_ts is not None else 0.0
                    two_hand_start_ts = None
                    two_hand_elapsed = 0.0
                elif driver_hands_count == 2:
                    if last_driver_hands_count != 2:
                        two_hand_start_ts = time.monotonic()
                    two_hand_elapsed = time.monotonic() - two_hand_start_ts if two_hand_start_ts is not None else 0.0
                    one_hand_start_ts = None
                    one_hand_elapsed = 0.0
                else:
                    one_hand_start_ts = None
                    two_hand_start_ts = None
                    one_hand_elapsed = 0.0
                    two_hand_elapsed = 0.0

                last_driver_hands_count = driver_hands_count

                try:
                    if driver_hands:
                        draw_hands(display_frame, [pts for pts, _ in driver_hands], color=(0, 255, 0), draw_box=True)
                    if passenger_hands:
                        draw_hands(display_frame, [pts for pts, _ in passenger_hands], color=(0, 165, 255), draw_box=True)
                    cv2.putText(
                        display_frame,
                        f"Driver hands: {driver_hands_count}",
                        (10, max(25, h_max - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    if driver_hands_count == 1:
                        cv2.putText(
                            display_frame,
                            f"One-hand time: {one_hand_elapsed:.1f} seconds",
                            (10, max(55, h_max - 35)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                    elif driver_hands_count == 2:
                        cv2.putText(
                            display_frame,
                            f"Two-hand time: {two_hand_elapsed:.1f} seconds",
                            (10, max(55, h_max - 35)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                except Exception:
                    pass
            else:
                driver_hands_count = 0
                one_hand_start_ts = None
                two_hand_start_ts = None
                one_hand_elapsed = 0.0
                two_hand_elapsed = 0.0
                last_driver_hands_count = 0

            eye_status = "Scanning"
            head_pose = "Center Focus"
            yawn_status = "Normal"

            if face_result.face_detected:
                consecutive_face_loss = 0
                fx, fy, fw, fh = face_result.bbox
                cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)

                rel_nose_x = face_result.head_pose_x_ratio
                if rel_nose_x < 0.40:
                    head_pose = "Looking Right"
                    last_valid_head_pose = "Looking Right"
                    distraction_counter += 1
                elif rel_nose_x > 0.60:
                    head_pose = "Looking Left"
                    last_valid_head_pose = "Looking Left"
                    distraction_counter += 1
                else:
                    head_pose = "Center Focus"
                    last_valid_head_pose = "Center Focus"
                    distraction_counter = max(0, distraction_counter - 2)

                avg_ear = face_result.avg_ear
                if avg_ear < config.EAR_THRESHOLD:
                    closed_eye_counter += 1
                    eye_status = f"Closed (EAR {avg_ear:.2f})"
                else:
                    closed_eye_counter = max(0, closed_eye_counter - 3)
                    eye_status = f"Open (EAR {avg_ear:.2f})"

                mar = face_result.mar
                if mar > config.MAR_THRESHOLD:
                    yawn_counter += 1
                    yawn_status = f"Yawning (MAR {mar:.2f})"
                    cv2.putText(display_frame, "YAWN", (fx, max(25, fy - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    yawn_counter = max(0, yawn_counter - 2)
                    yawn_status = f"Normal (MAR {mar:.2f})"
            else:
                consecutive_face_loss += 1
                if consecutive_face_loss <= 5:
                    head_pose = last_valid_head_pose
                    eye_status = "Searching..."
                elif last_valid_head_pose in ["Looking Left", "Looking Right"]:
                    head_pose = last_valid_head_pose
                    distraction_counter += 1
                    closed_eye_counter = max(0, closed_eye_counter - 1)
                    eye_status = "Face Turned Away"
                else:
                    head_pose = "Face Lost"
                    eye_status = "Unknown"
                    closed_eye_counter = max(0, closed_eye_counter - 1)

            # -----------------------------------------------------
            # Alert arbitration
            # -----------------------------------------------------
            violations = []
            if closed_eye_counter >= config.SLEEP_FRAME_LIMIT:
                violations.append(("CRITICAL: DRIVER SLEEP DETECTED", (0, 0, 255), 2200, 400))
            elif phone_present:
                violations.append(("MOBILE PHONE USAGE DETECTED", (0, 0, 255), 1400, 150))
            elif closed_eye_counter >= config.DROWSY_FRAME_LIMIT:
                violations.append(("WARNING: DROWSINESS DETECTED", (0, 165, 255), 900, 200))
            elif yawn_counter >= config.YAWN_FRAME_LIMIT:
                violations.append(("WARNING: YAWNING DETECTED", (0, 165, 255), 700, 250))
            elif distraction_counter >= config.DISTRACTION_FRAME_LIMIT:
                violations.append(("PLEASE FOCUS ON THE ROAD", (0, 165, 255), 550, 200))
            elif driver_hands_count == 2 and two_hand_start_ts is not None and two_hand_elapsed > config.TWO_HAND_MAX_SECONDS:
                violations.append(("CRITICAL: BOTH HANDS OFF WHEEL", (0, 0, 255), 1200, 300))
            elif driver_hands_count == 1 and one_hand_start_ts is not None and one_hand_elapsed > config.ONE_HAND_MAX_SECONDS:
                violations.append(("WARNING: ONE HAND OFF WHEEL TOO LONG", (0, 165, 255), 900, 250))

            if violations:
                active_alert, alert_color, beep_freq, beep_dur = violations[0]
                alert_beep(beep_freq, beep_dur)
            else:
                active_alert = "System Nominal"
                alert_color = (0, 255, 0)
                logger.reset()

            logger.log(cabin_occupancy, driver_classification, active_alert)

            # -----------------------------------------------------
            # HUD + Center Gridlines
            # -----------------------------------------------------
            metrics = [
                f"Cabin : {cabin_occupancy}",
                f"Driver: {driver_classification}",
                f"Eye   : {eye_status}",
                f"Pose  : {head_pose} [{distraction_counter}]",
                f"Phone : {'VIOLATION' if phone_present else 'Clean'}",
                f"Yawn  : {yawn_status}",
                f"Closed: {closed_eye_counter}",
                f"Face  : {consecutive_face_loss}",
                f"Light : {night_state.last_luma:.0f} luma",
            ]
            if driver_hands_count == 1:
                metrics.append(f"One-hand: {one_hand_elapsed:.1f}s")
            elif driver_hands_count == 2:
                metrics.append(f"Two-hand: {two_hand_elapsed:.1f}s")

            draw_zone_gridlines(display_frame, w_max, h_max, driver_side_in_image)
            draw_hud_panel(display_frame, w_max, metrics, night_active)
            draw_alert_banner(display_frame, w_max, h_max, active_alert, alert_color)

            # -----------------------------------------------------
            # Fullscreen letterbox + show
            # -----------------------------------------------------
            display = letterbox_to_fullscreen(display_frame, screen_w, screen_h)
            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        try:
            yolo_worker.stop()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            face_landmarker.close()
        except Exception:
            pass
        print(f"[INFO] Log saved to '{config.LOG_FILE}'.")


if __name__ == "__main__":
    main()