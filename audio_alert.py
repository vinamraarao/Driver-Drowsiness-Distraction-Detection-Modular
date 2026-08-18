"""
audio_alert.py
--------------
Non-blocking beep/alert sound, cross-platform (Windows uses winsound,
everything else falls back to the terminal bell).
"""

import platform
import threading
import time

import config

_last_beep_time = 0.0
_lock = threading.Lock()

if platform.system() == "Windows":
    import winsound

    def _sync_beep(freq, dur):
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass

    def alert_beep(frequency, duration):
        global _last_beep_time
        now = time.time()
        with _lock:
            if now - _last_beep_time > max(config.BEEP_COOLDOWN_SEC, duration / 1000.0):
                _last_beep_time = now
                threading.Thread(target=_sync_beep, args=(frequency, duration), daemon=True).start()
else:
    def alert_beep(frequency, duration):
        global _last_beep_time
        now = time.time()
        with _lock:
            if now - _last_beep_time > config.BEEP_COOLDOWN_SEC:
                _last_beep_time = now
                print('\a', end='', flush=True)
