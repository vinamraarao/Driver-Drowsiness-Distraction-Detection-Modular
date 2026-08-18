"""
event_logger.py
----------------
Appends one row to the CSV log every time the active alert *changes*
(so we get a clean event history instead of one row per frame).
"""

import csv
from datetime import datetime

import config


class EventLogger:
    def __init__(self, log_file=None):
        self.log_file = log_file or config.LOG_FILE
        self._last_logged_event = ""

    def reset(self):
        """Call when the system returns to nominal, so the next real
        event logs again even if it repeats the previous label."""
        self._last_logged_event = ""

    def log(self, cabin, identity, event):
        if event != self._last_logged_event and event != "System Nominal":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, cabin, identity, event])
            self._last_logged_event = event
