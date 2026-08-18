"""
hud.py
------
Everything that draws on top of the frame: the metrics panel (top horizontal bar),
center gridlines, the bottom alert banner, and the night-mode badge.
"""

import cv2

import config


def _fit_text_to_width(text, max_width, font_scale, thickness=1):
    """Return a text string that fits within max_width, preserving readability."""
    if max_width <= 0 or text is None:
        return ""

    candidate = text
    current_scale = font_scale
    while True:
        (tw, th), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, current_scale, thickness)
        if tw <= max_width:
            return candidate
        if current_scale <= 0.25:
            if len(candidate) > 3:
                candidate = candidate[:-3] + "..."
            else:
                return candidate[:max(1, min(len(candidate), 3))]
            continue
        current_scale = max(0.25, current_scale - 0.05)


def draw_hud_panel(frame, w_max, metrics, night_active):
    """Draw a compact horizontal HUD bar at the top of the frame."""
    hud_x = config.HUD_X_OFFSET
    hud_y = config.HUD_Y_OFFSET
    hud_w = w_max - (config.HUD_X_OFFSET * 2)
    hud_h = config.HUD_HEIGHT

    panel_color = (35, 15, 15) if night_active else (35, 35, 35)
    border_color = (0, 140, 0) if night_active else (120, 120, 120)

    cv2.rectangle(frame, (hud_x, hud_y),
                  (hud_x + hud_w, hud_y + hud_h),
                  panel_color, cv2.FILLED)
    cv2.rectangle(frame, (hud_x, hud_y),
                  (hud_x + hud_w, hud_y + hud_h),
                  border_color, 2)

    # Split metrics across two rows for compact horizontal layout
    mid = (len(metrics) + 1) // 2
    row1 = metrics[:mid]
    row2 = metrics[mid:]

    row_padding = 10
    row_max_w = max(120, hud_w - row_padding * 2)
    y_pos = hud_y + 20
    line1 = " | ".join(row1)
    line1 = _fit_text_to_width(line1, row_max_w, 0.40, 1)
    cv2.putText(frame, line1, (hud_x + row_padding, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    if row2:
        y_pos += 20
        line2 = " | ".join(row2)
        line2 = _fit_text_to_width(line2, row_max_w, 0.40, 1)
        cv2.putText(frame, line2, (hud_x + row_padding, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    if night_active:
        badge_text = "NIGHT VISION ACTIVE"
        (tw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_x = max(hud_x + 8, min(hud_x + hud_w - tw - 8, w_max - tw - 8))
        cv2.putText(frame, badge_text, (badge_x, hud_y + hud_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)


def draw_zone_gridlines(frame, w_max, h_max, driver_side):
    """Draw a vertical center divider and zone labels."""
    cx = w_max // 2
    color = (0, 200, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Vertical center line
    cv2.line(frame, (cx, 0), (cx, h_max), color, 2)

    # Subtle zone tint
    overlay = frame.copy()
    if driver_side == 'right':
        cv2.rectangle(overlay, (cx, 0), (w_max, h_max), (0, 60, 0), cv2.FILLED)
    else:
        cv2.rectangle(overlay, (0, 0), (cx, h_max), (0, 60, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Zone labels just below the HUD bar
    label_y = config.HUD_Y_OFFSET + config.HUD_HEIGHT + 22
    if driver_side == 'right':
        cv2.putText(frame, "DRIVER ZONE", (cx + 10, label_y),
                    font, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "PASSENGER ZONE", (10, label_y),
                    font, 0.55, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "DRIVER ZONE", (10, label_y),
                    font, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "PASSENGER ZONE", (cx + 10, label_y),
                    font, 0.55, color, 2, cv2.LINE_AA)


def draw_alert_banner(frame, w_max, h_max, active_alert, alert_color):
    banner_y1 = max(10, h_max - 60)
    banner_y2 = h_max - 10
    banner_x1 = 10
    banner_x2 = w_max - 10
    cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2),
                  alert_color, cv2.FILLED)

    max_text_width = max(120, banner_x2 - banner_x1 - 24)
    font_scale = 0.7
    display_text = _fit_text_to_width(active_alert, max_text_width, font_scale, 2)
    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    text_x = max(banner_x1 + 12, min((w_max - text_size[0]) // 2, banner_x2 - text_size[0] - 12))
    text_y = banner_y1 + (banner_y2 - banner_y1 + text_size[1]) // 2
    cv2.putText(frame, display_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def letterbox_to_fullscreen(frame, screen_w, screen_h):
    import numpy as np
    h_max, w_max = frame.shape[:2]
    scale = min(screen_w / w_max, screen_h / h_max)
    new_w, new_h = int(w_max * scale), int(h_max * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    display = np.zeros((screen_h, screen_w, 3), dtype=frame.dtype)
    y_off = (screen_h - new_h) // 2
    x_off = (screen_w - new_w) // 2
    display[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return display


def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080