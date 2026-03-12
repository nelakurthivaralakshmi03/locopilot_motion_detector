# utils/draw.py
# ──────────────────────────────────────────────────────────────────
# All OpenCV drawing helpers used to annotate the video frames
# ──────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Colour palette (BGR) ───────────────────────────────────────────
CLR_PILOT_1   = (0,   200, 255)   # cyan
CLR_PILOT_2   = (255, 165,   0)   # amber
CLR_ALERT     = (0,     0, 255)   # red   – distraction active
CLR_SAFE      = (0,   220,   0)   # green – no distraction
CLR_GADGET    = (0,    50, 255)   # bright red for gadget box
CLR_TEXT_BG   = (20,   20,  20)   # near-black background for text

# New colour for seat-absence visuals (magenta/purple — distinct
# from red gadget alerts so operator can tell distraction types apart)
CLR_ABSENCE   = (200,   0, 200)   # magenta – pilot away from seat
CLR_CALIB     = (180, 180,   0)   # yellow  – calibration in progress


def put_text(
    frame:  np.ndarray,
    text:   str,
    pos:    Tuple[int, int],
    colour: Tuple[int, int, int] = (220, 220, 220),
    scale:  float = 0.52,
    thick:  int   = 1,
) -> None:
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame,
                  (x - 2,      y - th - 3),
                  (x + tw + 2, y + bl + 1),
                  CLR_TEXT_BG, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def draw_pilot_box(
    frame:       np.ndarray,
    bbox:        Tuple[int, int, int, int],
    pilot_id:    int,
    distracted:  bool,
    gadgets:     List[str],
) -> None:
    """
    Draw the pilot bounding box + status label.
    Box turns RED when a gadget is detected, GREEN when clean.
    — UNCHANGED from original —
    """
    x1, y1, x2, y2 = bbox
    base_colour = CLR_PILOT_1 if pilot_id == 1 else CLR_PILOT_2
    box_colour  = CLR_ALERT if distracted else base_colour

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

    status = "DISTRACTED" if distracted else "OK"
    label  = f"Pilot {pilot_id}  [{status}]"
    put_text(frame, label, (x1, y1 - 8), box_colour, scale=0.55)

    for i, g in enumerate(gadgets):
        put_text(
            frame,
            f"  Gadget: {g}",
            (x1, y2 + 18 + i * 18),
            CLR_ALERT,
            scale=0.46,
        )


def draw_gadget_box(
    frame: np.ndarray,
    bbox:  Tuple[int, int, int, int],
    label: str,
    conf:  float,
) -> None:
    """Draw bounding box around the detected gadget object.
    — UNCHANGED from original —
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_GADGET, 2)
    put_text(
        frame,
        f"{label} {conf:.0%}",
        (x1, y1 - 6),
        CLR_GADGET,
        scale=0.45,
    )


def draw_hud(
    frame:      np.ndarray,
    video_time: float,
    frame_no:   int,
    pilot_count: int,
) -> None:
    """Top-left heads-up display: time, frame, pilot count.
    — UNCHANGED from original —
    """
    hh = int(video_time) // 3600
    mm = (int(video_time) % 3600) // 60
    ss = int(video_time) % 60
    put_text(
        frame,
        f"Time {hh:02d}:{mm:02d}:{ss:02d}  |  Frame {frame_no}  |  Pilots detected: {pilot_count}",
        (10, 22),
        (200, 200, 200),
        scale=0.50,
    )


def draw_alert_banner(frame: np.ndarray, pilot_id: int, gadget: str) -> None:
    """Full-width red banner at the bottom when distraction is critical.
    — UNCHANGED from original —
    """
    h, w = frame.shape[:2]
    banner_h = 36
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 180), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    msg = f"  !! CRITICAL - Pilot {pilot_id} using {gadget} !!"
    put_text(frame, msg, (8, h - 10), (255, 255, 255), scale=0.58, thick=1)


# ══════════════════════════════════════════════════════════════════
# SEAT ABSENCE DRAWING  (new — all functions below are additions)
# Nothing above this line was changed.
# ══════════════════════════════════════════════════════════════════

def draw_seat_zone(
    frame:     np.ndarray,
    seat_zone: Tuple[int, int, int, int],
    pilot_id:  int,
) -> None:
    """
    Draw the calibrated seat zone as a dashed rectangle.
    Shown in cyan/amber (same as pilot colour) so operator can see
    what zone each pilot is expected to stay within.
    """
    colour = CLR_PILOT_1 if pilot_id == 1 else CLR_PILOT_2
    x1, y1, x2, y2 = seat_zone
    _draw_dashed_rect(frame, x1, y1, x2, y2, colour, dash_len=10)


def draw_absence_overlay(
    frame:      np.ndarray,
    bbox:       Optional[Tuple[int,int,int,int]],
    pilot_id:   int,
    absent:     bool,
    timer_val:  float,
    calibrated: bool,
) -> None:
    """
    Draw absence status for one pilot:
      • During calibration  → yellow "CALIBRATING" label at seat zone
      • Pilot absent        → magenta "AWAY FROM SEAT" warning
      • Pilot present       → nothing extra drawn (pilot box handles it)

    Parameters
    ──────────
    bbox       : current pilot bbox if detected, else None
    pilot_id   : 1 or 2
    absent     : True when pilot has been away > threshold
    timer_val  : seconds the pilot has been absent
    calibrated : False during the initial calibration window
    """
    if not calibrated:
        # Show calibration-in-progress near top of frame, per pilot
        x_offset = 10 if pilot_id == 1 else 320
        put_text(
            frame,
            f"Pilot {pilot_id}: Calibrating seat...",
            (x_offset, 45),
            CLR_CALIB,
            scale=0.48,
        )
        return

    if absent:
        # If we still have a bbox (pilot partially visible / moved)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_ABSENCE, 3)
            put_text(
                frame,
                f"Pilot {pilot_id}  [AWAY FROM SEAT  {timer_val:.1f}s]",
                (x1, y1 - 8),
                CLR_ABSENCE,
                scale=0.55,
            )
        else:
            # Pilot completely out of frame — show warning in fixed position
            x_offset = 10 if pilot_id == 1 else 320
            put_text(
                frame,
                f"!! Pilot {pilot_id} NOT IN FRAME  [{timer_val:.1f}s] !!",
                (x_offset, 45),
                CLR_ABSENCE,
                scale=0.55,
            )


def draw_absence_banner(
    frame:    np.ndarray,
    pilot_id: int,
    duration: float,
) -> None:
    """
    Full-width MAGENTA banner at the bottom when pilot is away from
    seat past the threshold. Drawn ABOVE the gadget alert banner so
    both can be visible simultaneously if needed — the gadget banner
    stays at the very bottom (y = h-10) while this one sits at y-48.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 72), (w, h - 36), (150, 0, 150), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    msg = f"  !! CRITICAL - Pilot {pilot_id} AWAY FROM SEAT  ({duration:.1f}s) !!"
    put_text(frame, msg, (8, h - 46), (255, 255, 255), scale=0.58, thick=1)


# ── internal helper ───────────────────────────────────────────────

def _draw_dashed_rect(
    frame:    np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour:   Tuple[int,int,int],
    dash_len: int = 8,
    thick:    int = 1,
) -> None:
    """Draw a dashed rectangle using short line segments."""
    pts = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (sx, sy), (ex, ey) in pts:
        length = int(np.hypot(ex - sx, ey - sy))
        if length == 0:
            continue
        steps = max(1, length // (dash_len * 2))
        for i in range(steps):
            t0 = (2 * i    ) / (2 * steps)
            t1 = (2 * i + 1) / (2 * steps)
            p0 = (int(sx + t0*(ex-sx)), int(sy + t0*(ey-sy)))
            p1 = (int(sx + t1*(ex-sx)), int(sy + t1*(ey-sy)))
            cv2.line(frame, p0, p1, colour, thick)

