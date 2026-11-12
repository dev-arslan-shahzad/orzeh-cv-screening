"""
Visualization and Demo Overlay Generator.
Draws bounding boxes, keypoint skeletons, region crops, current behaviour states,
and timeline status bar onto video frames to generate clear visual proof / demonstration video.
"""

from typing import Optional, Dict
import numpy as np
import cv2
from src.detector import PersonDetection


# Color palette (BGR)
CLR_BG = (24, 24, 27)         # Dark zinc
CLR_PANEL = (39, 39, 42)      # Zinc 800
CLR_TEXT = (244, 244, 245)    # Zinc 100
CLR_MUTED = (161, 161, 170)   # Zinc 400
CLR_ACCENT = (14, 165, 233)   # Sky 500
CLR_GREEN = (34, 197, 94)     # Emerald 500
CLR_YELLOW = (234, 179, 8)    # Amber 500
CLR_GRAY = (113, 113, 122)    # Zinc 500
CLR_BLUE = (59, 130, 246)     # Blue 500

STATE_COLORS = {
    "gloves": CLR_GREEN,
    "bare_hands": CLR_ACCENT,
    "full_sleeve": CLR_BLUE,
    "half_sleeve": CLR_YELLOW,
    "cap": CLR_GREEN,
    "bare_head": CLR_ACCENT,
    "washing": CLR_GREEN,
    "not_washing": CLR_MUTED,
    "not_visible": CLR_GRAY,
}

# Skeleton connection pairs for COCO pose
SKELETON_PAIRS = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), # torso sides
    (11, 12),         # hips
]


def draw_visual_overlay(
    frame: np.ndarray,
    timestamp_sec: float,
    total_duration_sec: float,
    video_id: str,
    detection: Optional[PersonDetection],
    states: Dict[str, str],
    confidences: Dict[str, float],
) -> np.ndarray:
    """
    Renders professional candidate evaluation HUD onto the video frame.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Skeleton and Person Bounding Box if detected
    if detection is not None:
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 200, 120), 2)

        kpts = detection.keypoints
        for idx1, idx2 in SKELETON_PAIRS:
            p1, p2 = kpts[idx1], kpts[idx2]
            if p1[2] > 0.35 and p2[2] > 0.35:
                cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 200, 50), 2)

        for i in range(len(kpts)):
            if kpts[i][2] > 0.35:
                cv2.circle(canvas, (int(kpts[i][0]), int(kpts[i][1])), 4, (0, 140, 255), -1)

    # 2. Top-Left State Dashboard
    panel_w = 420
    panel_h = 210
    pad = 16

    overlay = canvas.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (pad, pad), (pad + panel_w, pad + panel_h), CLR_PANEL, 1)

    # Title & Time Header
    header_text = f"Orzeh CV Screen | Video: {video_id} | {timestamp_sec:05.1f}s / {total_duration_sec:05.1f}s"
    cv2.putText(canvas, header_text, (pad + 12, pad + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_TEXT, 1, cv2.LINE_AA)
    cv2.line(canvas, (pad + 12, pad + 36), (pad + panel_w - 12, pad + 36), CLR_PANEL, 1)

    # Behaviour Rows
    rows = [
        ("T1 Hand Covering", states.get("hand_covering", "not_visible"), confidences.get("hand_covering", 0.0)),
        ("T2 Sleeve Length", states.get("shirt_sleeve_length", "not_visible"), confidences.get("shirt_sleeve_length", 0.0)),
        ("T3 Head Covering", states.get("head_covering", "not_visible"), confidences.get("head_covering", 0.0)),
        ("T4 Hand Washing", states.get("hand_washing", "not_visible"), confidences.get("hand_washing", 0.0)),
    ]

    y_offset = pad + 66
    for label, state_val, conf_val in rows:
        clr = STATE_COLORS.get(state_val, CLR_TEXT)
        cv2.putText(canvas, f"{label}:", (pad + 14, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_MUTED, 1, cv2.LINE_AA)
        val_str = f"{state_val.upper()} ({conf_val:.2f})"
        cv2.putText(canvas, val_str, (pad + 190, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 2, cv2.LINE_AA)
        y_offset += 38

    # 3. Bottom Progress Bar
    bar_y = h - 22
    bar_h = 10
    bar_w = w - (pad * 2)
    cv2.rectangle(canvas, (pad, bar_y), (pad + bar_w, bar_y + bar_h), CLR_PANEL, -1)
    
    prog = min(1.0, max(0.0, timestamp_sec / max(total_duration_sec, 0.001)))
    fill_w = int(bar_w * prog)
    if fill_w > 0:
        cv2.rectangle(canvas, (pad, bar_y), (pad + fill_w, bar_y + bar_h), CLR_ACCENT, -1)

    return canvas

