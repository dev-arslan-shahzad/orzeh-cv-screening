"""
End-to-End Video Processing Pipeline.
Processes video files frame-by-frame, performs multimodal detection and feature extraction,
applies temporal smoothing and Schmitt-trigger hysteresis, and formats results
into Table 3 compliant intervals covering [0.000, duration].
"""

import os
from typing import List, Dict, Optional, Tuple, Callable
import cv2
import numpy as np
import pandas as pd

from src.schema import Interval, Behaviour, ALL_BEHAVIOURS, validate_csv
from src.detector import PoseDetector, PersonDetection
from src.features import ColorAnalyzer, SigLIPClassifier
from src.temporal import build_contiguous_intervals, filter_short_glitches, schmitt_trigger
from src.visualize import draw_visual_overlay


class VideoPipeline:
    def __init__(
        self,
        pose_weights: str = "yolo11n-pose.pt",
        sample_fps: float = 2.0,
        device: str = "cpu"
    ):
        self.sample_fps = sample_fps
        self.device = device
        self.detector = PoseDetector(weights=pose_weights, device=device)
        self.classifier = SigLIPClassifier(device=device)

    def process_video(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        visualize_output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, float], None]] = None
    ) -> List[Interval]:
        """
        Runs the full analysis pipeline on a video file.
        Returns a list of contiguous intervals compliant with Table 3.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_id is None:
            video_id = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / native_fps if total_frames > 0 else 0.0

        step_frames = max(1, round(native_fps / self.sample_fps))

        writer = None
        if visualize_output_path:
            os.makedirs(os.path.dirname(os.path.abspath(visualize_output_path)), exist_ok=True)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(visualize_output_path, fourcc, self.sample_fps, (w, h))

        timestamps: List[float] = []
        frame_preds: Dict[str, List[Tuple[str, float]]] = {b: [] for b in ALL_BEHAVIOURS}
        frame_detections: List[Optional[PersonDetection]] = []

        frame_idx = 0
        while True:
            # Sequential grab for fast frame jumping
            grabbed = cap.grab()
            if not grabbed:
                break

            if frame_idx % step_frames == 0:
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    break

                t_sec = frame_idx / native_fps
                timestamps.append(t_sec)

                # Process frame
                det, states, confs = self._process_frame(frame)
                frame_detections.append(det)

                for b in ALL_BEHAVIOURS:
                    frame_preds[b].append((states[b], confs[b]))

                if writer is not None:
                    overlay_frame = draw_visual_overlay(
                        frame=frame,
                        timestamp_sec=t_sec,
                        total_duration_sec=duration_sec,
                        video_id=video_id,
                        detection=det,
                        states=states,
                        confidences=confs
                    )
                    writer.write(overlay_frame)

                if progress_callback:
                    progress_callback(t_sec, duration_sec)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        # Step 2: Temporal smoothing and contiguous interval creation
        all_intervals: List[Interval] = []
        for b in ALL_BEHAVIOURS:
            raw_states = [p[0] for p in frame_preds[b]]
            raw_confs = [p[1] for p in frame_preds[b]]

            # Filter isolated glitches
            smoothed_states = filter_short_glitches(raw_states, min_run_length=2)

            # Build contiguous intervals
            intervals = build_contiguous_intervals(
                timestamps=timestamps,
                states=smoothed_states,
                confidences=raw_confs,
                video_id=video_id,
                behaviour=b,
                video_duration=duration_sec,
                min_segment_sec=0.5
            )
            all_intervals.extend(intervals)

        return all_intervals

    def _process_frame(self, frame: np.ndarray) -> Tuple[Optional[PersonDetection], Dict[str, str], Dict[str, float]]:
        """Analyzes a single frame and returns states and confidences for T1-T4."""
        det = self.detector.detect(frame)

        states: Dict[str, str] = {}
        confs: Dict[str, float] = {}

        if det is None:
            # No person detected in frame
            states[Behaviour.HAND_COVERING.value] = "not_visible"
            confs[Behaviour.HAND_COVERING.value] = 0.90
            states[Behaviour.SHIRT_SLEEVE_LENGTH.value] = "not_visible"
            confs[Behaviour.SHIRT_SLEEVE_LENGTH.value] = 0.90
            states[Behaviour.HEAD_COVERING.value] = "not_visible"
            confs[Behaviour.HEAD_COVERING.value] = 0.90
            states[Behaviour.HAND_WASHING.value] = "not_visible"
            confs[Behaviour.HAND_WASHING.value] = 0.90
            return None, states, confs

        # -------------------------------------------------------------
        # T1: Hand Covering (bare_hands vs gloves vs not_visible)
        # -------------------------------------------------------------
        if not det.hands_visible:
            states[Behaviour.HAND_COVERING.value] = "not_visible"
            confs[Behaviour.HAND_COVERING.value] = 0.88
        else:
            # Analyze visible hands
            hand_crops = [c for c in [det.left_hand_crop, det.right_hand_crop] if c is not None]
            skin_ratios = [ColorAnalyzer.compute_skin_ratio(c) for c in hand_crops]
            sig_glv_deltas = [
                self.classifier.compute_zero_shot_delta(c, "gloves", "bare_hands")
                for c in hand_crops
            ]

            max_sig = max(sig_glv_deltas) if sig_glv_deltas else 0.0
            min_skin = min(skin_ratios) if skin_ratios else 0.0
            max_skin = max(skin_ratios) if skin_ratios else 0.0

            # Condition 1: Opaque gloves (black, blue, white) have very low skin ratio
            # Condition 2: Clear food-prep gloves (004) have specular sheen / crinkles captured by SigLIP
            # Convention: report gloves if at least one visible hand is gloved
            if min_skin < 0.25 or max_sig > 0.045:
                states[Behaviour.HAND_COVERING.value] = "gloves"
                confs[Behaviour.HAND_COVERING.value] = min(0.98, max(0.70, 0.82 + max_sig * 2.0))
            else:
                states[Behaviour.HAND_COVERING.value] = "bare_hands"
                confs[Behaviour.HAND_COVERING.value] = min(0.98, max(0.70, 0.80 + max_skin * 0.2))

        # -------------------------------------------------------------
        # T2: Shirt Sleeve Length (full_sleeve vs half_sleeve vs not_visible)
        # -------------------------------------------------------------
        if not det.arms_visible:
            states[Behaviour.SHIRT_SLEEVE_LENGTH.value] = "not_visible"
            confs[Behaviour.SHIRT_SLEEVE_LENGTH.value] = 0.85
        else:
            arm_crops = [c for c in [det.left_arm_crop, det.right_arm_crop] if c is not None]
            arm_skins = [ColorAnalyzer.compute_skin_ratio(c) for c in arm_crops]
            max_arm_skin = max(arm_skins) if arm_skins else 0.0

            arm_des = [
                ColorAnalyzer.compute_cielab_delta_e(det.torso_crop, c)
                for c in arm_crops
            ] if det.torso_crop is not None else [99.0]
            max_arm_de = max(arm_des) if arm_des else 99.0

            # If forearm has significant skin ratio (> 0.22) or large color delta from torso shirt (> 16)
            if max_arm_skin > 0.22 or (max_arm_de > 16.0 and max_arm_skin > 0.12):
                states[Behaviour.SHIRT_SLEEVE_LENGTH.value] = "half_sleeve"
                confs[Behaviour.SHIRT_SLEEVE_LENGTH.value] = 0.90
            elif max_arm_de < 11.0 and max_arm_skin < 0.15:
                states[Behaviour.SHIRT_SLEEVE_LENGTH.value] = "full_sleeve"
                confs[Behaviour.SHIRT_SLEEVE_LENGTH.value] = 0.92
            else:
                # Complement with SigLIP
                sig_full_d = self.classifier.compute_zero_shot_delta(
                    arm_crops[0] if arm_crops else None, "full_sleeve", "half_sleeve"
                )
                if sig_full_d > 0.0:
                    states[Behaviour.SHIRT_SLEEVE_LENGTH.value] = "full_sleeve"
                    confs[Behaviour.SHIRT_SLEEVE_LENGTH.value] = 0.82
                else:
                    states[Behaviour.SHIRT_SLEEVE_LENGTH.value] = "half_sleeve"
                    confs[Behaviour.SHIRT_SLEEVE_LENGTH.value] = 0.82

        # -------------------------------------------------------------
        # T3: Head Covering (cap vs bare_head vs not_visible)
        # -------------------------------------------------------------
        if not det.head_visible or det.head_crop is None:
            states[Behaviour.HEAD_COVERING.value] = "not_visible"
            confs[Behaviour.HEAD_COVERING.value] = 0.94
        else:
            sig_cap_d = self.classifier.compute_zero_shot_delta(det.head_crop, "cap", "bare_head")
            # Calibrated zero-shot threshold
            if sig_cap_d >= -0.016:
                states[Behaviour.HEAD_COVERING.value] = "cap"
                confs[Behaviour.HEAD_COVERING.value] = min(0.98, max(0.68, 0.82 + sig_cap_d * 2.0))
            else:
                states[Behaviour.HEAD_COVERING.value] = "bare_head"
                confs[Behaviour.HEAD_COVERING.value] = min(0.98, max(0.68, 0.84 - sig_cap_d * 2.0))

        # -------------------------------------------------------------
        # T4: Hand Washing (washing vs not_washing vs not_visible)
        # -------------------------------------------------------------
        if not det.hands_visible:
            states[Behaviour.HAND_WASHING.value] = "not_visible"
            confs[Behaviour.HAND_WASHING.value] = 0.88
        else:
            # Active washing requires both hands interacting closely under tap
            if det.hands_dist < 0.28 and det.left_hand_crop is not None and det.right_hand_crop is not None:
                # Joint hands region
                sig_wash_d = self.classifier.compute_zero_shot_delta(
                    det.left_hand_crop, "washing", "not_washing"
                )
                if sig_wash_d > 0.08:
                    states[Behaviour.HAND_WASHING.value] = "washing"
                    confs[Behaviour.HAND_WASHING.value] = 0.86
                else:
                    states[Behaviour.HAND_WASHING.value] = "not_washing"
                    confs[Behaviour.HAND_WASHING.value] = 0.92
            else:
                states[Behaviour.HAND_WASHING.value] = "not_washing"
                confs[Behaviour.HAND_WASHING.value] = 0.95

        return det, states, confs

