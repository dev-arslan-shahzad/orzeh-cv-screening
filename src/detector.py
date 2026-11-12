"""
Pose and Body-Part Detector using YOLO11n-pose.
Extracts person bounding boxes, 17 COCO keypoints, and geometrically
crops head, torso, arm segments, and hands for downstream zero-shot & color analysis.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO


# COCO Pose Keypoint Indices
NOSE = 0
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12


@dataclass
class PersonDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    conf: float
    keypoints: np.ndarray  # [17, 3] (x, y, conf)
    
    # Derived crops (BGR numpy arrays)
    head_crop: Optional[np.ndarray] = None
    torso_crop: Optional[np.ndarray] = None
    left_arm_crop: Optional[np.ndarray] = None
    right_arm_crop: Optional[np.ndarray] = None
    left_hand_crop: Optional[np.ndarray] = None
    right_hand_crop: Optional[np.ndarray] = None

    # Geometry features
    hands_dist: float = 1.0  # Normalized inter-wrist distance
    head_visible: bool = False
    hands_visible: bool = False
    arms_visible: bool = False


class PoseDetector:
    def __init__(self, weights: str = "yolo11n-pose.pt", device: str = "cpu"):
        self.model = YOLO(weights)
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> Optional[PersonDetection]:
        """
        Runs pose detection on a single BGR frame.
        Selects the most dominant / foreground person and extracts body crops.
        """
        h_img, w_img = frame_bgr.shape[:2]
        results = self.model(frame_bgr, verbose=False, device=self.device)
        
        if not results or len(results) == 0:
            return None

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return None

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts_data = res.keypoints.data.cpu().numpy()  # [N, 17, 3]

        if len(boxes) == 0:
            return None

        # Select dominant person: largest area with conf > 0.35
        best_idx = -1
        best_score = -1.0
        for i in range(len(boxes)):
            if confs[i] < 0.30:
                continue
            x1, y1, x2, y2 = boxes[i]
            area = (x2 - x1) * (y2 - y1)
            # Area-weighted confidence
            score = area * confs[i]
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            return None

        bbox = boxes[best_idx]
        conf = float(confs[best_idx])
        kpts = kpts_data[best_idx]  # [17, 3]

        det = PersonDetection(bbox=bbox, conf=conf, keypoints=kpts)
        self._extract_crops(frame_bgr, det)
        return det

    def _extract_crops(self, img: np.ndarray, det: PersonDetection):
        h_img, w_img = img.shape[:2]
        kpts = det.keypoints
        x1, y1, x2, y2 = det.bbox

        # Torso scale reference
        ls = kpts[L_SHOULDER]
        rs = kpts[R_SHOULDER]
        lh = kpts[L_HIP]
        rh = kpts[R_HIP]

        shoulder_width = 100.0
        if ls[2] > 0.3 and rs[2] > 0.3:
            shoulder_width = np.linalg.norm(ls[:2] - rs[:2])

        # 1. Head Crop & Visibility
        # Check if head is cut off by the top of the frame
        nose = kpts[NOSE]
        lear = kpts[L_EAR]
        rear = kpts[R_EAR]
        
        head_confs = [kpts[i][2] for i in [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]]
        max_head_conf = max(head_confs) if head_confs else 0.0

        # If top bbox is at edge (y1 < 10) and nose/eyes not confident, head is not visible
        if y1 < 15 and max_head_conf < 0.40:
            det.head_visible = False
            det.head_crop = None
        elif max_head_conf >= 0.30:
            # Anchor head box around nose or midpoint between ears/shoulders
            hx = nose[0] if nose[2] > 0.3 else (0.5 * (ls[0] + rs[0]))
            hy = nose[1] if nose[2] > 0.3 else (0.5 * (ls[1] + rs[1]) - 0.4 * shoulder_width)
            
            box_size = max(shoulder_width * 0.9, 80.0)
            # Shift upward to capture caps/hairnets
            head_y1 = int(max(0, hy - box_size * 0.75))
            head_y2 = int(min(h_img, hy + box_size * 0.35))
            head_x1 = int(max(0, hx - box_size * 0.55))
            head_x2 = int(min(w_img, hx + box_size * 0.55))

            if head_y2 - head_y1 > 20 and head_x2 - head_x1 > 20:
                det.head_crop = img[head_y1:head_y2, head_x1:head_x2]
                det.head_visible = True
            else:
                det.head_visible = False
        else:
            det.head_visible = False

        # 2. Torso Crop (for shirt baseline color)
        if ls[2] > 0.3 and rs[2] > 0.3:
            tx1 = int(max(0, min(ls[0], rs[0]) + 0.15 * shoulder_width))
            tx2 = int(min(w_img, max(ls[0], rs[0]) - 0.15 * shoulder_width))
            ty1 = int(max(0, max(ls[1], rs[1]) + 0.10 * shoulder_width))
            ty2 = int(min(h_img, ty1 + 0.60 * shoulder_width))
            if tx2 > tx1 + 10 and ty2 > ty1 + 10:
                det.torso_crop = img[ty1:ty2, tx1:tx2]

        # 3. Arm and Hand Crops
        # Left Hand
        lw = kpts[L_WRIST]
        le = kpts[L_ELBOW]
        if lw[2] > 0.30:
            hand_size = max(shoulder_width * 0.40, 50.0)
            # Project hand center slightly beyond wrist along elbow->wrist direction
            if le[2] > 0.30:
                v = lw[:2] - le[:2]
                norm_v = np.linalg.norm(v)
                v_dir = v / norm_v if norm_v > 1e-3 else np.array([0.0, 1.0])
                hx, hy = lw[:2] + v_dir * (hand_size * 0.35)
            else:
                hx, hy = lw[:2]

            hx1 = int(max(0, hx - hand_size * 0.5))
            hx2 = int(min(w_img, hx + hand_size * 0.5))
            hy1 = int(max(0, hy - hand_size * 0.5))
            hy2 = int(min(h_img, hy + hand_size * 0.5))
            if hx2 > hx1 + 15 and hy2 > hy1 + 15:
                det.left_hand_crop = img[hy1:hy2, hx1:hx2]

        # Right Hand
        rw = kpts[R_WRIST]
        re = kpts[R_ELBOW]
        if rw[2] > 0.30:
            hand_size = max(shoulder_width * 0.40, 50.0)
            if re[2] > 0.30:
                v = rw[:2] - re[:2]
                norm_v = np.linalg.norm(v)
                v_dir = v / norm_v if norm_v > 1e-3 else np.array([0.0, 1.0])
                hx, hy = rw[:2] + v_dir * (hand_size * 0.35)
            else:
                hx, hy = rw[:2]

            hx1 = int(max(0, hx - hand_size * 0.5))
            hx2 = int(min(w_img, hx + hand_size * 0.5))
            hy1 = int(max(0, hy - hand_size * 0.5))
            hy2 = int(min(h_img, hy + hand_size * 0.5))
            if hx2 > hx1 + 15 and hy2 > hy1 + 15:
                det.right_hand_crop = img[hy1:hy2, hx1:hx2]

        det.hands_visible = (det.left_hand_crop is not None or det.right_hand_crop is not None)

        # Inter-wrist distance
        if lw[2] > 0.30 and rw[2] > 0.30:
            dist_px = np.linalg.norm(lw[:2] - rw[:2])
            det.hands_dist = float(dist_px / max(shoulder_width, 1.0))
        else:
            det.hands_dist = 1.0

        # Arm crops (elbow to wrist forearm region)
        det.arms_visible = False
        if le[2] > 0.30 and lw[2] > 0.30:
            det.arms_visible = True
            ax1 = int(max(0, min(le[0], lw[0]) - 15))
            ax2 = int(min(w_img, max(le[0], lw[0]) + 15))
            ay1 = int(max(0, min(le[1], lw[1]) - 15))
            ay2 = int(min(h_img, max(le[1], lw[1]) + 15))
            if ax2 > ax1 + 10 and ay2 > ay1 + 10:
                det.left_arm_crop = img[ay1:ay2, ax1:ax2]

        if re[2] > 0.30 and rw[2] > 0.30:
            det.arms_visible = True
            ax1 = int(max(0, min(re[0], rw[0]) - 15))
            ax2 = int(min(w_img, max(re[0], rw[0]) + 15))
            ay1 = int(max(0, min(re[1], rw[1]) - 15))
            ay2 = int(min(h_img, max(re[1], rw[1]) + 15))
            if ax2 > ax1 + 10 and ay2 > ay1 + 10:
                det.right_arm_crop = img[ay1:ay2, ax1:ax2]

