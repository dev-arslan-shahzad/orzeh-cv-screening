"""
Visual and Semantic Feature Extractors.
Combines classical computer vision (skin segmentation, CIELAB delta-E) with
open-source zero-shot vision (OpenCLIP ViT-B-16-SigLIP) for robust state prediction.
Strictly local, open-source, and free of commercial/cloud APIs.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import Tuple, Optional, Dict, List
import numpy as np
import cv2
import torch
from PIL import Image
import open_clip


class ColorAnalyzer:
    """Classical Computer Vision for skin color and Lab color difference."""

    @staticmethod
    def compute_skin_ratio(crop_bgr: Optional[np.ndarray]) -> float:
        """Calculates proportion of skin pixels in BGR crop using joint YCrCb and HSV."""
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0

        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv1 = cv2.inRange(hsv, np.array([0, 30, 50]), np.array([25, 240, 255]))
        mask_hsv2 = cv2.inRange(hsv, np.array([165, 30, 50]), np.array([180, 240, 255]))
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        skin = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        return float(np.count_nonzero(skin) / skin.size)

    @staticmethod
    def compute_cielab_delta_e(crop1_bgr: Optional[np.ndarray], crop2_bgr: Optional[np.ndarray]) -> float:
        """Calculates Euclidean color difference Delta E between two crops in CIELAB space."""
        if crop1_bgr is None or crop2_bgr is None or crop1_bgr.size == 0 or crop2_bgr.size == 0:
            return 99.0

        lab1 = cv2.cvtColor(crop1_bgr, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(crop2_bgr, cv2.COLOR_BGR2LAB)

        mean1 = np.mean(lab1, axis=(0, 1))
        mean2 = np.mean(lab2, axis=(0, 1))

        # Standard Euclidean distance in L*a*b* space (CIE76)
        delta_e = float(np.linalg.norm(mean1 - mean2))
        return delta_e


class SigLIPClassifier:
    """Zero-shot vision classifier using OpenCLIP ViT-B-16-SigLIP (webli)."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", pretrained="webli", device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
        self.model.eval()

        self._init_text_embeddings()

    def _init_text_embeddings(self):
        """Precomputes normalized text embeddings for target prompts."""
        self.prompts = {
            "cap": [
                "a person wearing a chef hat, hairnet, baseball cap, beanie, or kitchen head covering",
                "a chef wearing a white toque or cooking hat"
            ],
            "bare_head": [
                "a person with bare head and uncovered hair, no hat, no cap, no hairnet",
                "a person with bare hair and exposed head"
            ],
            "full_sleeve": [
                "a person wearing a long sleeve shirt covering arms down to wrists",
                "arms covered in fabric sleeves down to wrists"
            ],
            "half_sleeve": [
                "a person wearing a short sleeve t-shirt with bare arms and skin",
                "bare arms with skin exposed from elbows to wrists"
            ],
            "gloves": [
                "a hand wearing gloves, rubber nitrile gloves, disposable food prep gloves",
                "a gloved hand in kitchen food preparation"
            ],
            "bare_hands": [
                "a bare human hand with natural skin and bare fingers, no gloves",
                "uncovered bare hand holding food"
            ],
            "washing": [
                "hands actively washing under running water from a faucet tap at a kitchen sink",
                "hands being washed and lathered with soap and water at a sink"
            ],
            "not_washing": [
                "hands working at a kitchen prep counter cutting, cooking, or handling food, not washing",
                "hands resting or prepping ingredients on a stainless steel table"
            ],
        }

        self.text_features = {}
        with torch.no_grad():
            for key, prompt_list in self.prompts.items():
                tokens = self.tokenizer(prompt_list).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                mean_feat = feats.mean(dim=0, keepdim=True)
                mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
                self.text_features[key] = mean_feat

    def compute_zero_shot_delta(
        self, crop_bgr: Optional[np.ndarray], pos_key: str, neg_key: str
    ) -> float:
        """
        Computes cosine similarity delta: sim(crop, pos) - sim(crop, neg).
        Returns positive when crop aligns more with pos_key than neg_key.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            pos_feat = self.text_features[pos_key]
            neg_feat = self.text_features[neg_key]

            pos_sim = float((img_feat @ pos_feat.T).item())
            neg_sim = float((img_feat @ neg_feat.T).item())

        return pos_sim - neg_sim

