"""
Text Detector — EasyOCR Wrapper with 4K Tiling Support

Detects text regions in images using EasyOCR and returns
bounding boxes and detected text content. Text regions are
given the highest importance in SARC because text readability
is critical for information preservation.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SARCConfig, DEFAULT_CONFIG


class TextDetector:
    """Wraps EasyOCR for text region detection with tiling support."""

    def __init__(self, config: SARCConfig = None, languages: list = None):
        self.config = config or DEFAULT_CONFIG
        self.languages = languages or ["en"]
        self.reader = None

    def _load_model(self):
        """Lazy-load the EasyOCR reader on first use."""
        if self.reader is None:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=True)

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in the image, with tiling for large images.
        
        Args:
            image: BGR image array
        
        Returns:
            List of text region dicts with keys:
                - bbox: (x1, y1, x2, y2) in original image coordinates
                - text: Detected text string
                - confidence: OCR confidence [0, 1]
                - importance: Always 1.0 (text is highest priority)
                - detector_type: 'ocr'
        """
        self._load_model()
        h, w = image.shape[:2]

        if max(h, w) > self.config.tile_threshold:
            return self._detect_tiled(image)
        else:
            return self._detect_single(image)

    def _detect_single(self, image: np.ndarray) -> List[Dict]:
        """Run OCR on a single image."""
        # EasyOCR expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb)

        text_regions = []
        for (bbox_points, text, confidence) in results:
            # EasyOCR returns 4 corner points; convert to (x1,y1,x2,y2)
            pts = np.array(bbox_points)
            x1 = float(pts[:, 0].min())
            y1 = float(pts[:, 1].min())
            x2 = float(pts[:, 0].max())
            y2 = float(pts[:, 1].max())

            text_regions.append({
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "confidence": float(confidence),
                "importance": self.config.text_importance,
                "detector_type": "ocr"
            })

        return text_regions

    def _detect_tiled(self, image: np.ndarray) -> List[Dict]:
        """Run OCR on overlapping tiles for large images."""
        h, w = image.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.tile_overlap
        stride = tile_size - overlap

        all_regions = []

        for y_start in range(0, h, stride):
            for x_start in range(0, w, stride):
                y_end = min(y_start + tile_size, h)
                x_end = min(x_start + tile_size, w)

                tile = image[y_start:y_end, x_start:x_end]

                if tile.shape[0] < 64 or tile.shape[1] < 64:
                    continue

                tile_regions = self._detect_single(tile)

                # Remap coordinates to full image space
                for region in tile_regions:
                    region["bbox"][0] += x_start
                    region["bbox"][1] += y_start
                    region["bbox"][2] += x_start
                    region["bbox"][3] += y_start
                    all_regions.append(region)

        # Remove duplicates from overlapping tiles
        all_regions = self._deduplicate(all_regions)

        return all_regions

    def _deduplicate(
        self,
        regions: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Remove duplicate text detections from overlapping tiles."""
        if not regions:
            return []

        regions = sorted(regions, key=lambda r: r["confidence"], reverse=True)

        kept = []
        for region in regions:
            is_duplicate = False
            for kept_region in kept:
                iou = self._compute_iou(region["bbox"], kept_region["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(region)

        return kept

    @staticmethod
    def _compute_iou(bbox1: list, bbox2: list) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0
