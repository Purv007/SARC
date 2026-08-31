"""
YOLO Object Detector — Wrapper for YOLOv8 with 4K Tiling Support

Detects objects using a pretrained YOLOv8 model and returns
bounding boxes, class names, confidence scores, and importance weights.

For images larger than the tile threshold, detection is performed
on overlapping tiles with coordinate remapping to full image space.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SARCConfig, DEFAULT_CONFIG


class ObjectDetector:
    """Wraps YOLOv8 for object detection with tiling support for large images."""

    def __init__(self, config: SARCConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.model = None

    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.config.yolo_model)

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run object detection on the image, with tiling for large images.
        
        Args:
            image: BGR image array
        
        Returns:
            List of detection dicts with keys:
                - bbox: (x1, y1, x2, y2) in original image coordinates
                - class_name: YOLO class name string
                - confidence: Detection confidence [0, 1]
                - importance: Importance score from config lookup
        """
        self._load_model()
        h, w = image.shape[:2]

        # Check if tiling is needed for large images
        if max(h, w) > self.config.tile_threshold:
            return self._detect_tiled(image)
        else:
            return self._detect_single(image)

    def _detect_single(self, image: np.ndarray) -> List[Dict]:
        """Run detection on a single image (no tiling)."""
        results = self.model(
            image,
            conf=self.config.yolo_confidence,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                class_name = result.names[class_id]

                importance = self.config.object_importance.get(
                    class_name, self.config.default_object_importance
                )

                detections.append({
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "class_name": class_name,
                    "confidence": confidence,
                    "importance": importance,
                    "detector_type": "yolo"
                })

        return detections

    def _detect_tiled(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on overlapping tiles for large (4K+) images.
        
        Tiles overlap to ensure objects at tile boundaries are detected.
        Duplicate detections from overlapping regions are filtered via NMS.
        """
        h, w = image.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.tile_overlap
        stride = tile_size - overlap

        all_detections = []

        for y_start in range(0, h, stride):
            for x_start in range(0, w, stride):
                y_end = min(y_start + tile_size, h)
                x_end = min(x_start + tile_size, w)

                tile = image[y_start:y_end, x_start:x_end]

                # Skip very small edge tiles
                if tile.shape[0] < 64 or tile.shape[1] < 64:
                    continue

                tile_detections = self._detect_single(tile)

                # Remap coordinates from tile space to full image space
                for det in tile_detections:
                    det["bbox"][0] += x_start
                    det["bbox"][1] += y_start
                    det["bbox"][2] += x_start
                    det["bbox"][3] += y_start
                    all_detections.append(det)

        # Remove duplicate detections from overlapping tiles
        all_detections = self._nms_filter(all_detections, iou_threshold=0.5)

        return all_detections

    def _nms_filter(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Non-Maximum Suppression to remove duplicate detections."""
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

        kept = []
        for det in detections:
            is_duplicate = False
            for kept_det in kept:
                if det["class_name"] != kept_det["class_name"]:
                    continue
                iou = self._compute_iou(det["bbox"], kept_det["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(det)

        return kept

    @staticmethod
    def _compute_iou(bbox1: list, bbox2: list) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0
