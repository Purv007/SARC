"""
SARC Importance Scorer — The Core Innovation

Computes a pixel-level importance map by fusing 5 heterogeneous signals:
    1. Object Detection (YOLO bounding boxes → class-weighted masks)
    2. Text Density (OCR regions → dilated text importance mask)
    3. Edge Density (Canny → box-filtered local edge density)
    4. Spatial Attention (center Gaussian + rule-of-thirds)
    5. Context Factor (global bandwidth/device modifier)

Formula:
    I(x,y) = clamp( w₁·S_obj + w₂·S_text + w₃·S_edge + w₄·S_spatial + w₅·S_context, 0, 1 )
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SARCConfig, DEFAULT_CONFIG


class ImportanceScorer:
    """
    Fuses multiple signals into a single pixel-level importance map.
    
    This is the novel contribution of the SARC algorithm — no prior system
    combines all 5 signals into one closed-form scoring function for
    general-purpose image compression.
    """

    def __init__(self, config: SARCConfig = None):
        self.config = config or DEFAULT_CONFIG

    def compute_object_score(
        self,
        image_shape: Tuple[int, int],
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Convert YOLO detections into a per-pixel object importance map.
        
        Each pixel inside a bounding box gets the importance score of
        the detected object class. Overlapping detections use max().
        
        Args:
            image_shape: (height, width) of the image
            detections: List of dicts with keys: 
                        'bbox' (x1,y1,x2,y2), 'class_name', 'confidence'
        
        Returns:
            np.ndarray of shape (H, W) with values in [0, 1]
        """
        h, w = image_shape[:2]
        score_map = np.zeros((h, w), dtype=np.float32)

        for det in detections:
            class_name = det.get("class_name", "unknown")
            confidence = det.get("confidence", 1.0)
            bbox = det["bbox"]  # (x1, y1, x2, y2)

            # Look up class importance
            base_importance = self.config.object_importance.get(
                class_name, self.config.default_object_importance
            )

            # Scale by detection confidence
            importance = base_importance * confidence

            # Apply to bounding box region (clamp to image bounds)
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            # Use max to handle overlapping detections
            score_map[y1:y2, x1:x2] = np.maximum(
                score_map[y1:y2, x1:x2], importance
            )

        return score_map

    def compute_text_score(
        self,
        image_shape: Tuple[int, int],
        text_regions: List[Dict]
    ) -> np.ndarray:
        """
        Convert OCR text regions into a dilated text importance mask.
        
        Text regions get the highest importance because text readability
        is critical for information preservation.
        
        Args:
            image_shape: (height, width) of the image
            text_regions: List of dicts with key 'bbox' (x1,y1,x2,y2)
        
        Returns:
            np.ndarray of shape (H, W) with values in [0, 1]
        """
        h, w = image_shape[:2]
        text_mask = np.zeros((h, w), dtype=np.float32)

        for region in text_regions:
            bbox = region["bbox"]  # (x1, y1, x2, y2)
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            text_mask[y1:y2, x1:x2] = self.config.text_importance

        # Dilate the text mask to protect surrounding context
        if text_regions:
            kernel_size = self.config.text_dilation_kernel
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)
            # Smooth the dilated edges
            text_mask = cv2.GaussianBlur(text_mask, (0, 0), sigmaX=5.0)
            text_mask = np.clip(text_mask, 0, 1)

        return text_mask

    def compute_edge_score(self, image: np.ndarray) -> np.ndarray:
        """
        Compute local edge density as a proxy for structural detail.
        
        High edge density regions contain fine details (graphs, diagrams,
        textures) that need preservation. Low edge density regions are
        flat/smooth (sky, walls) and can be compressed more.
        
        Args:
            image: BGR image array
        
        Returns:
            np.ndarray of shape (H, W) with values in [0, 1]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(
            gray,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold
        ).astype(np.float32) / 255.0

        # Compute local edge density using box filter
        kernel_size = self.config.edge_density_kernel_size
        density = cv2.boxFilter(
            edges, ddepth=-1,
            ksize=(kernel_size, kernel_size),
            normalize=True
        )

        # Normalize to [0, 1] using adaptive scaling
        max_density = density.max()
        if max_density > 0:
            density = density / max_density

        return density.astype(np.float32)

    def compute_spatial_score(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate a spatial attention map based on viewing patterns.
        
        Combines:
        1. Center-weighted Gaussian (humans focus on image center)
        2. Rule-of-thirds boost (key compositional points)
        
        Args:
            image_shape: (height, width) of the image
        
        Returns:
            np.ndarray of shape (H, W) with values in [0, 1]
        """
        h, w = image_shape[:2]

        # 1. Center-weighted Gaussian
        cy, cx = h / 2, w / 2
        sigma_y = h * self.config.spatial_center_sigma
        sigma_x = w * self.config.spatial_center_sigma

        y_coords = np.arange(h).reshape(-1, 1).astype(np.float32)
        x_coords = np.arange(w).reshape(1, -1).astype(np.float32)

        gaussian = np.exp(
            -((y_coords - cy) ** 2 / (2 * sigma_y ** 2) +
              (x_coords - cx) ** 2 / (2 * sigma_x ** 2))
        )

        # 2. Rule-of-thirds boost
        thirds_map = np.zeros((h, w), dtype=np.float32)
        third_points = [
            (h // 3, w // 3), (h // 3, 2 * w // 3),
            (2 * h // 3, w // 3), (2 * h // 3, 2 * w // 3)
        ]
        boost_radius_y = h // 8
        boost_radius_x = w // 8

        for py, px in third_points:
            y_mask = np.abs(y_coords - py) < boost_radius_y
            x_mask = np.abs(x_coords - px) < boost_radius_x
            thirds_map += (y_mask & x_mask).astype(np.float32) * self.config.spatial_thirds_boost

        # Combine
        spatial = gaussian + thirds_map
        spatial = spatial / spatial.max()  # Normalize to [0, 1]

        return spatial.astype(np.float32)

    def compute_importance_map(
        self,
        image: np.ndarray,
        object_detections: List[Dict],
        text_regions: List[Dict],
        context_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute the final pixel-level importance map by fusing all 5 signals.
        
        This is the SARC formula:
            I(x,y) = clamp( w₁·S_obj + w₂·S_text + w₃·S_edge 
                          + w₄·S_spatial + w₅·S_context, 0, 1 )
        
        Args:
            image: BGR image array (H, W, 3)
            object_detections: YOLO detection results
            text_regions: OCR text region results
            context_factor: Override for bandwidth/device context (0-1)
        
        Returns:
            np.ndarray of shape (H, W) with values in [0, 1]
        """
        h, w = image.shape[:2]
        ctx = context_factor if context_factor is not None else self.config.context_factor

        # Compute individual signal maps
        s_obj = self.compute_object_score((h, w), object_detections)
        s_text = self.compute_text_score((h, w), text_regions)
        s_edge = self.compute_edge_score(image)
        s_spatial = self.compute_spatial_score((h, w))
        s_context = np.full((h, w), ctx, dtype=np.float32)

        # Weighted fusion — the SARC importance formula
        importance_map = (
            self.config.w_obj * s_obj +
            self.config.w_text * s_text +
            self.config.w_edge * s_edge +
            self.config.w_spatial * s_spatial +
            self.config.w_context * s_context
        )

        # Clamp to [0, 1]
        importance_map = np.clip(importance_map, 0.0, 1.0)

        return importance_map

    def generate_heatmap_overlay(
        self,
        image: np.ndarray,
        importance_map: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create a visual heatmap overlay showing importance scores on the image.
        
        Uses JET colormap: Blue (low importance) → Red (high importance)
        
        Args:
            image: Original BGR image
            importance_map: Importance map [0, 1]
            alpha: Overlay transparency (0 = fully transparent, 1 = fully opaque)
        
        Returns:
            BGR image with heatmap overlay
        """
        # Convert importance map to 8-bit for colormap
        heatmap_gray = (importance_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

        # Blend with original image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

        return overlay
