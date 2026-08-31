"""
Edge Analyzer — Canny Edge Density Computation

Computes a local edge density map using Canny edge detection
followed by box filtering. High edge density indicates regions
with fine structural detail (graphs, diagrams, textures) that
should receive higher compression quality.
"""

import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SARCConfig, DEFAULT_CONFIG


class EdgeAnalyzer:
    """Computes local edge density as a structural detail proxy."""

    def __init__(self, config: SARCConfig = None):
        self.config = config or DEFAULT_CONFIG

    def analyze(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the edge density map for the image.
        
        Pipeline:
        1. Convert to grayscale
        2. Apply Canny edge detection
        3. Compute local density via box filter
        4. Normalize to [0, 1]
        
        Args:
            image: BGR image array
        
        Returns:
            np.ndarray of shape (H, W) with values in [0, 1]
            representing local edge density
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold
        ).astype(np.float32) / 255.0

        # Compute local edge density using box filter
        kernel_size = self.config.edge_density_kernel_size
        density = cv2.boxFilter(
            edges,
            ddepth=-1,
            ksize=(kernel_size, kernel_size),
            normalize=True
        )

        # Normalize to [0, 1] with adaptive scaling
        max_density = density.max()
        if max_density > 0:
            density = density / max_density

        return density.astype(np.float32)

    def get_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Return the raw Canny edge map (for visualization)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(
            blurred,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold
        )
        return edges
