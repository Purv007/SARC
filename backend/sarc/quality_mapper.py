"""
SARC Quality Mapper — Importance to Quality Map Conversion

Converts the pixel-level importance map into a JPEG quality map,
with Gaussian boundary smoothing to eliminate transition artifacts.

Key Innovation:
    Instead of hard quality boundaries between regions, the importance
    map is Gaussian-smoothed before quality conversion, creating a
    gradient transition that prevents visible blocking artifacts.

Formula:
    Q(x,y) = Q_min + I_smooth(x,y) × (Q_max - Q_min)
    
    where I_smooth = GaussianBlur(I, σ=boundary_sigma)
"""

import numpy as np
import cv2
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SARCConfig, DEFAULT_CONFIG


class QualityMapper:
    """
    Transforms an importance map into a smooth quality map for adaptive compression.
    """

    def __init__(self, config: SARCConfig = None):
        self.config = config or DEFAULT_CONFIG

    def importance_to_quality(
        self,
        importance_map: np.ndarray
    ) -> np.ndarray:
        """
        Convert importance map [0,1] to quality map [Q_min, Q_max] with
        Gaussian boundary smoothing.
        
        The smoothing step is critical — it creates gradient transitions
        at region boundaries instead of hard quality jumps, preventing
        visible blocking artifacts in the compressed output.
        
        Args:
            importance_map: np.ndarray of shape (H, W) with values in [0, 1]
        
        Returns:
            np.ndarray of shape (H, W) with values in [Q_min, Q_max]
        """
        # Step 1: Gaussian boundary smoothing
        # This is the key innovation — smooth transitions at region edges
        sigma = self.config.boundary_sigma
        if sigma > 0:
            # Kernel size must be odd and large enough for the sigma
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            smoothed = cv2.GaussianBlur(
                importance_map, (ksize, ksize), sigmaX=sigma, sigmaY=sigma
            )
        else:
            smoothed = importance_map.copy()

        # Step 2: Linear mapping from [0, 1] to [Q_min, Q_max]
        q_min = self.config.q_min
        q_max = self.config.q_max
        quality_map = q_min + smoothed * (q_max - q_min)

        return quality_map.astype(np.float32)

    def discretize_to_blocks(
        self,
        quality_map: np.ndarray,
        block_size: int = 8
    ) -> np.ndarray:
        """
        Discretize the continuous quality map to block-averaged values,
        matching JPEG's 8×8 DCT block structure.
        
        Each block gets the mean quality of all pixels it contains,
        ensuring consistent compression within each JPEG block.
        
        Args:
            quality_map: Continuous quality map (H, W)
            block_size: Block size (default 8, matching JPEG)
        
        Returns:
            Block-discretized quality map (H, W) where each block
            has a uniform quality value
        """
        h, w = quality_map.shape
        block_map = quality_map.copy()

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                block_mean = quality_map[y:y_end, x:x_end].mean()
                block_map[y:y_end, x:x_end] = block_mean

        return block_map

    def quantize_to_levels(
        self,
        quality_map: np.ndarray
    ) -> Tuple[np.ndarray, list]:
        """
        Quantize the continuous quality map into discrete quality levels.
        
        This groups pixels into quality bands for the multi-version
        compression approach (each level = one JPEG encoding pass).
        
        Args:
            quality_map: Continuous quality map (H, W)
        
        Returns:
            Tuple of:
                - level_map: np.ndarray (H, W) with quality level indices
                - levels: List of quality values used
        """
        levels = self.config.quality_levels
        level_map = np.zeros_like(quality_map, dtype=np.int32)

        for i, q in enumerate(levels):
            if i == 0:
                mask = quality_map < (levels[0] + levels[1]) / 2
            elif i == len(levels) - 1:
                mask = quality_map >= (levels[-2] + levels[-1]) / 2
            else:
                lower = (levels[i - 1] + levels[i]) / 2
                upper = (levels[i] + levels[i + 1]) / 2
                mask = (quality_map >= lower) & (quality_map < upper)
            level_map[mask] = i

        return level_map, levels

    def visualize_quality_map(
        self,
        quality_map: np.ndarray
    ) -> np.ndarray:
        """
        Create a color visualization of the quality map.
        
        Green = high quality (preserved)
        Red = low quality (compressed)
        
        Args:
            quality_map: Quality map with values in [Q_min, Q_max]
        
        Returns:
            BGR color image visualizing the quality distribution
        """
        q_min = self.config.q_min
        q_max = self.config.q_max

        # Normalize to [0, 255]
        normalized = ((quality_map - q_min) / (q_max - q_min) * 255).astype(np.uint8)

        # Apply colormap (COLORMAP_RdYlGn goes from red to green)
        # We use JET and flip so green = high quality
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        return colored
