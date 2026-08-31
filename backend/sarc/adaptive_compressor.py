"""
SARC Adaptive Compressor — Region-Aware Multi-Quality Compression

Instead of encoding each 8×8 block individually (which would require
custom JPEG internals), this uses a multi-quality blending approach:

1. Compress the full image at multiple quality levels (e.g., Q=20, Q=40, Q=60, Q=80, Q=95)
2. Use the quality map to alpha-blend between adjacent quality versions
3. The result smoothly varies in quality across the image

This approach:
    - Requires NO custom JPEG encoder
    - Naturally avoids boundary artifacts (smooth blending)
    - Is fast (just N JPEG encodes + vectorized blending)
    - Produces genuinely different compression per region
"""

import numpy as np
import cv2
import io
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SARCConfig, DEFAULT_CONFIG


class AdaptiveCompressor:
    """
    Performs region-aware adaptive JPEG compression using multi-quality blending.
    """

    def __init__(self, config: SARCConfig = None):
        self.config = config or DEFAULT_CONFIG

    def compress_uniform(
        self,
        image: np.ndarray,
        quality: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Standard uniform JPEG compression (baseline for comparison).
        
        Args:
            image: BGR image array
            quality: JPEG quality (1-100). Uses config default if None.
            output_path: Optional path to save the compressed image
        
        Returns:
            Tuple of (compressed_image_array, file_size_bytes)
        """
        q = quality or self.config.uniform_quality

        # Encode to JPEG in memory
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, q]
        success, encoded = cv2.imencode(".jpg", image, encode_params)
        if not success:
            raise RuntimeError("Failed to encode JPEG")

        file_size = len(encoded)

        # Decode back to get the compressed image (with compression artifacts)
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Save if path provided
        if output_path:
            with open(output_path, "wb") as f:
                f.write(encoded.tobytes())

        return compressed, file_size

    def compress_adaptive(
        self,
        image: np.ndarray,
        quality_map: np.ndarray,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        SARC adaptive compression using multi-quality blending.
        
        The key technique:
        1. Compress the full image at each quality level
        2. For each pixel, interpolate between the two nearest quality versions
           based on the quality map value
        3. Encode the blended result as a single JPEG
        
        Args:
            image: BGR image array (H, W, 3)
            quality_map: Per-pixel quality map (H, W) with values in [Q_min, Q_max]
            output_path: Optional path to save the compressed image
        
        Returns:
            Tuple of (compressed_image_array, file_size_bytes)
        """
        levels = sorted(self.config.quality_levels)
        h, w = image.shape[:2]

        # Step 1: Compress at each quality level
        compressed_versions = []
        for q in levels:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, q]
            _, encoded = cv2.imencode(".jpg", image, encode_params)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            compressed_versions.append(decoded.astype(np.float32))

        # Step 2: Multi-quality blending
        # For each pixel, find the two bracketing quality levels and interpolate
        result = np.zeros_like(image, dtype=np.float32)

        for i in range(len(levels) - 1):
            q_low = levels[i]
            q_high = levels[i + 1]

            # Mask for pixels in this quality band
            if i == 0:
                mask = quality_map < q_high
            elif i == len(levels) - 2:
                mask = quality_map >= q_low
            else:
                mask = (quality_map >= q_low) & (quality_map < q_high)

            if not mask.any():
                continue

            # Compute interpolation alpha within this band
            band_range = q_high - q_low
            if band_range > 0:
                alpha = (quality_map - q_low) / band_range
            else:
                alpha = np.zeros_like(quality_map)
            alpha = np.clip(alpha, 0, 1)

            # Expand alpha to 3 channels for blending
            alpha_3ch = alpha[:, :, np.newaxis]
            mask_3ch = mask[:, :, np.newaxis]

            # Interpolate between adjacent quality versions
            blended = (
                (1 - alpha_3ch) * compressed_versions[i] +
                alpha_3ch * compressed_versions[i + 1]
            )

            # Apply only to pixels in this band
            result = np.where(mask_3ch, blended, result)

        # Handle pixels at or above the highest quality level
        top_mask = quality_map >= levels[-1]
        if top_mask.any():
            top_mask_3ch = top_mask[:, :, np.newaxis]
            result = np.where(top_mask_3ch, compressed_versions[-1], result)

        # Handle pixels below the lowest quality level
        bottom_mask = quality_map <= levels[0]
        if bottom_mask.any():
            bottom_mask_3ch = bottom_mask[:, :, np.newaxis]
            result = np.where(bottom_mask_3ch, compressed_versions[0], result)

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Step 3: Encode the final blended image
        # Use a quality level that matches the average quality for file size
        avg_quality = int(quality_map.mean())
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, avg_quality]
        success, encoded = cv2.imencode(".jpg", result, encode_params)
        if not success:
            raise RuntimeError("Failed to encode final SARC image")

        file_size = len(encoded)

        # Save if path provided
        if output_path:
            with open(output_path, "wb") as f:
                f.write(encoded.tobytes())

        # Decode to return the actual compressed version (what the file contains)
        final = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return final, file_size

    def get_compression_stats(
        self,
        original_size: int,
        compressed_size: int
    ) -> dict:
        """
        Calculate compression statistics.
        
        Args:
            original_size: Original file size in bytes
            compressed_size: Compressed file size in bytes
        
        Returns:
            Dict with compression_ratio, savings_percent, etc.
        """
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        savings = ((original_size - compressed_size) / original_size * 100
                   if original_size > 0 else 0)

        return {
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "original_size_kb": round(original_size / 1024, 2),
            "compressed_size_kb": round(compressed_size / 1024, 2),
            "compression_ratio": round(ratio, 2),
            "savings_percent": round(savings, 2),
        }
