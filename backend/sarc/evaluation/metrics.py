"""
SARC Evaluation Metrics

Comprehensive quality and efficiency metrics for comparing
SARC adaptive compression against uniform compression:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - File size and compression ratio
    - Bandwidth savings
    - Processing time
"""

import numpy as np
import cv2
from typing import Dict, Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class MetricsEvaluator:
    """Computes quality and efficiency metrics for compression evaluation."""

    def __init__(self, use_lpips: bool = True):
        self.use_lpips = use_lpips
        self._lpips_model = None

    def _load_lpips(self):
        """Lazy-load LPIPS model."""
        if self._lpips_model is None and self.use_lpips:
            try:
                import torch
                import lpips
                self._lpips_model = lpips.LPIPS(net='alex')
                if torch.cuda.is_available():
                    self._lpips_model = self._lpips_model.cuda()
            except ImportError:
                print("Warning: lpips not available. Skipping LPIPS metric.")
                self.use_lpips = False

    def compute_psnr(
        self,
        original: np.ndarray,
        compressed: np.ndarray
    ) -> float:
        """
        Compute PSNR between original and compressed images.
        Higher PSNR = better quality (less distortion).
        Typical range: 25-50 dB
        """
        return float(peak_signal_noise_ratio(original, compressed))

    def compute_ssim(
        self,
        original: np.ndarray,
        compressed: np.ndarray
    ) -> float:
        """
        Compute SSIM between original and compressed images.
        Range: [0, 1], where 1 = identical.
        Values > 0.95 are generally considered excellent quality.
        """
        # Convert to grayscale for SSIM (standard practice)
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            comp_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            comp_gray = compressed

        return float(structural_similarity(orig_gray, comp_gray))

    def compute_lpips(
        self,
        original: np.ndarray,
        compressed: np.ndarray
    ) -> Optional[float]:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity).
        
        LPIPS correlates better with human perception than PSNR/SSIM.
        Range: [0, 1], where 0 = identical (lower = better).
        """
        if not self.use_lpips:
            return None

        self._load_lpips()
        if self._lpips_model is None:
            return None

        try:
            import torch

            # Convert BGR to RGB and normalize to [-1, 1]
            orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            comp_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

            # Resize if too large (LPIPS is memory-heavy)
            max_dim = 512
            h, w = orig_rgb.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                orig_rgb = cv2.resize(orig_rgb, (new_w, new_h))
                comp_rgb = cv2.resize(comp_rgb, (new_w, new_h))

            # Convert to torch tensors: (B, C, H, W), range [-1, 1]
            orig_tensor = torch.from_numpy(orig_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            comp_tensor = torch.from_numpy(comp_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1

            if torch.cuda.is_available():
                orig_tensor = orig_tensor.cuda()
                comp_tensor = comp_tensor.cuda()

            with torch.no_grad():
                distance = self._lpips_model(orig_tensor, comp_tensor)

            return float(distance.item())
        except Exception as e:
            print(f"Warning: LPIPS computation failed: {e}")
            return None

    def evaluate(
        self,
        original: np.ndarray,
        compressed: np.ndarray,
        original_size: int,
        compressed_size: int,
        processing_time: float = 0.0
    ) -> Dict:
        """
        Compute all metrics for a compression result.
        
        Args:
            original: Original image array (BGR)
            compressed: Compressed image array (BGR)
            original_size: Original file size in bytes
            compressed_size: Compressed file size in bytes
            processing_time: Time taken for compression in seconds
        
        Returns:
            Dict with all metrics
        """
        psnr = self.compute_psnr(original, compressed)
        ssim = self.compute_ssim(original, compressed)
        lpips_val = self.compute_lpips(original, compressed)

        compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else 0
        )
        bandwidth_savings = (
            (original_size - compressed_size) / original_size * 100
            if original_size > 0 else 0
        )

        # Simulate transmission time at different bandwidths
        bandwidths = {
            "slow_3g_kbps": 400,      # 400 Kbps
            "fast_3g_kbps": 1500,     # 1.5 Mbps
            "4g_kbps": 10000,         # 10 Mbps
            "wifi_kbps": 50000,       # 50 Mbps
        }

        transmission_times = {}
        for name, bw_kbps in bandwidths.items():
            bw_bytes_per_sec = bw_kbps * 1000 / 8
            original_time = original_size / bw_bytes_per_sec
            compressed_time = compressed_size / bw_bytes_per_sec
            transmission_times[name] = {
                "original_seconds": round(original_time, 3),
                "compressed_seconds": round(compressed_time, 3),
                "saved_seconds": round(original_time - compressed_time, 3),
            }

        return {
            "psnr": round(psnr, 3),
            "ssim": round(ssim, 5),
            "lpips": round(lpips_val, 5) if lpips_val is not None else None,
            "file_size_bytes": compressed_size,
            "file_size_kb": round(compressed_size / 1024, 2),
            "compression_ratio": round(compression_ratio, 2),
            "bandwidth_savings_percent": round(bandwidth_savings, 2),
            "processing_time_seconds": round(processing_time, 3),
            "transmission_times": transmission_times,
        }
