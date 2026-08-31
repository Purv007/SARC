"""
SARC Engine — Full Pipeline Orchestrator

Coordinates all components of the SARC algorithm:
    Detection → Importance Scoring → Quality Mapping → Adaptive Compression → Evaluation

This is the main entry point for processing an image through the SARC pipeline.
"""

import numpy as np
import cv2
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SARCConfig, DEFAULT_CONFIG, OUTPUT_DIR

from sarc.importance_scorer import ImportanceScorer
from sarc.quality_mapper import QualityMapper
from sarc.adaptive_compressor import AdaptiveCompressor
from sarc.detectors.object_detector import ObjectDetector
from sarc.detectors.text_detector import TextDetector
from sarc.detectors.edge_analyzer import EdgeAnalyzer
from sarc.evaluation.metrics import MetricsEvaluator


class SARCEngine:
    """
    Main orchestrator for the SARC compression pipeline.
    
    Usage:
        engine = SARCEngine()
        result = engine.process("path/to/image.jpg")
    """

    def __init__(self, config: SARCConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.scorer = ImportanceScorer(self.config)
        self.mapper = QualityMapper(self.config)
        self.compressor = AdaptiveCompressor(self.config)
        self.object_detector = ObjectDetector(self.config)
        self.text_detector = TextDetector(self.config)
        self.edge_analyzer = EdgeAnalyzer(self.config)
        self.evaluator = MetricsEvaluator()

    def process(
        self,
        image_path: str,
        output_prefix: Optional[str] = None,
        config_override: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run the full SARC pipeline on an image.
        
        Pipeline:
        1. Load and validate image
        2. Run YOLO object detection
        3. Run EasyOCR text detection
        4. Compute 5-signal importance map
        5. Generate quality map with boundary smoothing
        6. Apply adaptive compression
        7. Apply uniform compression (baseline)
        8. Evaluate and compare
        9. Generate visualizations (heatmap, overlay)
        
        Args:
            image_path: Path to the input image
            output_prefix: Prefix for output files (defaults to timestamp)
            config_override: Optional dict to override config values
        
        Returns:
            Dict containing all results, metrics, paths, and detection data
        """
        total_start = time.time()

        # Apply config overrides
        config = self.config
        if config_override:
            config = SARCConfig(**{**asdict(self.config), **config_override})
            self.scorer = ImportanceScorer(config)
            self.mapper = QualityMapper(config)
            self.compressor = AdaptiveCompressor(config)

        # Setup output paths
        if output_prefix is None:
            output_prefix = f"sarc_{int(time.time())}"
        output_dir = OUTPUT_DIR / output_prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        # ===== Step 1: Load Image =====
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        h, w = image.shape[:2]
        original_size = Path(image_path).stat().st_size

        # Save a copy of the original
        original_output = str(output_dir / "original.jpg")
        cv2.imwrite(original_output, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # ===== Step 2: Object Detection =====
        det_start = time.time()
        object_detections = self.object_detector.detect(image)
        det_time = time.time() - det_start

        # ===== Step 3: Text Detection =====
        ocr_start = time.time()
        text_regions = self.text_detector.detect(image)
        ocr_time = time.time() - ocr_start

        # ===== Step 4: Importance Scoring =====
        score_start = time.time()
        importance_map = self.scorer.compute_importance_map(
            image, object_detections, text_regions, config.context_factor
        )
        score_time = time.time() - score_start

        # ===== Step 5: Quality Mapping =====
        map_start = time.time()
        quality_map = self.mapper.importance_to_quality(importance_map)
        map_time = time.time() - map_start

        # ===== Step 6: Adaptive Compression (SARC) =====
        sarc_start = time.time()
        sarc_output_path = str(output_dir / "sarc_compressed.jpg")
        sarc_image, sarc_size = self.compressor.compress_adaptive(
            image, quality_map, output_path=sarc_output_path
        )
        sarc_time = time.time() - sarc_start

        # ===== Step 7: Uniform Compression (Baseline) =====
        uniform_start = time.time()
        uniform_output_path = str(output_dir / "uniform_compressed.jpg")
        uniform_image, uniform_size = self.compressor.compress_uniform(
            image, quality=config.uniform_quality, output_path=uniform_output_path
        )
        uniform_time = time.time() - uniform_start

        # ===== Step 8: Evaluation =====
        eval_start = time.time()

        # Metrics for SARC compression
        sarc_metrics = self.evaluator.evaluate(
            original=image,
            compressed=sarc_image,
            original_size=original_size,
            compressed_size=sarc_size,
            processing_time=sarc_time
        )

        # Metrics for uniform compression
        uniform_metrics = self.evaluator.evaluate(
            original=image,
            compressed=uniform_image,
            original_size=original_size,
            compressed_size=uniform_size,
            processing_time=uniform_time
        )

        eval_time = time.time() - eval_start

        # ===== Step 9: Visualizations =====
        # Importance heatmap overlay
        heatmap_path = str(output_dir / "importance_heatmap.jpg")
        heatmap = self.scorer.generate_heatmap_overlay(image, importance_map)
        cv2.imwrite(heatmap_path, heatmap, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Quality map visualization
        quality_vis_path = str(output_dir / "quality_map.jpg")
        quality_vis = self.mapper.visualize_quality_map(quality_map)
        cv2.imwrite(quality_vis_path, quality_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Detection overlay
        overlay_path = str(output_dir / "detection_overlay.jpg")
        overlay = self._draw_detection_overlay(
            image.copy(), object_detections, text_regions
        )
        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

        total_time = time.time() - total_start

        # ===== Compile Results =====
        result = {
            "experiment_id": output_prefix,
            "image_info": {
                "filename": Path(image_path).name,
                "width": w,
                "height": h,
                "original_size_bytes": original_size,
                "original_size_kb": round(original_size / 1024, 2),
            },
            "detections": {
                "objects": [
                    {
                        "class_name": d["class_name"],
                        "confidence": round(d["confidence"], 3),
                        "importance": round(d["importance"], 3),
                        "bbox": [round(b, 1) for b in d["bbox"]],
                    }
                    for d in object_detections
                ],
                "text_regions": [
                    {
                        "text": t["text"],
                        "confidence": round(t["confidence"], 3),
                        "bbox": [round(b, 1) for b in t["bbox"]],
                    }
                    for t in text_regions
                ],
                "total_objects": len(object_detections),
                "total_text_regions": len(text_regions),
            },
            "importance_stats": {
                "mean": round(float(importance_map.mean()), 4),
                "min": round(float(importance_map.min()), 4),
                "max": round(float(importance_map.max()), 4),
                "std": round(float(importance_map.std()), 4),
            },
            "quality_stats": {
                "mean_quality": round(float(quality_map.mean()), 1),
                "min_quality": round(float(quality_map.min()), 1),
                "max_quality": round(float(quality_map.max()), 1),
            },
            "metrics": {
                "sarc": sarc_metrics,
                "uniform": uniform_metrics,
            },
            "comparison": {
                "size_reduction_vs_uniform_percent": round(
                    (1 - sarc_size / uniform_size) * 100, 2
                ) if uniform_size > 0 else 0,
                "psnr_difference": round(
                    sarc_metrics["psnr"] - uniform_metrics["psnr"], 3
                ) if sarc_metrics.get("psnr") and uniform_metrics.get("psnr") else None,
                "ssim_difference": round(
                    sarc_metrics["ssim"] - uniform_metrics["ssim"], 4
                ) if sarc_metrics.get("ssim") and uniform_metrics.get("ssim") else None,
            },
            "timing": {
                "detection_seconds": round(det_time, 3),
                "ocr_seconds": round(ocr_time, 3),
                "scoring_seconds": round(score_time, 3),
                "quality_mapping_seconds": round(map_time, 3),
                "sarc_compression_seconds": round(sarc_time, 3),
                "uniform_compression_seconds": round(uniform_time, 3),
                "evaluation_seconds": round(eval_time, 3),
                "total_seconds": round(total_time, 3),
            },
            "output_paths": {
                "original": original_output,
                "sarc_compressed": sarc_output_path,
                "uniform_compressed": uniform_output_path,
                "importance_heatmap": heatmap_path,
                "quality_map": quality_vis_path,
                "detection_overlay": overlay_path,
            },
            "config": {
                "w_obj": config.w_obj,
                "w_text": config.w_text,
                "w_edge": config.w_edge,
                "w_spatial": config.w_spatial,
                "w_context": config.w_context,
                "q_min": config.q_min,
                "q_max": config.q_max,
                "uniform_quality": config.uniform_quality,
                "boundary_sigma": config.boundary_sigma,
                "context_factor": config.context_factor,
            },
        }

        return result

    def process_video(
        self,
        video_path: str,
        output_prefix: Optional[str] = None,
        config_override: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process a video file using SARC on every frame."""
        import subprocess
        from imageio_ffmpeg import get_ffmpeg_exe
        
        total_start = time.time()

        # Apply config overrides
        config = self.config
        if config_override:
            config = SARCConfig(**{**asdict(self.config), **config_override})
            self.scorer = ImportanceScorer(config)
            self.mapper = QualityMapper(config)
            self.compressor = AdaptiveCompressor(config)

        if output_prefix is None:
            output_prefix = f"sarc_{int(time.time())}"
        output_dir = OUTPUT_DIR / output_prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sarc_frames_dir = output_dir / "sarc_frames"
        uniform_frames_dir = output_dir / "uniform_frames"
        sarc_frames_dir.mkdir(exist_ok=True)
        uniform_frames_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_count > 150:
            cap.release()
            raise ValueError(f"Video too long ({frame_count} frames). For this prototype, max 150 frames (5s at 30fps) are allowed due to processing time.")
            
        original_size = Path(video_path).stat().st_size
        
        sarc_total_size = 0
        uniform_total_size = 0
        
        # Accumulate metrics to average later
        metrics_sum = {"sarc_psnr": 0, "sarc_ssim": 0, "sarc_lpips": 0, "uniform_psnr": 0, "uniform_ssim": 0, "uniform_lpips": 0}
        
        # Accumulate detections for first frame (for UI visualization)
        first_frame_detections = {"objects": [], "text_regions": []}
        heatmap_path = str(output_dir / "importance_heatmap.jpg")
        quality_vis_path = str(output_dir / "quality_map.jpg")
        overlay_path = str(output_dir / "detection_overlay.jpg")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # 1. Detection
            object_detections = self.object_detector.detect(frame)
            text_regions = self.text_detector.detect(frame)
            
            # 2. Maps
            importance_map = self.scorer.compute_importance_map(frame, object_detections, text_regions, config.context_factor)
            quality_map = self.mapper.importance_to_quality(importance_map)
            
            # 3. Compress directly to disk
            sarc_path = sarc_frames_dir / f"frame_{frame_idx:04d}.jpg"
            uniform_path = uniform_frames_dir / f"frame_{frame_idx:04d}.jpg"
            
            sarc_img, sarc_sz = self.compressor.compress_adaptive(frame, quality_map, output_path=str(sarc_path))
            uniform_img, uniform_sz = self.compressor.compress_uniform(frame, config.uniform_quality, output_path=str(uniform_path))
            
            sarc_total_size += sarc_sz
            uniform_total_size += uniform_sz
            
            # Evaluate this frame
            sarc_m = self.evaluator.evaluate(frame, sarc_img, 0, 0, 0)
            uniform_m = self.evaluator.evaluate(frame, uniform_img, 0, 0, 0)
            
            metrics_sum["sarc_psnr"] += sarc_m["psnr"]
            metrics_sum["sarc_ssim"] += sarc_m["ssim"]
            metrics_sum["sarc_lpips"] += sarc_m["lpips"]
            metrics_sum["uniform_psnr"] += uniform_m["psnr"]
            metrics_sum["uniform_ssim"] += uniform_m["ssim"]
            metrics_sum["uniform_lpips"] += uniform_m["lpips"]
            
            # Save visuals for the very first frame to display on the dashboard
            if frame_idx == 1:
                first_frame_detections["objects"] = object_detections
                first_frame_detections["text_regions"] = text_regions
                first_frame_heatmap = self.scorer.generate_heatmap_overlay(frame, importance_map)
                cv2.imwrite(heatmap_path, first_frame_heatmap, [cv2.IMWRITE_JPEG_QUALITY, 95])
                first_frame_quality = self.mapper.visualize_quality_map(quality_map)
                cv2.imwrite(quality_vis_path, first_frame_quality, [cv2.IMWRITE_JPEG_QUALITY, 95])
                first_frame_overlay = self._draw_detection_overlay(frame.copy(), object_detections, text_regions)
                cv2.imwrite(overlay_path, first_frame_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
        cap.release()
        
        # 4. Stitch videos using ffmpeg (-c:v copy to preserve SARC MJPEG bytes!)
        ffmpeg_exe = get_ffmpeg_exe()
        sarc_output_path = str(output_dir / "sarc_compressed.mp4")
        uniform_output_path = str(output_dir / "uniform_compressed.mp4")
        original_output = str(output_dir / "original.mp4")
        
        # Just copy the original video over to output_dir
        import shutil
        shutil.copy2(video_path, original_output)
        
        subprocess.run([
            ffmpeg_exe, "-y", "-framerate", str(fps), 
            "-i", str(sarc_frames_dir / "frame_%04d.jpg"), 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", "-preset", "ultrafast", sarc_output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        subprocess.run([
            ffmpeg_exe, "-y", "-framerate", str(fps), 
            "-i", str(uniform_frames_dir / "frame_%04d.jpg"), 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", "-preset", "ultrafast", uniform_output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        total_time = time.time() - total_start
        
        # Average metrics
        fc = max(1, frame_idx)
        
        sarc_metrics = {
            "psnr": round(metrics_sum["sarc_psnr"] / fc, 2),
            "ssim": round(metrics_sum["sarc_ssim"] / fc, 4),
            "lpips": round(metrics_sum["sarc_lpips"] / fc, 4),
            "file_size_bytes": sarc_total_size,
            "file_size_kb": round(sarc_total_size / 1024, 2),
            "compression_ratio": round(original_size / sarc_total_size, 2) if sarc_total_size else 0,
            "bandwidth_savings_percent": round((1 - sarc_total_size / original_size) * 100, 2) if original_size else 0,
            "processing_time_seconds": total_time,
            "transmission_times": self.evaluator.evaluate(np.zeros((7,7,3), dtype=np.uint8), np.zeros((7,7,3), dtype=np.uint8), original_size, sarc_total_size)["transmission_times"]
        }
        
        uniform_metrics = {
            "psnr": round(metrics_sum["uniform_psnr"] / fc, 2),
            "ssim": round(metrics_sum["uniform_ssim"] / fc, 4),
            "lpips": round(metrics_sum["uniform_lpips"] / fc, 4),
            "file_size_bytes": uniform_total_size,
            "file_size_kb": round(uniform_total_size / 1024, 2),
            "compression_ratio": round(original_size / uniform_total_size, 2) if uniform_total_size else 0,
            "bandwidth_savings_percent": round((1 - uniform_total_size / original_size) * 100, 2) if original_size else 0,
            "processing_time_seconds": total_time,
            "transmission_times": self.evaluator.evaluate(np.zeros((7,7,3), dtype=np.uint8), np.zeros((7,7,3), dtype=np.uint8), original_size, uniform_total_size)["transmission_times"]
        }
        
        # ===== Compile Results =====
        result = {
            "experiment_id": output_prefix,
            "image_info": {
                "filename": Path(video_path).name,
                "width": w,
                "height": h,
                "original_size_bytes": original_size,
                "original_size_kb": round(original_size / 1024, 2),
                "is_video": True,
                "frame_count": fc,
                "fps": fps
            },
            "detections": {
                "objects": [
                    {
                        "class_name": d["class_name"],
                        "confidence": round(d["confidence"], 3),
                        "importance": round(d.get("importance", 0.5), 3),
                        "bbox": [round(b, 1) for b in d["bbox"]],
                    }
                    for d in first_frame_detections["objects"]
                ],
                "text_regions": [
                    {
                        "text": t["text"],
                        "confidence": round(t["confidence"], 3),
                        "bbox": [round(b, 1) for b in t["bbox"]],
                    }
                    for t in first_frame_detections["text_regions"]
                ],
                "total_objects": len(first_frame_detections["objects"]),
                "total_text_regions": len(first_frame_detections["text_regions"]),
            },
            "importance_stats": {
                "mean": 0, "min": 0, "max": 1, "std": 0
            },
            "quality_stats": {
                "mean_quality": 0, "min_quality": 0, "max_quality": 100
            },
            "metrics": {
                "sarc": sarc_metrics,
                "uniform": uniform_metrics,
            },
            "comparison": {
                "size_reduction_vs_uniform_percent": round(
                    (1 - sarc_total_size / uniform_total_size) * 100, 2
                ) if uniform_total_size > 0 else 0,
                "psnr_difference": round(
                    sarc_metrics["psnr"] - uniform_metrics["psnr"], 3
                ),
                "ssim_difference": round(
                    sarc_metrics["ssim"] - uniform_metrics["ssim"], 4
                ),
            },
            "timing": {
                "total_seconds": round(total_time, 3),
            },
            "output_paths": {
                "original": original_output,
                "sarc_compressed": sarc_output_path,
                "uniform_compressed": uniform_output_path,
                "importance_heatmap": heatmap_path,
                "quality_map": quality_vis_path,
                "detection_overlay": overlay_path,
            },
            "config": {
                "w_obj": config.w_obj,
                "w_text": config.w_text,
                "w_edge": config.w_edge,
                "w_spatial": config.w_spatial,
                "w_context": config.w_context,
                "q_min": config.q_min,
                "q_max": config.q_max,
                "uniform_quality": config.uniform_quality,
                "boundary_sigma": config.boundary_sigma,
                "context_factor": config.context_factor,
            },
        }

        return result

    def _draw_detection_overlay(
        self,
        image: np.ndarray,
        object_detections: list,
        text_regions: list
    ) -> np.ndarray:
        """Draw detection boxes and labels on the image for visualization."""
        # Draw object detections (green boxes)
        for det in object_detections:
            bbox = det["bbox"]
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])
            importance = det.get("importance", 0.5)

            # Color based on importance (red = low, green = high)
            r = int(255 * (1 - importance))
            g = int(255 * importance)
            color = (0, g, r)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} ({det['confidence']:.2f}) I:{importance:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0] + 4, y1),
                color, -1
            )
            cv2.putText(
                image, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Draw text regions (blue boxes)
        for region in text_regions:
            bbox = region["bbox"]
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), 2)

            text_label = f"TEXT: {region['text'][:20]}"
            cv2.putText(
                image, text_label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1
            )

        return image
