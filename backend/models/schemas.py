"""
Pydantic schemas for SARC API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class CompressionConfig(BaseModel):
    """Optional configuration overrides for compression."""
    w_obj: Optional[float] = Field(None, ge=0, le=1, description="Object detection weight")
    w_text: Optional[float] = Field(None, ge=0, le=1, description="Text density weight")
    w_edge: Optional[float] = Field(None, ge=0, le=1, description="Edge density weight")
    w_spatial: Optional[float] = Field(None, ge=0, le=1, description="Spatial attention weight")
    w_context: Optional[float] = Field(None, ge=0, le=1, description="Context factor weight")
    context_factor: Optional[float] = Field(None, ge=0, le=1, description="Bandwidth/device context")
    q_min: Optional[int] = Field(None, ge=1, le=100, description="Minimum JPEG quality")
    q_max: Optional[int] = Field(None, ge=1, le=100, description="Maximum JPEG quality")
    uniform_quality: Optional[int] = Field(None, ge=1, le=100, description="Uniform compression quality")
    boundary_sigma: Optional[float] = Field(None, ge=0, description="Boundary smoothing sigma")


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetection(BaseModel):
    """Single object detection result."""
    class_name: str
    confidence: float
    importance: float
    bbox: List[float]


class TextRegion(BaseModel):
    """Single text region detection result."""
    text: str
    confidence: float
    bbox: List[float]


class DetectionResults(BaseModel):
    """All detection results for an image."""
    objects: List[ObjectDetection]
    text_regions: List[TextRegion]
    total_objects: int
    total_text_regions: int


class TransmissionTime(BaseModel):
    """Transmission time at a specific bandwidth."""
    original_seconds: float
    compressed_seconds: float
    saved_seconds: float


class MetricsResult(BaseModel):
    """Quality and efficiency metrics for a compression method."""
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    file_size_bytes: int
    file_size_kb: float
    compression_ratio: float
    bandwidth_savings_percent: float
    processing_time_seconds: float
    transmission_times: Optional[Dict[str, TransmissionTime]] = None


class ImageInfo(BaseModel):
    """Information about the input image."""
    filename: str
    width: int
    height: int
    original_size_bytes: int
    original_size_kb: float


class ImportanceStats(BaseModel):
    """Statistics about the importance map."""
    mean: float
    min: float
    max: float
    std: float


class QualityStats(BaseModel):
    """Statistics about the quality map."""
    mean_quality: float
    min_quality: float
    max_quality: float


class ComparisonResult(BaseModel):
    """Comparison between SARC and uniform compression."""
    size_reduction_vs_uniform_percent: float
    psnr_difference: Optional[float] = None
    ssim_difference: Optional[float] = None


class TimingResult(BaseModel):
    """Timing breakdown for the pipeline."""
    detection_seconds: float
    ocr_seconds: float
    scoring_seconds: float
    quality_mapping_seconds: float
    sarc_compression_seconds: float
    uniform_compression_seconds: float
    evaluation_seconds: float
    total_seconds: float


class CompressionResponse(BaseModel):
    """Full response from the compression endpoint."""
    experiment_id: str
    image_info: ImageInfo
    detections: DetectionResults
    importance_stats: ImportanceStats
    quality_stats: QualityStats
    metrics: Dict[str, MetricsResult]
    comparison: ComparisonResult
    timing: TimingResult
    output_paths: Dict[str, str]
    config: Dict[str, Any]


class ExperimentSummary(BaseModel):
    """Summary of a past experiment."""
    id: int
    filename: str
    width: int
    height: int
    original_size_bytes: int
    created_at: str
    detection_count: Optional[int] = 0
    metrics: Optional[List[Dict]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
