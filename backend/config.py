"""
SARC Configuration — Semantically-Aware Regional Compression

All algorithm parameters, model paths, and system settings are centralized here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# Base directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = BASE_DIR / "sarc.db"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SARCConfig:
    """
    Configuration for the SARC importance scoring and compression pipeline.
    
    The importance score for each pixel is:
        I(x,y) = clamp( w_obj * S_obj + w_text * S_text + w_edge * S_edge 
                       + w_spatial * S_spatial + w_context * S_context, 0, 1 )
    """

    # --- Signal weights (must sum to ~1.0 for normalized output) ---
    w_obj: float = 0.35       # Object detection weight
    w_text: float = 0.30      # Text density weight
    w_edge: float = 0.15      # Edge density weight
    w_spatial: float = 0.10   # Spatial attention weight
    w_context: float = 0.10   # Context factor weight

    # --- Context factor (global modifier) ---
    # 1.0 = normal, 0.5 = low bandwidth (only critical content preserved)
    context_factor: float = 1.0

    # --- Quality mapping ---
    q_min: int = 20           # Minimum JPEG quality (for lowest importance regions)
    q_max: int = 95           # Maximum JPEG quality (for highest importance regions)
    boundary_sigma: float = 15.0  # Gaussian sigma for boundary smoothing (pixels)

    # --- Compression settings ---
    uniform_quality: int = 75  # Quality for uniform/baseline JPEG compression
    quality_levels: list = field(default_factory=lambda: [20, 40, 60, 80, 95])

    # --- Object importance lookup ---
    # Maps YOLO class names to importance scores [0, 1]
    object_importance: Dict[str, float] = field(default_factory=lambda: {
        # Critical content — highest importance
        "book": 0.95,
        "clock": 0.85,
        "laptop": 0.90,
        "cell phone": 0.90,
        "tv": 0.85,
        "keyboard": 0.80,
        
        # People — high importance
        "person": 0.90,
        
        # Vehicles — medium-high importance
        "car": 0.65,
        "truck": 0.65,
        "bus": 0.65,
        "motorcycle": 0.65,
        "bicycle": 0.60,
        "airplane": 0.70,
        "boat": 0.60,
        "train": 0.65,
        
        # Animals — medium importance
        "cat": 0.70,
        "dog": 0.70,
        "horse": 0.65,
        "bird": 0.55,
        "sheep": 0.50,
        "cow": 0.50,
        "elephant": 0.60,
        "bear": 0.60,
        "zebra": 0.55,
        "giraffe": 0.55,
        
        # Furniture/objects — medium importance
        "chair": 0.40,
        "couch": 0.40,
        "dining table": 0.40,
        "bed": 0.40,
        "toilet": 0.35,
        "sink": 0.35,
        "refrigerator": 0.40,
        "oven": 0.35,
        "microwave": 0.35,
        
        # Food — medium-low importance
        "banana": 0.45,
        "apple": 0.45,
        "sandwich": 0.45,
        "orange": 0.45,
        "pizza": 0.50,
        "cake": 0.50,
        "donut": 0.45,
        "hot dog": 0.45,
        "bowl": 0.35,
        "cup": 0.40,
        "fork": 0.30,
        "knife": 0.30,
        "spoon": 0.30,
        "bottle": 0.40,
        "wine glass": 0.40,
        
        # Misc objects — low-medium importance
        "umbrella": 0.35,
        "handbag": 0.40,
        "suitcase": 0.45,
        "tie": 0.50,
        "backpack": 0.40,
        "sports ball": 0.50,
        "kite": 0.40,
        "baseball bat": 0.45,
        "baseball glove": 0.45,
        "skateboard": 0.45,
        "surfboard": 0.45,
        "tennis racket": 0.45,
        "frisbee": 0.40,
        "skis": 0.45,
        "snowboard": 0.45,
        
        # Decorative/background — low importance
        "potted plant": 0.30,
        "vase": 0.30,
        "scissors": 0.35,
        "teddy bear": 0.40,
        "hair drier": 0.30,
        "toothbrush": 0.30,
        
        # Traffic — low importance
        "traffic light": 0.35,
        "fire hydrant": 0.30,
        "stop sign": 0.50,
        "parking meter": 0.25,
        "bench": 0.25,
    })

    # Default importance for unknown classes
    default_object_importance: float = 0.50

    # --- Edge analysis ---
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    edge_density_kernel_size: int = 31  # Box filter kernel for density computation

    # --- Spatial attention ---
    spatial_center_sigma: float = 0.4   # Gaussian sigma as fraction of image size
    spatial_thirds_boost: float = 0.15  # Bonus for rule-of-thirds intersections

    # --- Text detection ---
    text_dilation_kernel: int = 15  # Dilation kernel size for text regions
    text_importance: float = 1.0   # Importance score for text pixels

    # --- 4K tiling ---
    tile_threshold: int = 2048     # Process as tiles if any dimension exceeds this
    tile_size: int = 1024          # Size of each tile
    tile_overlap: int = 64         # Overlap between tiles to avoid edge artifacts

    # --- YOLO model ---
    yolo_model: str = "yolov8s.pt"
    yolo_confidence: float = 0.25  # Minimum detection confidence

    # --- LPIPS ---
    lpips_network: str = "alex"    # 'alex' (fast) or 'vgg' (more accurate)


# Global default config instance
DEFAULT_CONFIG = SARCConfig()
