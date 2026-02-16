"""Default configuration values — single source of truth."""

from pathlib import Path

WEIGHTS_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent / "weights" / "yolo_animal_detector.pt"
)
CONFIDENCE_THRESHOLD: float = 0.15
GRAYSCALE: bool = False
SHOW_IMAGE_PREVIEWS: bool = True

# Three-tier embedding layers for YOLOv9c.
# Each entry: (layer_index, key, display_name)
EMBEDDING_LAYERS: list[tuple[int, str, str]] = [
    (15, "p3_fine_detail", "P3 Fine Detail"),
    (18, "p4_object_parts", "P4 Object Parts"),
    (21, "p5_global_context", "P5 Global Context"),
]
