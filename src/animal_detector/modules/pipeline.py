from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from animal_detector.modules.config import CONFIDENCE_THRESHOLD, GRAYSCALE
from animal_detector.modules.detector import Detection, YoloObjectDetector


@dataclass
class DetectionResult:
    image: Image.Image
    detections: list[Detection]
    summary: dict[str, int]


class DetectionPipeline:
    """Chains detector into a single inference pipeline.

    Plotting is left to the caller so that interactive filters
    (confidence, label) can be applied between detection and visualisation.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        grayscale: bool = GRAYSCALE,
    ) -> None:
        self.detector = YoloObjectDetector(weights_path, confidence_threshold, grayscale=grayscale)

    def run(self, image: Image.Image) -> DetectionResult:
        detections = self.detector.detect(image)
        summary = dict(Counter(det.label for det in detections))
        return DetectionResult(image=image, detections=detections, summary=summary)

    def extract_embeddings(
        self, image: Image.Image, layer_indices: list[int]
    ) -> dict[int, npt.NDArray[np.float32]]:
        """Delegate to the underlying detector's multi-layer embedding extraction."""
        return self.detector.extract_embeddings(image, layer_indices)
