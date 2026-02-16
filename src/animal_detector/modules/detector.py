from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image
from ultralytics import YOLO  # type: ignore[attr-defined]

from animal_detector.modules.config import CONFIDENCE_THRESHOLD, GRAYSCALE


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class YoloObjectDetector:
    """Runs YOLO inference on images and returns structured detections."""

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        grayscale: bool = GRAYSCALE,
    ) -> None:
        self.model = YOLO(str(weights_path))
        self.confidence_threshold = confidence_threshold
        self.grayscale = grayscale

    def detect(self, image: Image.Image) -> list[Detection]:
        if self.grayscale:
            image = image.convert("L").convert("RGB")
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                label = result.names[int(box.cls[0])]
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections

    def extract_embeddings(
        self, image: Image.Image, layer_indices: list[int]
    ) -> dict[int, npt.NDArray[np.float32]]:
        """Extract feature embeddings from multiple layers in a single forward pass.

        Uses forward hooks to capture intermediate activations without
        altering model state (the ``embed`` parameter corrupts subsequent calls).

        Returns a dict mapping layer index to a 1-D numpy array.
        """
        if self.grayscale:
            image = image.convert("L").convert("RGB")

        captured: dict[int, npt.NDArray[np.float32]] = {}
        handles = []

        def _make_hook(idx: int):  # type: ignore[no-untyped-def]
            def _hook(_module: object, _input: object, output: object) -> None:
                arr: npt.NDArray[np.float32] = output.cpu().numpy()  # type: ignore[union-attr]
                if arr.ndim == 4:
                    arr = arr.mean(axis=(2, 3))
                captured[idx] = arr.flatten().astype(np.float32)

            return _hook

        for layer_idx in layer_indices:
            handle = self.model.model.model[layer_idx].register_forward_hook(
                _make_hook(layer_idx)
            )
            handles.append(handle)

        try:
            self.model(image, conf=self.confidence_threshold, verbose=False)
        finally:
            for h in handles:
                h.remove()

        return captured
