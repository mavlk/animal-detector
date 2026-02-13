from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from ultralytics import YOLO  # type: ignore[attr-defined]


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class YoloObjectDetector:
    """Runs YOLO inference on images and returns structured detections."""

    def __init__(self, weights_path: str | Path, confidence_threshold: float = 0.25) -> None:
        self.model = YOLO(str(weights_path))
        self.confidence_threshold = confidence_threshold

    def detect(self, image: Image.Image) -> list[Detection]:
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                label = result.names[int(box.cls[0])]
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections
