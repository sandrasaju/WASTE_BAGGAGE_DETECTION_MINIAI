"""
detector.py — YOLOv8 inference wrapper for WasteGuard.

Runs object detection on uploaded images and maps raw COCO/custom class
labels to the three Target categories: Person, Waste, and Plate.
"""

import logging
import os
import sys
from dataclasses import dataclass, field

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping constants
# ---------------------------------------------------------------------------

PERSON_LABELS: set[str] = {"person"}

WASTE_LABELS: set[str] = {
    "waste", "garbage", "trash", "bottle", "bag", "baggage",
    "plastic", "cup", "can",
}

PLATE_LABELS: set[str] = {
    "license plate", "number plate", "plate", "car", "motorcycle",
    "bicycle", "vehicle",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TargetResult:
    found: bool
    count: int
    boxes: list = field(default_factory=list)   # [[x1, y1, x2, y2], ...]
    scores: list = field(default_factory=list)  # [float, ...]
    labels: list = field(default_factory=list)  # [str, ...]


@dataclass
class DetectionResult:
    person: TargetResult
    waste: TargetResult
    plate: TargetResult
    total_objects: int
    raw_boxes: list = field(default_factory=list)
    raw_scores: list = field(default_factory=list)
    raw_labels: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def map_label_to_category(label: str) -> str:
    """Map a raw YOLO class label to a Target category.

    Returns one of: "person", "waste", "plate", or "unknown".
    Pure function — deterministic, no side effects.
    """
    normalised = label.strip().lower()
    if normalised in PERSON_LABELS:
        return "person"
    if normalised in WASTE_LABELS:
        return "waste"
    if normalised in PLATE_LABELS:
        return "plate"
    return "unknown"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """Wraps a YOLOv8 model and provides structured detection results."""

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            logger.critical("Model file not found: %s — exiting.", model_path)
            sys.exit(1)
        self.model = YOLO(model_path)
        logger.info("YOLOv8 model loaded from %s", model_path)

    def detect(self, image_path: str) -> DetectionResult:
        """Run inference on *image_path* and return a structured DetectionResult."""
        try:
            results = self.model(image_path)

            raw_boxes: list = []
            raw_scores: list = []
            raw_labels: list = []

            person_boxes, person_scores, person_labels = [], [], []
            waste_boxes, waste_scores, waste_labels = [], [], []
            plate_boxes, plate_scores, plate_labels = [], [], []

            for result in results:
                boxes_tensor = result.boxes
                for box in boxes_tensor:
                    xyxy = box.xyxy[0].tolist()          # [x1, y1, x2, y2]
                    score = float(box.conf[0])
                    cls_idx = int(box.cls[0])
                    label = result.names[cls_idx]

                    raw_boxes.append(xyxy)
                    raw_scores.append(score)
                    raw_labels.append(label)

                    category = map_label_to_category(label)
                    if category == "person":
                        person_boxes.append(xyxy)
                        person_scores.append(score)
                        person_labels.append(label)
                    elif category == "waste":
                        waste_boxes.append(xyxy)
                        waste_scores.append(score)
                        waste_labels.append(label)
                    elif category == "plate":
                        plate_boxes.append(xyxy)
                        plate_scores.append(score)
                        plate_labels.append(label)

            person = TargetResult(
                found=len(person_boxes) > 0,
                count=len(person_boxes),
                boxes=person_boxes,
                scores=person_scores,
                labels=person_labels,
            )
            waste = TargetResult(
                found=len(waste_boxes) > 0,
                count=len(waste_boxes),
                boxes=waste_boxes,
                scores=waste_scores,
                labels=waste_labels,
            )
            plate = TargetResult(
                found=len(plate_boxes) > 0,
                count=len(plate_boxes),
                boxes=plate_boxes,
                scores=plate_scores,
                labels=plate_labels,
            )

            return DetectionResult(
                person=person,
                waste=waste,
                plate=plate,
                total_objects=person.count + waste.count + plate.count,
                raw_boxes=raw_boxes,
                raw_scores=raw_scores,
                raw_labels=raw_labels,
            )

        except Exception:
            logger.error("Inference failed for %s", image_path, exc_info=True)
            raise
