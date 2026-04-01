"""
annotator.py — Image annotation utilities for WasteGuard.

Draws colour-coded bounding boxes and confidence-score overlays onto a
copy of the original image using OpenCV, then saves the result to disk.
"""

import logging
import shutil

import cv2

from detector import DetectionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour constants (BGR for OpenCV)
# ---------------------------------------------------------------------------

COLOURS = {
    "person": (0, 0, 255),    # red
    "waste":  (0, 165, 255),  # amber
    "plate":  (255, 165, 0),  # blue
}


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

def annotate(image_path: str, result: DetectionResult, output_path: str) -> str:
    """Draw bounding boxes and label overlays onto the image and save to output_path.

    When result.total_objects == 0, the original image is copied unchanged.
    Returns output_path.
    """
    if result.total_objects == 0:
        shutil.copy(image_path, output_path)
        logger.info("Annotated image saved to %s (total objects: 0)", output_path)
        return output_path

    image = cv2.imread(image_path)

    for category in ("person", "waste", "plate"):
        target = getattr(result, category)
        colour = COLOURS[category]

        for box, score, label in zip(target.boxes, target.scores, target.labels):
            x1, y1, x2, y2 = (int(v) for v in box)

            # Bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness=2)

            # Text overlay above the box
            text = f"{label} {score:.0%}"
            cv2.putText(
                image,
                text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    cv2.imwrite(output_path, image)
    logger.info("Annotated image saved to %s (total objects: %d)", output_path, result.total_objects)
    return output_path
