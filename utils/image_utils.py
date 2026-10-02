"""
Image utility helpers – cropping, saving, and evidence-frame drawing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# JPEG quality for all saved images (0-100).  95 keeps plate text legible.
JPEG_QUALITY = 95


def save_jpeg(path: str | Path, image: np.ndarray) -> str:
    """Write *image* as JPEG at the given *path* (creates parent dirs)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError(f"Could not write image: {target}")
    return str(target)


def crop_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int = 6,
) -> np.ndarray | None:
    """Return a copy of the pixels inside *box* (x1,y1,x2,y2), padded."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None


def draw_evidence(
    frame: np.ndarray,
    *,
    candidate_box: tuple[int, int, int, int],
    plate_box: tuple[int, int, int, int] | None,
    plate_text: str,
    helmet_conf: float,
    plate_conf: float,
    track_id: int | None,
    debug: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes and overlays on a copy of *frame* for the evidence image.

    Red   box  = NO_HELMET rider
    Orange box = detected number plate
    """
    out = frame.copy()
    x1, y1, x2, y2 = candidate_box

    # --- Rider / NO_HELMET box ---
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 2)

    # Multi-line overlay above the rider box
    labels = ["NO HELMET", f"Plate: {plate_text}"]
    if debug:
        labels.append(f"Helm conf: {helmet_conf:.2f}")
        if track_id is not None:
            labels.append(f"Track ID: {track_id}")

    for i, label in enumerate(labels):
        ly = max(12, y1 - 8 - (len(labels) - 1 - i) * 20)
        cv2.putText(out, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)

    # --- Plate box ---
    if plate_box is not None:
        px1, py1, px2, py2 = plate_box
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 140, 255), 2)
        plate_label = f"{plate_text}  ({plate_conf:.2f})" if debug else plate_text
        cv2.putText(
            out, plate_label,
            (px1, max(12, py1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2,
        )

    return out
