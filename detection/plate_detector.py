"""
Number plate detector wrapper.

Runs the custom YOLO plate model on a cropped region of the frame
and returns detections sorted by confidence (highest first).
"""

from pathlib import Path
from ultralytics import YOLO


class PlateDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.50,
        iou: float = 0.45,
        device: str | None = None,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Number plate model not found: {path}")
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.iou = iou
        self.device = device or ""

    def detect(self, frame, region=None):
        """
        Detect licence plates inside *region* (x1,y1,x2,y2) of *frame*.

        Returns a list of (confidence, (x1,y1,x2,y2)) tuples in frame
        coordinates, sorted highest-confidence first.
        """
        offset_x, offset_y = 0, 0
        source = frame
        if region is not None:
            x1, y1, x2, y2 = map(int, region)
            source = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        if source is None or source.size == 0:
            return []

        result = self.model.predict(
            source,
            conf=self.confidence,
            iou=self.iou,
            imgsz=1280,
            device=self.device or None,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                bx1 += offset_x
                bx2 += offset_x
                by1 += offset_y
                by2 += offset_y
                detections.append((float(box.conf[0]), (bx1, by1, bx2, by2)))

        return sorted(detections, reverse=True)
