"""
Helmet detection wrapper around a custom-trained YOLO model.

Expected class names in the model (case-insensitive, spaces → underscores):
  with_helmet   → rider is wearing a helmet
  without_helmet / no_helmet → rider is NOT wearing a helmet
  person / rider             → generic person (optional; used for fallback logic)

If the model only has with_helmet / without_helmet (two-class), the pipeline
detects violations directly from the without_helmet predictions.
"""

from pathlib import Path
from ultralytics import YOLO


class HelmetDetector:
    """Wrapper around the supplied helmet model; no model training is performed."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.50,
        iou: float = 0.45,
        device: str | None = None,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Helmet model not found: {path}")
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.iou = iou
        self.device = device or ""

        # Inspect and cache class names for quick look-up
        raw_names: dict = self.model.names  # type: ignore[assignment]
        self.class_names: dict[int, str] = {
            k: str(v).lower().replace(" ", "_").replace("-", "_")
            for k, v in raw_names.items()
        }
        # Determine which class IDs map to which semantic roles
        self.helmet_ids: set[int] = set()
        self.no_helmet_ids: set[int] = set()
        self.person_ids: set[int] = set()
        for cid, name in self.class_names.items():
            if name in ("with_helmet", "helmet"):
                self.helmet_ids.add(cid)
            elif name in ("without_helmet", "no_helmet", "nohelmet"):
                self.no_helmet_ids.add(cid)
            elif name in ("person", "rider", "motorcyclist"):
                self.person_ids.add(cid)

    # ------------------------------------------------------------------
    def track(self, frame):
        """Run YOLO tracking on a single BGR frame and return the result."""
        return self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            iou=self.iou,
            device=self.device or None,
            verbose=False,
            imgsz=640,
        )[0]

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_name(name: str) -> str:
        return str(name).lower().replace(" ", "_").replace("-", "_")

    def semantic_role(self, class_id: int) -> str:
        """Return one of: 'HELMET', 'NO_HELMET', 'PERSON', 'OTHER'."""
        if class_id in self.helmet_ids:
            return "HELMET"
        if class_id in self.no_helmet_ids:
            return "NO_HELMET"
        if class_id in self.person_ids:
            return "PERSON"
        return "OTHER"

    @staticmethod
    def name(result, class_id: int) -> str:
        names = result.names
        raw = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
        return raw.lower().replace(" ", "_").replace("-", "_")
