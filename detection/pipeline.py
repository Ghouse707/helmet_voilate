"""
Main detection pipeline — two-phase violation workflow.

Phase 1 — Temporal confirmation
    Run YOLO helmet tracker every frame.
    A NO_HELMET detection must appear ≥ N_CONFIRM_FRAMES times within
    a CONFIRM_WINDOW-frame sliding window before it is trusted.

Phase 2 — Best-plate collection
    Once confirmed the track enters a PLATE_COLLECT_FRAMES window.
    Every frame: run the plate model on the expanded rider region.
    Keep the clearest plate crop using detector confidence and sharpness.
    At the end of the window (or end of video): run OCR ONCE on the
    best crop → save ONE row to SQLite.

Only one database row per event is written; recognized plates are unique in SQLite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from database.database import ViolationDatabase
from detection.helmet_detector import HelmetDetector
from detection.plate_detector import PlateDetector
from detection.tracker import (
    EventTracker,
    STATUS_CONFIRMED,
    STATUS_COLLECTING,
    STATUS_EXPIRED,
    STATUS_DONE,
)
from ocr.plate_ocr import PlateOCR
from utils.image_utils import crop_box, draw_evidence, save_jpeg

log = logging.getLogger(__name__)

# ── Default paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

HELMET_MODEL_PATH = str(PROJECT_ROOT / "models" / "pretrained" / "helmet_yolov8n.pt")
PLATE_MODEL_PATH  = str(PROJECT_ROOT / "models" / "pretrained" / "indian_plate_yolo11n.pt")

# ── Default thresholds (all overridable from app.py sidebar) ─────────────────
HELMET_CONFIDENCE: float = 0.50
PLATE_CONFIDENCE:  float = 0.50
IOU_THRESHOLD:     float = 0.45

# ── Performance ──────────────────────────────────────────────────────────────
FRAME_STRIDE:   int = 1   # process every Nth frame
DISPLAY_STRIDE: int = 5   # send frame to UI every Nth processed frame
PLATE_DETECT_STRIDE: int = 5  # plate inference every N frames during collection


# ── Collection accumulator ───────────────────────────────────────────────────

@dataclass
class CollectionData:
    """
    Holds state for one confirmed NO_HELMET violation while we accumulate
    the best plate crop across multiple frames.
    """
    key:             str
    track_id:        Optional[int]
    confirmed_frame: int
    candidate_box:   tuple          # current rider bounding box (updated each frame)
    helmet_conf:     float

    # Best plate seen so far (updated whenever a higher-conf plate is found)
    best_plate_score:  float                  = -1.0
    best_plate_conf:  float                  = 0.0
    best_plate_box:   Optional[tuple]        = None
    best_plate_crop:  Optional[np.ndarray]   = None   # BGR, tight crop
    best_raw_frame:   Optional[np.ndarray]   = None   # full frame at best-plate moment
    best_rider_box:   Optional[tuple]        = None   # rider box at best-plate moment

    # Fallback: frame captured at the moment of confirmation
    confirmed_raw_frame: Optional[np.ndarray] = None

    # Up to six rider crops sampled at separate points in the video.
    object_samples: list[tuple[float, np.ndarray]] = field(default_factory=list)
    last_sample_frame: Optional[int] = None


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter  = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-6, area_a + area_b - inter)


def _center_inside(inner: tuple, outer: tuple) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _head_region(person_box: tuple) -> tuple:
    """Top 45 % of a person bounding box — where a helmet would sit."""
    x1, y1, x2, y2 = person_box
    return (x1, y1, x2, y1 + max(1, int((y2 - y1) * 0.45)))


def _vehicle_search_region(frame: np.ndarray, rider_box: tuple) -> tuple:
    """
    Expand a detected rider box to cover the likely motorcycle area.

    Explicit no-helmet detections are often head-sized, so those need a much
    taller and wider context window than a full-person detection.
    """
    x1, y1, x2, y2 = map(int, rider_box)
    h, w = frame.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if bh <= 1.6 * bw:
        # Head-sized box: make a tight portrait crop around the rider and bike.
        return (
            max(0, x1 - int(1.7 * bw)),
            max(0, y1 - bh // 2),
            min(w, x2 + int(1.7 * bw)),
            min(h, y2 + 7 * bh),
        )

    # Full-person box: the vehicle extends below and beside the rider.
    return (
        max(0,    x1 - bw),
        max(0,    y1 - bh // 2),
        min(w,    x2 + bw),
        min(h,    y2 + 3 * bh),
    )


# ── Detection parsing ─────────────────────────────────────────────────────────

def _parse_detections(result, detector: HelmetDetector) -> list[dict]:
    """Convert a YOLO result object → list of detection dicts."""
    out = []
    if result.boxes is None:
        return out
    for box in result.boxes:
        coords = tuple(map(int, box.xyxy[0].tolist()))
        cid    = int(box.cls[0])
        conf   = float(box.conf[0])
        tid    = int(box.id[0]) if box.id is not None else None
        role   = detector.semantic_role(cid)
        name   = detector.class_names.get(cid, str(cid))
        out.append({"box": coords, "role": role, "name": name,
                    "conf": conf, "track_id": tid})
    return out


def _find_violations(detections: list[dict]) -> list[dict]:
    """
    Return detections that represent NO_HELMET violations.

    Strategy A (preferred): model explicitly outputs a NO_HELMET / WITHOUT_HELMET class.
    Strategy B (fallback): model has PERSON detections; infer no-helmet by checking
                           whether any HELMET box overlaps the person's head region.
    """
    explicit = [d for d in detections if d["role"] == "NO_HELMET"]
    if explicit:
        return explicit

    persons = [d for d in detections if d["role"] == "PERSON"]
    helmets  = [d for d in detections if d["role"] == "HELMET"]
    if not persons:
        return []

    violations = []
    for person in persons:
        head = _head_region(person["box"])
        helmet_found = any(
            _iou(head, h["box"]) >= 0.10 or _center_inside(h["box"], head)
            for h in helmets
        )
        if not helmet_found:
            violations.append(person)
    return violations


# ── Finalisation ──────────────────────────────────────────────────────────────

def _finalize_and_save(
    cdata: CollectionData,
    *,
    ocr:        PlateOCR,
    db:         ViolationDatabase,
    project_root: Path,
    video_name: str,
    debug:      bool,
) -> dict | None:
    """
    Run OCR on the best accumulated plate crop, draw the evidence frame,
    persist images and a single DB row.  Returns the record dict or None.
    """
    log.info(
        "Finalising track key=%s  best_plate_conf=%.2f",
        cdata.key, cdata.best_plate_conf,
    )

    # ── OCR ──────────────────────────────────────────────────────────────────
    plate_text = "UNKNOWN"
    ocr_conf   = 0.0
    if cdata.best_plate_crop is not None and cdata.best_plate_crop.size > 0:
        ocr_result = ocr.read_with_confidence(cdata.best_plate_crop)
        plate_text = ocr_result.text
        ocr_conf   = ocr_result.confidence
        log.info("  OCR → '%s'  (conf=%.2f)", plate_text, ocr_conf)
    else:
        log.info("  No plate crop available → plate_text=UNKNOWN")

    # ── Choose the best frame to use as evidence ──────────────────────────────
    # Prefer the frame where the best plate was detected.
    # Fall back to the frame captured at confirmation time.
    evidence_raw = (
        cdata.best_raw_frame
        if cdata.best_raw_frame is not None
        else cdata.confirmed_raw_frame
    )
    rider_box = cdata.best_rider_box or cdata.candidate_box

    if evidence_raw is None:
        log.warning("  No raw frame available for track %s — skipping save.", cdata.key)
        return None

    # ── Annotate evidence frame ───────────────────────────────────────────────
    evidence_annotated = draw_evidence(
        evidence_raw,
        candidate_box = rider_box,
        plate_box     = cdata.best_plate_box,
        plate_text    = plate_text,
        helmet_conf   = cdata.helmet_conf,
        plate_conf    = cdata.best_plate_conf,
        track_id      = cdata.track_id,
        debug         = debug,
    )

    # ── Persist plate crop ────────────────────────────────────────────────────
    plate_path_str: str | None = None
    if cdata.best_plate_crop is not None and cdata.best_plate_crop.size > 0:
        plate_filename = f"plate_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        plate_path_str = save_jpeg(
            project_root / "plates" / plate_filename,
            cdata.best_plate_crop,
        )
        log.info("  Plate image saved: %s", plate_path_str)

    # ── Persist timestamped rider crops ─────────────────────────────────────
    sample_rows: list[dict] = []
    for sample_index, (video_seconds, sample_image) in enumerate(cdata.object_samples, start=1):
        sample_filename = f"violation_{datetime.now():%Y%m%d_%H%M%S_%f}_rider_{sample_index:02d}.jpg"
        sample_path = save_jpeg(project_root / "evidence" / sample_filename, sample_image)
        sample_rows.append({
            "sample_index": sample_index,
            "video_seconds": video_seconds,
            "image_path": sample_path,
        })

    # ── Persist evidence frame ────────────────────────────────────────────────
    ev_filename = f"violation_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    evidence_path_str = save_jpeg(
        project_root / "evidence" / ev_filename,
        evidence_annotated,
    )
    log.info("  Evidence image saved: %s", evidence_path_str)

    # ── Database row ──────────────────────────────────────────────────────────
    stamp  = datetime.now().astimezone().isoformat(timespec="seconds")
    row_id = db.add_violation(
        timestamp            = stamp,
        track_id             = cdata.track_id,
        plate_number         = plate_text,
        plate_confidence     = round(cdata.best_plate_conf, 4),
        helmet_confidence    = round(cdata.helmet_conf, 4),
        helmet_status        = "NO_HELMET",
        plate_image_path     = plate_path_str,
        violation_image_path = evidence_path_str,
        video_source         = video_name,
        event_key            = cdata.key,
    )
    if row_id is None:
        # This event or recognized plate already exists in SQLite.
        Path(evidence_path_str).unlink(missing_ok=True)
        if plate_path_str:
            Path(plate_path_str).unlink(missing_ok=True)
        existing_id = db.event_id(video_name, cdata.key)
        if existing_id is not None and sample_rows:
            # Refresh timestamped context crops for this event without creating
            # a duplicate violation row.
            db.replace_image_samples(existing_id, sample_rows)
        else:
            for sample in sample_rows:
                Path(sample["image_path"]).unlink(missing_ok=True)
        log.info("  Duplicate event/plate skipped: %s", cdata.key)
        return None
    db.replace_image_samples(row_id, sample_rows)
    log.info("  DB row inserted: id=%d  plate=%s", row_id, plate_text)

    return {
        "id":                   row_id,
        "timestamp":            stamp,
        "track_id":             cdata.track_id,
        "plate_number":         plate_text,
        "plate_confidence":     round(cdata.best_plate_conf, 4),
        "helmet_confidence":    round(cdata.helmet_conf, 4),
        "helmet_status":        "NO_HELMET",
        "plate_image_path":     plate_path_str,
        "violation_image_path": evidence_path_str,
        "video_source":         video_name,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    *,
    helmet_model:      str   = HELMET_MODEL_PATH,
    plate_model:       str   = PLATE_MODEL_PATH,
    helmet_confidence: float = HELMET_CONFIDENCE,
    plate_confidence:  float = PLATE_CONFIDENCE,
    iou_threshold:     float = IOU_THRESHOLD,
    frame_stride:      int   = FRAME_STRIDE,
    debug:             bool  = False,
    on_progress = None,
    on_frame    = None,
) -> tuple[list[dict], str]:
    """
    Process *video_path* end-to-end.

    Returns (records, device_string).
    *records* is a list of dicts matching ViolationDatabase.all() rows.
    """
    logging.basicConfig(
        level   = logging.DEBUG if debug else logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
    )

    device = "0" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", "GPU" if device == "0" else "CPU")

    helmet  = HelmetDetector(helmet_model, confidence=helmet_confidence,
                             iou=iou_threshold, device=device)
    plate   = PlateDetector(plate_model,   confidence=plate_confidence,
                            iou=iou_threshold, device=device)
    ocr     = PlateOCR()
    db      = ViolationDatabase(str(PROJECT_ROOT / "violations.db"))
    tracker = EventTracker()

    log.info("Helmet model classes: %s", helmet.class_names)
    log.info("  HELMET=%s  NO_HELMET=%s  PERSON=%s",
             helmet.helmet_ids, helmet.no_helmet_ids, helmet.person_ids)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(video_fps) or video_fps <= 0:
        video_fps = 30.0
    frame_num = 0
    records:  list[dict]                = []
    annotated: np.ndarray | None        = None

    # key → CollectionData for every confirmed-but-unsaved track
    active_collections: dict[str, CollectionData] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1

            # ── Skip frames for speed (tracking still persists via YOLO) ──────
            if (frame_num - 1) % frame_stride != 0:
                if on_frame and annotated is not None and frame_num % DISPLAY_STRIDE == 0:
                    on_frame(annotated)
                if on_progress and frame_num % 10 == 0:
                    on_progress(min(frame_num / total, 1.0))
                continue

            # ── 1. Helmet detection + ByteTrack ──────────────────────────────
            tracked_result = helmet.track(frame)
            detections     = _parse_detections(tracked_result, helmet)
            violations     = _find_violations(detections)

            # ── 2. Annotate live preview frame ───────────────────────────────
            annotated = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                is_viol = det in violations
                color   = (0, 0, 220) if is_viol else (0, 180, 0)
                label   = "NO HELMET" if is_viol else det["name"].replace("_", " ").upper()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if debug:
                    cv2.putText(
                        annotated,
                        f"c={det['conf']:.2f} id={det['track_id']}",
                        (x1, min(frame.shape[0] - 4, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    )

            # ── 3. Process NO_HELMET detections through tracker ──────────────
            for candidate in violations:
                cbox   = candidate["box"]
                cx, cy = (cbox[0] + cbox[2]) / 2, (cbox[1] + cbox[3]) / 2
                tid    = candidate["track_id"]
                hconf  = candidate["conf"]

                status, key = tracker.process_nohel(tid, (cx, cy), frame_num, hconf)

                if status == STATUS_CONFIRMED:
                    log.info(
                        "Frame %d: violation CONFIRMED  key=%s  track_id=%s  conf=%.2f",
                        frame_num, key, tid, hconf,
                    )
                    # Start collecting — store the confirmation frame as fallback evidence
                    active_collections[key] = CollectionData(
                        key                 = key,
                        track_id            = tid,
                        confirmed_frame     = frame_num,
                        candidate_box       = cbox,
                        helmet_conf         = hconf,
                        confirmed_raw_frame = frame.copy(),
                    )

                elif status == STATUS_COLLECTING:
                    if key in active_collections:
                        # Keep rider position up-to-date as they move
                        active_collections[key].candidate_box = cbox

                # STATUS_DONE or STATUS_PENDING → nothing to do

            # ── 4. Plate detection for every active collection ────────────────
            #    Sample periodically to reduce expensive plate-model inference.
            for key, cdata in list(active_collections.items()):
                sample_interval = max(1, tracker.collect_frames // 5)
                if (
                    len(cdata.object_samples) < 6
                    and (
                        cdata.last_sample_frame is None
                        or frame_num - cdata.last_sample_frame >= sample_interval
                    )
                ):
                    vehicle_region = _vehicle_search_region(frame, cdata.candidate_box)
                    vehicle_crop = crop_box(frame, vehicle_region, padding=0)
                    if vehicle_crop is not None and vehicle_crop.size > 0:
                        cdata.object_samples.append((frame_num / video_fps, vehicle_crop))
                        cdata.last_sample_frame = frame_num

                should_detect_plate = (
                    frame_num == cdata.confirmed_frame
                    or frame_num % PLATE_DETECT_STRIDE == 0
                )
                plate_dets = []
                if should_detect_plate:
                    search_region = _vehicle_search_region(frame, cdata.candidate_box)
                    plate_dets = plate.detect(frame, search_region)

                if plate_dets:
                    p_conf, p_box = plate_dets[0]   # highest-confidence first
                    crop = crop_box(frame, p_box)
                    if crop is not None and crop.size > 0:
                        # Rank captures by detector confidence and sharpness so
                        # the OCR input favors a clear plate, not just a confident box.
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        sharpness_score = sharpness / (sharpness + 100.0)
                        quality_score = 0.65 * p_conf + 0.35 * sharpness_score
                        if quality_score > cdata.best_plate_score:
                            cdata.best_plate_score = quality_score
                            cdata.best_plate_conf = p_conf
                            cdata.best_plate_box  = p_box
                            cdata.best_plate_crop = crop.copy()
                            cdata.best_raw_frame  = frame.copy()
                            cdata.best_rider_box  = cdata.candidate_box
                            if debug:
                                log.debug(
                                    "  Frame %d: key=%s  new best plate  conf=%.2f quality=%.2f",
                                    frame_num, key, p_conf, quality_score,
                                )

                        # Draw plate on preview
                        px1, py1, px2, py2 = p_box
                        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 140, 255), 2)
                        cv2.putText(
                            annotated,
                            f"PLATE {p_conf:.2f}" if debug else "PLATE",
                            (px1, max(12, py1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2,
                        )

                # Draw "COLLECTING" indicator on preview
                cx1, cy1, cx2, cy2 = map(int, cdata.candidate_box)
                cv2.putText(
                    annotated,
                    f"COLLECTING... best={cdata.best_plate_conf:.2f}",
                    (cx1, max(14, cy1 - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2,
                )

            # ── 5. Finalise expired collections ──────────────────────────────
            for key in tracker.check_expired(frame_num):
                cdata = active_collections.pop(key, None)
                if cdata is None:
                    tracker.mark_saved(key)
                    continue
                record = _finalize_and_save(
                    cdata,
                    ocr           = ocr,
                    db            = db,
                    project_root  = PROJECT_ROOT,
                    video_name    = Path(video_path).name,
                    debug         = debug,
                )
                tracker.mark_saved(key)
                if record:
                    records.append(record)

            # ── 6. UI callbacks ──────────────────────────────────────────────
            if on_frame and frame_num % DISPLAY_STRIDE == 0 and annotated is not None:
                on_frame(annotated)
            if on_progress and total > 0 and frame_num % 5 == 0:
                on_progress(min(frame_num / total, 1.0))

    finally:
        cap.release()

        # ── 7. End-of-video flush — save any pending collections ──────────────
        pending_keys = tracker.get_all_pending()
        if pending_keys:
            log.info(
                "End of video: flushing %d pending collection(s).", len(pending_keys)
            )
        for key in pending_keys:
            cdata = active_collections.pop(key, None)
            if cdata is None:
                tracker.mark_saved(key)
                continue
            record = _finalize_and_save(
                cdata,
                ocr          = ocr,
                db           = db,
                project_root = PROJECT_ROOT,
                video_name   = Path(video_path).name,
                debug        = debug,
            )
            tracker.mark_saved(key)
            if record:
                records.append(record)

        if on_progress:
            on_progress(1.0)

    log.info("Done. %d violation(s) saved.", len(records))
    return records, ("GPU" if device == "0" else "CPU")
