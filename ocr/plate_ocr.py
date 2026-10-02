"""
OCR engine for licence plates using EasyOCR with multi-variant preprocessing.

Pipeline
--------
1. Receive tight plate crop (BGR, any size).
2. Upscale to a canonical height if too small.
3. Run several OpenCV preprocessing variants (grayscale, CLAHE, threshold…).
4. Run EasyOCR on all variants and collect candidate strings + confidences.
5. Pick the highest-confidence string and clean it.
6. Return the cleaned string or "UNKNOWN" when confidence is too low.
"""

from __future__ import annotations

import re
import logging
from typing import NamedTuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Minimum EasyOCR character-level confidence to trust a result
MIN_OCR_CONFIDENCE = 0.35
# If the plate crop is shorter than this (px), upscale before OCR
MIN_HEIGHT_PX = 48


class OcrResult(NamedTuple):
    text: str
    confidence: float   # 0–1


class PlateOCR:
    def __init__(self):
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is required. Install requirements.txt and restart the app."
            ) from exc
        import torch

        gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(["en"], gpu=gpu)
        log.info("PlateOCR initialised (gpu=%s)", gpu)

    # ------------------------------------------------------------------
    def read(self, image) -> str:
        """Return the cleaned plate string, or 'UNKNOWN' on failure."""
        result = self.read_with_confidence(image)
        return result.text

    def read_with_confidence(self, image) -> OcrResult:
        """Return OcrResult(text, confidence)."""
        if image is None or image.size == 0:
            return OcrResult("UNKNOWN", 0.0)

        variants = self._make_variants(image)
        candidates: list[OcrResult] = []

        for variant in variants:
            rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
            try:
                raw = self.reader.readtext(rgb, detail=1, paragraph=False)
            except Exception as exc:
                log.debug("EasyOCR error on variant: %s", exc)
                continue

            if not raw:
                continue

            # raw items: (bbox, text, confidence)
            total_conf = sum(item[2] for item in raw) / len(raw)
            combined_text = "".join(item[1] for item in raw)
            cleaned = self._clean(combined_text)
            # Very short OCR fragments such as "75" are usually stray text,
            # not a usable vehicle registration, and should not become IDs.
            if len(cleaned) >= 6:
                candidates.append(OcrResult(cleaned, total_conf))

        if not candidates:
            return OcrResult("UNKNOWN", 0.0)

        # Pick highest-confidence candidate
        best = max(candidates, key=lambda r: r.confidence)
        if best.confidence < MIN_OCR_CONFIDENCE:
            log.debug("OCR confidence %.2f below threshold; returning UNKNOWN", best.confidence)
            return OcrResult("UNKNOWN", best.confidence)

        return best

    # ------------------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        """Uppercase, keep only A-Z and 0-9."""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    # ------------------------------------------------------------------
    @staticmethod
    def _make_variants(bgr) -> list:
        """
        Generate several preprocessed versions of the plate crop.

        Each variant tries to maximise EasyOCR accuracy under different
        lighting / contrast conditions.
        """
        variants = []

        # Ensure a minimum height for legibility
        h, w = bgr.shape[:2]
        if h < MIN_HEIGHT_PX:
            scale = MIN_HEIGHT_PX / h
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # 1. Original colour
        variants.append(bgr)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2. Plain grayscale
        variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

        # 3. CLAHE-enhanced grayscale (improves low-contrast plates)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        # 4. Bilateral filter + Otsu threshold (clean binary)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

        # 5. Inverted binary (for dark-background plates)
        variants.append(cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR))

        # 6. Sharpened colour
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(bgr, -1, kernel)
        variants.append(sharpened)

        return variants
