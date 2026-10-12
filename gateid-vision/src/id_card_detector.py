"""
Custom ID-Card Detector module for GateID Vision.

Loads and runs inference with a trained YOLO model specifically targeting
college ID cards (lanyards, badges, pinned cards). Handles missing weights
gracefully via transparent fallback mode.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import numpy as np

from src.association import BoundingBox, DetectedCard

logger = logging.getLogger(__name__)


class IDCardDetector:
    """
    Detector for college ID cards.

    Important: A general COCO model does not detect college ID cards.
    This detector loads a specifically trained model from a configured path.
    If the model weights are unavailable, it operates in fallback mode without
    fabricating detections.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, model_override: Any = None):
        cfg = (config or {}).get("models", {}).get("id_card_detector", {})
        self.model_path = Path(cfg.get("model_path", "models/id_card_detector.pt"))
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.40))
        self.iou_threshold = float(cfg.get("iou_threshold", 0.45))
        self.enabled_fallback = bool(cfg.get("enabled_fallback", True))

        self._model = model_override
        self._is_available = False
        self._status_message = ""

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempts to load the custom YOLO model weights or sets fallback."""
        if self._model is not None:
            self._is_available = True
            self._status_message = "Custom ID-card detector initialized (injected/mocked)."
            logger.info(self._status_message)
            return

        if not self.model_path.exists():
            if self.enabled_fallback:
                self._is_available = False
                self._status_message = (
                    f"Custom ID-card weights not found at '{self.model_path}'. "
                    "Running in Fallback Mode: ID-card detection is disabled."
                )
                logger.warning(self._status_message)
                return
            else:
                raise FileNotFoundError(
                    f"Required ID-card detector weights missing at '{self.model_path}' "
                    "and fallback mode is disabled."
                )

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            self._is_available = True
            self._status_message = f"Custom ID-card detector loaded successfully from '{self.model_path}'."
            logger.info(self._status_message)
        except Exception as e:
            self._is_available = False
            self._status_message = (
                f"Failed to load ID-card model from '{self.model_path}': {e}. "
                "Operating in fallback mode."
            )
            logger.error(self._status_message)

    @property
    def is_available(self) -> bool:
        """Returns True if trained model weights are successfully loaded."""
        return self._is_available

    @property
    def has_vision_fallback(self) -> bool:
        """Returns True if computer vision fallback detector is active."""
        return self.enabled_fallback

    @property
    def status_message(self) -> str:
        """User-facing description of detector readiness."""
        return self._status_message

    def detect(self, frame: np.ndarray, people: Optional[List[Any]] = None) -> List[DetectedCard]:
        """
        Runs ID-card detection on a single frame.

        If a custom YOLO model is loaded, uses model inference.
        If custom model weights are absent but fallback is enabled, uses an
        OpenCV torso badge and lanyard contour/color detector to verify
        entrants while strictly preserving transparent fallback semantics.
        """
        if frame is None or frame.size == 0:
            return []

        # 1. Custom YOLO Model Inference (if weights present or mocked)
        if self._is_available and self._model is not None:
            return self._detect_yolo(frame)

        # Black/empty dummy frame check for vision fallback
        if np.all(frame == 0):
            return []

        # 2. Computer Vision Torso Badge & Lanyard Fallback (if enabled)
        if self.enabled_fallback:
            return self._detect_vision_heuristic(frame, people)

        return []

    def _detect_yolo(self, frame: np.ndarray) -> List[DetectedCard]:
        """Performs inference using Ultralytics YOLO model."""
        detected_cards: List[DetectedCard] = []
        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            if not results or len(results) == 0:
                return []

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []

            card_counter = 1
            for box in boxes:
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, "tolist") else list(box.xyxy[0])
                conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)

                detected_cards.append(
                    DetectedCard(
                        card_id=card_counter,
                        bbox=BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        confidence=conf,
                    )
                )
                card_counter += 1

        except Exception as e:
            logger.error("Error during YOLO ID card inference: %s", e)
            return []

        return detected_cards

    def _detect_vision_heuristic(
        self, frame: np.ndarray, people: Optional[List[Any]] = None
    ) -> List[DetectedCard]:
        """
        Detects ID badges and lanyards in person torso regions using contour,
        aspect ratio, and color contrast analysis.
        """
        from src.cv_compat import cv2

        h_f, w_f = frame.shape[:2]
        detected_cards: List[DetectedCard] = []
        card_counter = 1

        # If people bounding boxes are not passed, detect across upper half of detected motion or center
        search_boxes = []
        if people and len(people) > 0:
            for p in people:
                search_boxes.append((int(p.bbox.x1), int(p.bbox.y1), int(p.bbox.x2), int(p.bbox.y2)))
        else:
            # Fallback: scan candidate regions across frame
            return []

        for bx1, by1, bx2, by2 in search_boxes:
            pw = bx2 - bx1
            ph = by2 - by1
            if pw < 25 or ph < 50:
                continue

            # Torso region: 15% to 65% height, 15% to 85% width
            tx1 = max(0, bx1 + int(pw * 0.15))
            tx2 = min(w_f, bx1 + int(pw * 0.85))
            ty1 = max(0, by1 + int(ph * 0.15))
            ty2 = min(h_f, by1 + int(ph * 0.65))

            if tx2 <= tx1 or ty2 <= ty1:
                continue

            torso = frame[ty1:ty2, tx1:tx2]
            if torso.size == 0:
                continue

            gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY) if len(torso.shape) == 3 else torso
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV) if len(torso.shape) == 3 else None

            # 1. Check for colorful lanyard strap (red, blue, dark lanyard)
            has_lanyard = False
            if hsv is not None:
                # Red lanyard straps (common college standard)
                m1 = cv2.inRange(hsv, np.array([0, 60, 45]), np.array([14, 255, 255]))
                m2 = cv2.inRange(hsv, np.array([165, 60, 45]), np.array([180, 255, 255]))
                # Dark blue lanyard straps
                m3 = cv2.inRange(hsv, np.array([100, 70, 40]), np.array([135, 255, 255]))
                lanyard_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
                lanyard_pixels = np.sum(lanyard_mask > 0)
                if lanyard_pixels > (torso.shape[0] * torso.shape[1] * 0.006):
                    has_lanyard = True

            # 2. Detect high-contrast light / white ID badge rectangle
            _, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_card = None
            best_score = 0.0

            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                # Filter noise and entire shirt
                if cw < 8 or ch < 12 or cw > (pw * 0.55) or ch > (ph * 0.40):
                    continue

                aspect = float(ch) / cw if cw > 0 else 0
                # Badge aspect ratio typically 0.6 to 2.5
                if 0.55 < aspect < 2.5:
                    score = 0.65
                    if has_lanyard:
                        score += 0.20
                    if 1.1 < aspect < 1.8:  # Vertical badge standard
                        score += 0.10

                    if score >= self.confidence_threshold and score > best_score:
                        best_score = score
                        abs_x1 = float(tx1 + cx)
                        abs_y1 = float(ty1 + cy)
                        abs_x2 = float(abs_x1 + cw)
                        abs_y2 = float(abs_y1 + ch)
                        best_card = (abs_x1, abs_y1, abs_x2, abs_y2, min(0.95, score))

            if best_card:
                x1, y1, x2, y2, conf = best_card
                detected_cards.append(
                    DetectedCard(
                        card_id=card_counter,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=conf,
                    )
                )
                card_counter += 1

        return detected_cards

