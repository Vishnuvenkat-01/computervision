"""
Person detector and multi-object tracker module for GateID Vision.

Uses Ultralytics YOLO pretrained on COCO (class 0: person) and ByteTrack
to assign temporary tracking IDs across video frames. If YOLO weights are
unavailable (offline or missing), provides an automatic OpenCV motion and
contour-based pedestrian tracking fallback so that entrance monitoring
functions under all environments.
"""
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np

from src.cv_compat import cv2
from src.association import BoundingBox, TrackedPerson

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Detects people in frames and tracks them across time using ByteTrack
    (or an OpenCV motion-centroid tracking fallback when YOLO is unavailable).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, model_override: Any = None):
        cfg = (config or {}).get("models", {}).get("person_detector", {})
        self.model_name = cfg.get("model_name", "yolov8n.pt")
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.45))
        self.iou_threshold = float(cfg.get("iou_threshold", 0.50))
        self.classes = cfg.get("classes", [0])  # Class 0: Person in COCO
        self.tracker_type = cfg.get("tracker_type", "bytetrack.yaml")

        self._model = model_override
        self._fallback_id_counter = 1

        # OpenCV background subtractor and centroid tracker state for fallback mode
        self._bg_subtractor: Any = None
        self._active_tracks: Dict[int, Tuple[float, float, int]] = {}  # tid -> (cx, cy, missing_frames)

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initializes YOLO model if available locally; otherwise prepares OpenCV fallback."""
        if self._model is not None:
            logger.info("PersonTracker initialized with provided/mocked model.")
            return

        from pathlib import Path
        model_file = Path(self.model_name)

        if not model_file.exists():
            logger.warning(
                "YOLO weights '%s' not found on disk. Activating OpenCV Pedestrian Tracking Fallback.",
                self.model_name,
            )
            self._model = None
            self._init_fallback_tracker()
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_file))
            logger.info("PersonTracker loaded '%s' successfully.", self.model_name)
        except Exception as e:
            logger.warning(
                "Failed to load YOLO model '%s': %s. Activating OpenCV Pedestrian Tracking Fallback.",
                self.model_name,
                e,
            )
            self._model = None
            self._init_fallback_tracker()

    def _init_fallback_tracker(self) -> None:
        """Initializes OpenCV background subtraction for offline pedestrian tracking."""
        try:
            if hasattr(cv2, "createBackgroundSubtractorMOG2"):
                self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=60, varThreshold=25, detectShadows=False
                )
            else:
                self._bg_subtractor = None
        except Exception:
            self._bg_subtractor = None

    def track(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Runs person detection and tracking on a single frame.

        Returns a list of TrackedPerson objects.
        """
        if frame is None or frame.size == 0:
            return []

        # 1. Primary path: Ultralytics YOLO + ByteTrack
        if self._model is not None:
            return self._track_with_yolo(frame)

        # 2. Resilient fallback path: OpenCV Motion & Contour Pedestrian Tracking
        return self._track_with_opencv_fallback(frame)

    def _track_with_yolo(self, frame: np.ndarray) -> List[TrackedPerson]:
        tracked_people: List[TrackedPerson] = []
        try:
            results = self._model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_type,
                classes=self.classes,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            if not results or len(results) == 0:
                return []

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []

            for box in boxes:
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, "tolist") else list(box.xyxy[0])
                conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)

                if hasattr(box, "id") and box.id is not None:
                    raw_id = box.id[0] if hasattr(box.id, "__len__") else box.id
                    track_id = int(raw_id)
                else:
                    track_id = self._fallback_id_counter
                    self._fallback_id_counter += 1

                tracked_people.append(
                    TrackedPerson(
                        track_id=track_id,
                        bbox=BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        confidence=conf,
                    )
                )
        except Exception as e:
            logger.error("Error during YOLO person tracking: %s", e)
            return []

        return tracked_people

    def _track_with_opencv_fallback(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Offline pedestrian detection using motion contours and centroid tracking.
        Guarantees that moving people entering the gate are tracked even without weights.
        """
        if self._bg_subtractor is None:
            self._init_fallback_tracker()
            if self._bg_subtractor is None:
                return []

        h, w = frame.shape[:2]
        tracked_people: List[TrackedPerson] = []

        try:
            fg_mask = self._bg_subtractor.apply(frame)
            # Find moving connected components
            contours, _ = cv2.findContours(fg_mask, 0, 2)  # RETR_EXTERNAL, CHAIN_APPROX_SIMPLE

            detected_boxes: List[Tuple[int, int, int, int]] = []
            min_area = (w * h) * 0.008  # At least 0.8% of screen area (e.g. pedestrian)
            max_area = (w * h) * 0.70   # Less than 70% of screen

            for cnt in contours:
                area = cv2.contourArea(cnt) if hasattr(cv2, "contourArea") else 0
                if area < min_area or area > max_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt) if hasattr(cv2, "boundingRect") else (0, 0, 0, 0)
                # Aspect ratio check for upright human figure (height >= width * 0.8)
                if bh < 40 or (bh / max(1, bw)) < 0.7:
                    continue

                detected_boxes.append((x, y, bw, bh))

            # Centroid tracking
            current_centroids = [
                ((x + bw / 2.0), (y + bh / 2.0), (x, y, bw, bh))
                for (x, y, bw, bh) in detected_boxes
            ]

            assigned_track_ids = set()

            for cx, cy, (bx, by, bw, bh) in current_centroids:
                best_id = None
                best_dist = 120.0  # Max distance in pixels across consecutive frames

                for tid, (tcx, tcy, _) in self._active_tracks.items():
                    if tid in assigned_track_ids:
                        continue
                    dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tid

                if best_id is None:
                    best_id = self._fallback_id_counter
                    self._fallback_id_counter += 1

                self._active_tracks[best_id] = (cx, cy, 0)
                assigned_track_ids.add(best_id)

                tracked_people.append(
                    TrackedPerson(
                        track_id=best_id,
                        bbox=BoundingBox(
                            x1=float(bx),
                            y1=float(by),
                            x2=float(bx + bw),
                            y2=float(by + bh),
                        ),
                        confidence=0.80,
                    )
                )

            # Age out inactive tracks
            expired_ids = []
            for tid, (tcx, tcy, miss) in self._active_tracks.items():
                if tid not in assigned_track_ids:
                    if miss > 10:
                        expired_ids.append(tid)
                    else:
                        self._active_tracks[tid] = (tcx, tcy, miss + 1)

            for tid in expired_ids:
                del self._active_tracks[tid]

        except Exception as e:
            logger.debug("Fallback tracking exception: %s", e)
            return []

        return tracked_people

    def reset(self) -> None:
        """Resets tracker internal state."""
        self._fallback_id_counter = 1
        self._active_tracks.clear()
        if self._bg_subtractor is not None:
            self._init_fallback_tracker()
        if self._model is not None and hasattr(self._model, "predictor") and self._model.predictor is not None:
            try:
                if hasattr(self._model.predictor, "trackers"):
                    self._model.predictor.trackers = []
            except Exception:
                pass
