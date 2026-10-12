"""
Automatic event capture module for GateID Vision.

Captures timestamped snapshots and metadata for confirmed ID detections,
verification-required alerts, and uncertain observations, while enforcing
cooldowns and snapshot caps to prevent disk flooding.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import uuid
import numpy as np

from src.cv_compat import cv2
from src.association import AssociationMatch, TrackedPerson
from src.status_classifier import ClassificationDecision

logger = logging.getLogger(__name__)


@dataclass
class CapturedEvent:
    """Metadata recorded for an automatic capture event."""
    event_id: str
    video_filename: str
    video_timestamp: float
    frame_number: int
    track_id: int
    status: str
    person_confidence: float
    card_confidence: Optional[float]
    association_outcome: str
    reason: str
    snapshot_path: str
    crop_path: Optional[str] = None
    manual_review_outcome: Optional[str] = None  # Populated later by human reviewer


class EventCapturer:
    """
    Manages snapshot capture, filesystem persistence, and event metadata logging.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("event_capture", {})
        self.cooldown_seconds = float(cfg.get("cooldown_seconds", 3.0))
        self.max_snapshots_per_person = int(cfg.get("max_snapshots_per_person", 2))
        self.storage_dir = Path(cfg.get("storage_directory", "outputs/snapshots"))
        self.save_crop = bool(cfg.get("save_crop", True))
        self.img_format = cfg.get("format", "jpg")
        self.jpeg_quality = int(cfg.get("jpeg_quality", 90))

        # Memory store of captured events
        self.events: List[CapturedEvent] = []

        # Tracking cooldowns: track_id -> dict of {status: last_timestamp_sec}
        self._last_capture_time: Dict[int, Dict[str, float]] = {}
        # Count of snapshots per track_id
        self._snapshot_count: Dict[int, int] = {}

        # Ensure output directories exist
        self._init_directories()

    def _init_directories(self) -> None:
        """Creates the directory structure for snapshot categories."""
        for sub in ["id_visible", "verification_required", "uncertain"]:
            target = self.storage_dir / sub
            target.mkdir(parents=True, exist_ok=True)

    def should_capture(
        self,
        decision: ClassificationDecision,
        current_timestamp: float,
    ) -> bool:
        """
        Determines whether an event warrants a new snapshot based on
        status changes, cooldown timers, and per-person snapshot caps.
        """
        tid = decision.track_id

        # Check total snapshot cap for this person
        if self._snapshot_count.get(tid, 0) >= self.max_snapshots_per_person:
            return False

        # Status tracking dict for this track ID
        if tid not in self._last_capture_time:
            self._last_capture_time[tid] = {}

        last_time = self._last_capture_time[tid].get(decision.status, None)

        # First time seeing this status for this person: capture if it's meaningful
        if last_time is None:
            # We capture on id_visible, or verification_required, or when uncertain and observed > 5 frames
            if decision.status in ["id_visible", "verification_required"]:
                return True
            if decision.status == "uncertain" and decision.frames_observed >= 2:
                return True
            return False

        # If cooldown has elapsed and status is still active or changed
        if (current_timestamp - last_time) >= self.cooldown_seconds:
            return True

        return False

    def capture_event(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        association: AssociationMatch,
        decision: ClassificationDecision,
        video_filename: str,
        video_timestamp: float,
        frame_number: int,
    ) -> Optional[CapturedEvent]:
        """
        Extracts snapshot and localized crop, writes files, and logs event.
        """
        if not self.should_capture(decision, video_timestamp):
            return None

        tid = person.track_id
        event_id = f"evt_{uuid.uuid4().hex[:8]}"

        # Subdirectory based on classification
        status_sub = decision.status if decision.status in [
            "id_visible", "verification_required", "uncertain"
        ] else "uncertain"
        target_dir = self.storage_dir / status_sub

        filename_base = f"track_{tid}_f{frame_number}_{event_id}"
        snap_filename = f"{filename_base}.{self.img_format}"
        snap_path = target_dir / snap_filename

        crop_rel_path: Optional[str] = None

        try:
            # 1. Save annotated or raw full snapshot
            # Draw person box and card box if available for quick inspection
            annotated = frame.copy()
            bx1, by1, bx2, by2 = map(int, [person.bbox.x1, person.bbox.y1, person.bbox.x2, person.bbox.y2])
            
            # Color code based on status: Green (id_visible), Yellow/Orange (verify), Gray/Red (uncertain)
            box_color = (0, 200, 0) if decision.status == "id_visible" else (
                (0, 165, 255) if decision.status == "verification_required" else (0, 0, 255)
            )
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2)

            if association.card is not None:
                cx1, cy1, cx2, cy2 = map(
                    int,
                    [
                        association.card.bbox.x1,
                        association.card.bbox.y1,
                        association.card.bbox.x2,
                        association.card.bbox.y2,
                    ],
                )
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

            # Write full frame
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            cv2.imwrite(str(snap_path), annotated, encode_params)

            # 2. Save person crop if configured
            if self.save_crop:
                h, w = frame.shape[:2]
                cx1, cy1 = max(0, bx1), max(0, by1)
                cx2, cy2 = min(w, bx2), min(h, by2)
                if (cx2 > cx1) and (cy2 > cy1):
                    crop_img = frame[cy1:cy2, cx1:cx2]
                    crop_filename = f"{filename_base}_crop.{self.img_format}"
                    crop_full_path = target_dir / crop_filename
                    cv2.imwrite(str(crop_full_path), crop_img, encode_params)
                    crop_rel_path = str(crop_full_path.as_posix())

        except Exception as e:
            logger.error("Failed to write snapshot image for track %s: %s", tid, e)
            return None

        # Update cooldown state
        self._last_capture_time[tid][decision.status] = video_timestamp
        self._snapshot_count[tid] = self._snapshot_count.get(tid, 0) + 1

        card_conf = association.card.confidence if association.card else None

        event = CapturedEvent(
            event_id=event_id,
            video_filename=video_filename,
            video_timestamp=round(video_timestamp, 2),
            frame_number=frame_number,
            track_id=tid,
            status=decision.status,
            person_confidence=round(person.confidence, 4),
            card_confidence=round(card_conf, 4) if card_conf is not None else None,
            association_outcome=association.status,
            reason=decision.reason,
            snapshot_path=str(snap_path.as_posix()),
            crop_path=crop_rel_path,
        )

        self.events.append(event)
        logger.info(
            "Captured event %s [Track %s, Status: %s, Reason: %s]",
            event_id,
            tid,
            decision.status,
            decision.reason,
        )
        return event

    def get_events(self) -> List[CapturedEvent]:
        """Returns all captured events."""
        return list(self.events)

    def reset(self) -> None:
        """Resets in-memory event registry and cooldown timers."""
        self.events.clear()
        self._last_capture_time.clear()
        self._snapshot_count.clear()
