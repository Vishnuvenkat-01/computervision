"""
Temporal status classifier module for GateID Vision.

Classifies tracked persons into three primary statuses:
1. 'id_visible': Confirmed ID card associated with high confidence.
2. 'verification_required': No card detected after the minimum observation period
   under adequate visual conditions (review prompt, NOT an accusation).
3. 'uncertain': Blurred image, edge clipping, crowd overlap ambiguity, short
   observation window, or unavailable ID-card detector.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import logging

from src.association import AssociationMatch, BoundingBox, TrackedPerson

logger = logging.getLogger(__name__)


# Status constants
STATUS_ID_VISIBLE = "id_visible"
STATUS_VERIFICATION_REQUIRED = "verification_required"
STATUS_UNCERTAIN = "uncertain"


@dataclass
class TrackHistory:
    """Historical observation data for a specific tracked person."""
    track_id: int
    frames_observed: int = 0
    frames_with_card: int = 0
    consecutive_frames_with_card: int = 0
    last_status: str = STATUS_UNCERTAIN
    current_status: str = STATUS_UNCERTAIN
    last_reason: str = ""
    highest_card_confidence: float = 0.0
    first_frame_seen: int = 0
    last_frame_seen: int = 0


@dataclass
class ClassificationDecision:
    """Classification outcome for a person in a given frame."""
    track_id: int
    status: str  # id_visible, verification_required, uncertain
    confidence: float
    reason: str
    is_new_status: bool  # True if state changed from last frame
    frames_observed: int


class StatusClassifier:
    """
    Stateful temporal classifier that evaluates whether an ID card is visible,
    needs manual verification, or remains uncertain.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg_proc = (config or {}).get("processing", {})
        self.min_observation_frames = int(cfg_proc.get("minimum_observation_frames", 15))
        self.blur_threshold_variance = float(cfg_proc.get("blur_threshold_variance", 60.0))
        self.edge_boundary_margin = float(cfg_proc.get("edge_boundary_margin", 0.05))

        # Track ID -> TrackHistory
        self.history: Dict[int, TrackHistory] = {}

    def is_box_clipped(
        self,
        box: BoundingBox,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """
        Determines if a bounding box touches or is dangerously close to frame edges,
        meaning the person's body or chest may be partially out of frame.
        """
        margin_x = frame_width * self.edge_boundary_margin
        margin_y = frame_height * self.edge_boundary_margin

        if box.x1 <= margin_x or box.x2 >= (frame_width - margin_x):
            return True
        if box.y1 <= margin_y or box.y2 >= (frame_height - margin_y):
            return True
        return False

    def classify(
        self,
        person: TrackedPerson,
        association: AssociationMatch,
        frame_number: int,
        frame_width: int,
        frame_height: int,
        laplacian_variance: float,
        is_detector_available: bool = True,
    ) -> ClassificationDecision:
        """
        Classifies a single tracked person for the current frame.
        """
        tid = person.track_id

        # Initialize history if new track
        if tid not in self.history:
            self.history[tid] = TrackHistory(
                track_id=tid,
                first_frame_seen=frame_number,
                last_frame_seen=frame_number,
            )

        hist = self.history[tid]
        hist.frames_observed += 1
        hist.last_frame_seen = frame_number

        # 1. Condition: Custom ID-card detector is not loaded/available
        if not is_detector_available:
            status = STATUS_UNCERTAIN
            reason = "ID-card detector weights unavailable (fallback mode active)"
            conf = 0.0
            return self._finalize_decision(hist, status, conf, reason)

        # 2. Condition: Frame is blurry
        if laplacian_variance < self.blur_threshold_variance:
            status = STATUS_UNCERTAIN
            reason = f"Frame image blurry (sharpness {laplacian_variance:.1f} < {self.blur_threshold_variance:.1f})"
            conf = 0.0
            return self._finalize_decision(hist, status, conf, reason)

        # 3. Condition: Person box is clipped near image boundary
        if self.is_box_clipped(person.bbox, frame_width, frame_height):
            # If card is clearly detected even near edge, we still prioritize detection
            if association.status != "associated":
                status = STATUS_UNCERTAIN
                reason = "Person partially outside frame boundary (torso clipped)"
                conf = 0.0
                return self._finalize_decision(hist, status, conf, reason)

        # 4. Condition: Association is ambiguous due to crowd overlap
        if association.status == "ambiguous_overlap":
            status = STATUS_UNCERTAIN
            reason = association.reason or "Card association ambiguous due to crowd overlap"
            conf = 0.0
            return self._finalize_decision(hist, status, conf, reason)

        # 5. Condition: ID card is detected and associated
        if association.status == "associated" and association.card is not None:
            hist.frames_with_card += 1
            hist.consecutive_frames_with_card += 1
            hist.highest_card_confidence = max(
                hist.highest_card_confidence, association.card.confidence
            )

            status = STATUS_ID_VISIBLE
            reason = f"ID card visible with {association.card.confidence:.1%} confidence"
            conf = association.association_confidence
            return self._finalize_decision(hist, status, conf, reason)

        # 6. Condition: Card NOT detected in current frame
        hist.consecutive_frames_with_card = 0

        # If card was confidently confirmed in recent observation frames, retain id_visible
        # (avoid flickering when person turns head or card momentarily tilts)
        if hist.frames_with_card >= 3 and (hist.frames_with_card / hist.frames_observed) >= 0.25:
            status = STATUS_ID_VISIBLE
            reason = "ID card confirmed from recent temporal observations"
            conf = hist.highest_card_confidence
            return self._finalize_decision(hist, status, conf, reason)

        # 7. Condition: Check observation duration
        if hist.frames_observed < self.min_observation_frames:
            status = STATUS_UNCERTAIN
            reason = f"Observing ({hist.frames_observed}/{self.min_observation_frames} frames required)"
            conf = 0.0
            return self._finalize_decision(hist, status, conf, reason)

        # 8. Adequate observation, clear image, but no card detected -> Verification Required
        status = STATUS_VERIFICATION_REQUIRED
        reason = f"No ID card detected after {hist.frames_observed} clear frames (manual check recommended)"
        conf = 0.0
        return self._finalize_decision(hist, status, conf, reason)

    def _finalize_decision(
        self,
        hist: TrackHistory,
        new_status: str,
        confidence: float,
        reason: str,
    ) -> ClassificationDecision:
        """Updates history and constructs the decision."""
        is_new_status = (new_status != hist.current_status)
        hist.last_status = hist.current_status
        hist.current_status = new_status
        hist.last_reason = reason

        return ClassificationDecision(
            track_id=hist.track_id,
            status=new_status,
            confidence=confidence,
            reason=reason,
            is_new_status=is_new_status,
            frames_observed=hist.frames_observed,
        )

    def get_track_history(self, track_id: int) -> Optional[TrackHistory]:
        """Returns the tracking history for a specific track ID."""
        return self.history.get(track_id)

    def reset(self) -> None:
        """Clears all historical track states."""
        self.history.clear()
