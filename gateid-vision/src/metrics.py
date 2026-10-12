"""
Metrics calculation module for GateID Vision.

Computes empirical pipeline metrics (unique tracked visitors, event totals,
processing throughput FPS) and ground-truth validation metrics (precision,
recall, F1) when verified test labels are provided.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from src.event_capture import CapturedEvent

logger = logging.getLogger(__name__)


@dataclass
class OperationalSummary:
    """Real-time operational summary computed from pipeline events."""
    unique_tracked_people: int
    total_events_captured: int
    id_visible_count: int
    verification_required_count: int
    uncertain_count: int
    total_frames_processed: int
    processing_time_seconds: float
    approx_fps: float
    id_card_model_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_tracked_people": self.unique_tracked_people,
            "total_events_captured": self.total_events_captured,
            "id_visible_count": self.id_visible_count,
            "verification_required_count": self.verification_required_count,
            "uncertain_count": self.uncertain_count,
            "total_frames_processed": self.total_frames_processed,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "approx_fps": round(self.approx_fps, 2),
            "id_card_model_available": self.id_card_model_available,
        }


def calculate_pipeline_metrics(
    events: List[CapturedEvent],
    unique_track_ids: int,
    total_frames: int,
    elapsed_time_sec: float,
    is_detector_available: bool,
) -> OperationalSummary:
    """
    Computes genuine operational metrics from actual recorded events.

    Does not fabricate accuracy scores without ground truth.
    """
    id_visible = 0
    verification_required = 0
    uncertain = 0

    for evt in events:
        if evt.status == "id_visible":
            id_visible += 1
        elif evt.status == "verification_required":
            verification_required += 1
        elif evt.status == "uncertain":
            uncertain += 1

    fps = (total_frames / elapsed_time_sec) if elapsed_time_sec > 0.0 else 0.0

    return OperationalSummary(
        unique_tracked_people=max(0, unique_track_ids),
        total_events_captured=len(events),
        id_visible_count=id_visible,
        verification_required_count=verification_required,
        uncertain_count=uncertain,
        total_frames_processed=total_frames,
        processing_time_seconds=elapsed_time_sec,
        approx_fps=fps,
        id_card_model_available=is_detector_available,
    )


def calculate_ground_truth_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes Precision, Recall, and F1-score against verified ground truth.

    Returns explicit notice if ground truth data is missing.
    """
    if not ground_truth:
        return {
            "status": "unavailable",
            "message": "Ground-truth annotations not provided. Precision, Recall, and F1 cannot be calculated.",
            "precision": None,
            "recall": None,
            "f1_score": None,
        }

    tp = 0
    fp = 0
    fn = 0

    matched_gt = set()

    for pred in predictions:
        pred_box = pred.get("bbox")
        matched = False
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            gt_box = gt.get("bbox")
            if _compute_iou(pred_box, gt_box) >= iou_threshold:
                tp += 1
                matched_gt.add(gt_idx)
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "status": "computed",
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def _compute_iou(box1: Any, box2: Any) -> float:
    """Helper to compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0
