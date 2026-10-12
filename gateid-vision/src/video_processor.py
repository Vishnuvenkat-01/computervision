"""
Video processing pipeline for GateID Vision.

Integrates person tracking, ID-card detection, spatial association,
temporal status classification, event capture, annotation rendering,
and annotated video export.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import time
import numpy as np

from src.cv_compat import cv2
from src.association import AssociationEngine, AssociationMatch, BoundingBox, DetectedCard, TrackedPerson
from src.event_capture import CapturedEvent, EventCapturer
from src.id_card_detector import IDCardDetector
from src.metrics import OperationalSummary, calculate_pipeline_metrics
from src.person_tracker import PersonTracker
from src.report_generator import ReportGenerator
from src.status_classifier import (
    STATUS_ID_VISIBLE,
    STATUS_UNCERTAIN,
    STATUS_VERIFICATION_REQUIRED,
    ClassificationDecision,
    StatusClassifier,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result returned after processing a video file."""
    video_path: str
    total_frames: int
    processed_frames: int
    elapsed_time_sec: float
    output_video_path: Optional[str]
    report_csv_path: Optional[str]
    events: List[CapturedEvent]
    metrics: OperationalSummary
    success: bool
    error_message: Optional[str] = None


class VideoProcessor:
    """
    Orchestrates the frame-by-frame computer vision pipeline for recorded campus entry videos.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        person_tracker: Optional[PersonTracker] = None,
        id_detector: Optional[IDCardDetector] = None,
        association_engine: Optional[AssociationEngine] = None,
        status_classifier: Optional[StatusClassifier] = None,
        event_capturer: Optional[EventCapturer] = None,
        report_generator: Optional[ReportGenerator] = None,
    ):
        self.config = config or {}
        cfg_proc = self.config.get("processing", {})
        self.frame_stride = max(1, int(cfg_proc.get("frame_stride", 2)))
        self.max_res = cfg_proc.get("max_processing_resolution", [1280, 720])
        self.blur_threshold = float(cfg_proc.get("blur_threshold_variance", 60.0))

        cfg_out = self.config.get("output", {})
        self.codec = cfg_out.get("video_codec", "mp4v")
        self.processed_videos_dir = Path(cfg_out.get("processed_videos_directory", "outputs/processed_videos"))
        self.processed_videos_dir.mkdir(parents=True, exist_ok=True)

        # Inject or initialize pipeline subcomponents
        self.person_tracker = person_tracker or PersonTracker(config=self.config)
        self.id_detector = id_detector or IDCardDetector(config=self.config)
        self.association_engine = association_engine or AssociationEngine(config=self.config)
        self.status_classifier = status_classifier or StatusClassifier(config=self.config)
        self.event_capturer = event_capturer or EventCapturer(config=self.config)
        self.report_generator = report_generator or ReportGenerator(config=self.config)

    def calculate_sharpness(self, frame: np.ndarray) -> float:
        """Computes Laplacian variance as an image sharpness metric."""
        if frame is None or frame.size == 0:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def resize_frame_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Resizes frame if it exceeds maximum configured resolution, maintaining aspect ratio."""
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        max_w, max_h = self.max_res[0], self.max_res[1]

        if w <= max_w and h <= max_h:
            return frame

        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def annotate_frame(
        self,
        frame: np.ndarray,
        people: List[TrackedPerson],
        cards: List[DetectedCard],
        associations: Dict[int, AssociationMatch],
        decisions: Dict[int, ClassificationDecision],
        frame_idx: int,
        timestamp_sec: float,
        sharpness: float,
    ) -> np.ndarray:
        """
        Renders bounding boxes, tracking labels, torso guides, and status banners.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw detected ID cards first (cyan boxes)
        for card in cards:
            cx1, cy1, cx2, cy2 = map(int, [card.bbox.x1, card.bbox.y1, card.bbox.x2, card.bbox.y2])
            cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
            cv2.putText(
                annotated,
                f"ID Card {card.confidence:.0%}",
                (cx1, max(15, cy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw people and status decisions
        for person in people:
            tid = person.track_id
            decision = decisions.get(tid)
            assoc = associations.get(tid)
            status = decision.status if decision else STATUS_UNCERTAIN

            # Color scheme:
            # Green = id_visible
            # Amber/Orange = verification_required
            # Magenta/Red = uncertain
            if status == STATUS_ID_VISIBLE:
                color = (40, 200, 40)
                status_txt = "ID Visible"
            elif status == STATUS_VERIFICATION_REQUIRED:
                color = (0, 165, 255)
                status_txt = "Verify ID"
            else:
                color = (180, 50, 220)
                status_txt = "Uncertain"

            px1, py1, px2, py2 = map(int, [person.bbox.x1, person.bbox.y1, person.bbox.x2, person.bbox.y2])
            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)

            # Torso inspection zone (subtle dotted outline)
            torso = self.association_engine.compute_torso_region(person.bbox)
            tx1, ty1, tx2, ty2 = map(int, [torso.x1, torso.y1, torso.x2, torso.y2])
            cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (120, 120, 120), 1, cv2.LINE_4)

            # Label banner above person
            label = f"Person {tid} | {status_txt}"
            cv2.putText(
                annotated,
                label,
                (px1, max(20, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Top Information Banner
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Overlay text
        mins = int(timestamp_sec // 60)
        secs = int(timestamp_sec % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        vis_count = sum(1 for d in decisions.values() if d.status == STATUS_ID_VISIBLE)
        req_count = sum(1 for d in decisions.values() if d.status == STATUS_VERIFICATION_REQUIRED)
        unc_count = sum(1 for d in decisions.values() if d.status == STATUS_UNCERTAIN)

        info_text = (
            f"Frame: {frame_idx} | Time: {time_str} | Active: {len(people)} | "
            f"ID Visible: {vis_count} | Verify: {req_count} | Uncertain: {unc_count}"
        )
        cv2.putText(annotated, info_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        # Detector fallback warning badge if custom weights are missing
        if not self.id_detector.is_available:
            cv2.putText(
                annotated,
                "[FALLBACK MODE: NO ID MODEL]",
                (w - 270, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def process_video(
        self,
        video_path: str,
        output_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[float, int, int, np.ndarray, Dict[str, Any]], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> ProcessingResult:
        """
        Processes recorded video file frame by frame.

        Parameters:
            video_path: Absolute or relative path to input video file.
            output_filename: Optional name for annotated output video.
            progress_callback: Optional hook for Streamlit (progress_ratio, frame_num, total_frames, frame, stats).
            stop_check: Optional hook returning True if operator requested cancellation.
        """
        input_p = Path(video_path)
        if not input_p.exists():
            return ProcessingResult(
                video_path=video_path,
                total_frames=0,
                processed_frames=0,
                elapsed_time_sec=0.0,
                output_video_path=None,
                report_csv_path=None,
                events=[],
                metrics=OperationalSummary(0, 0, 0, 0, 0, 0, 0.0, 0.0, False),
                success=False,
                error_message=f"Video file does not exist: {video_path}",
            )

        if input_p.stat().st_size == 0:
            return ProcessingResult(
                video_path=video_path,
                total_frames=0,
                processed_frames=0,
                elapsed_time_sec=0.0,
                output_video_path=None,
                report_csv_path=None,
                events=[],
                metrics=OperationalSummary(0, 0, 0, 0, 0, 0, 0.0, 0.0, False),
                success=False,
                error_message=f"Video file is empty (0 bytes): {video_path}",
            )

        cap = cv2.VideoCapture(str(input_p))
        if not cap.isOpened():
            return ProcessingResult(
                video_path=video_path,
                total_frames=0,
                processed_frames=0,
                elapsed_time_sec=0.0,
                output_video_path=None,
                report_csv_path=None,
                events=[],
                metrics=OperationalSummary(0, 0, 0, 0, 0, 0, 0.0, 0.0, False),
                success=False,
                error_message=f"Could not open video file: {video_path}",
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            cap.release()
            return ProcessingResult(
                video_path=video_path,
                total_frames=0,
                processed_frames=0,
                elapsed_time_sec=0.0,
                output_video_path=None,
                report_csv_path=None,
                events=[],
                metrics=OperationalSummary(0, 0, 0, 0, 0, 0, 0.0, 0.0, False),
                success=False,
                error_message="Video contains 0 readable frames.",
            )

        # Prepare Video Writer
        out_name = output_filename or f"annotated_{input_p.stem}.mp4"
        out_video_path = self.processed_videos_dir / out_name

        # Determine target resolution after scale
        test_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        scaled_test = self.resize_frame_if_needed(test_frame)
        target_h, target_w = scaled_test.shape[:2]

        effective_fps = max(1.0, orig_fps / self.frame_stride)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(out_video_path), fourcc, effective_fps, (target_w, target_h))

        if not writer.isOpened():
            # Codec fallback to mp4v or XVID
            logger.warning("VideoWriter failed with codec '%s', attempting fallback 'mp4v'", self.codec)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, effective_fps, (target_w, target_h))

        # Reset states
        self.person_tracker.reset()
        self.status_classifier.reset()
        self.event_capturer.reset()

        unique_tracked_ids = set()
        frame_idx = 0
        processed_count = 0
        start_time = time.time()

        try:
            while True:
                if stop_check and stop_check():
                    logger.info("Processing halted by user stop request.")
                    break

                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    break

                frame_idx += 1

                # Apply frame sampling stride
                if (frame_idx - 1) % self.frame_stride != 0:
                    continue

                processed_count += 1
                timestamp_sec = frame_idx / orig_fps

                frame = self.resize_frame_if_needed(raw_frame)
                sharpness = self.calculate_sharpness(frame)

                # 1. Person Detection & Tracking
                tracked_people = self.person_tracker.track(frame)
                for p in tracked_people:
                    unique_tracked_ids.add(p.track_id)

                # 2. ID-Card Detection
                detected_cards = self.id_detector.detect(frame, people=tracked_people)

                # 3. Spatial & Geometric Association
                associations = self.association_engine.associate(tracked_people, detected_cards)

                # 4. Status Classification
                detector_ready = self.id_detector.is_available or getattr(self.id_detector, "has_vision_fallback", False)
                current_decisions: Dict[int, ClassificationDecision] = {}
                for person in tracked_people:
                    assoc = associations[person.track_id]
                    decision = self.status_classifier.classify(
                        person=person,
                        association=assoc,
                        frame_number=frame_idx,
                        frame_width=target_w,
                        frame_height=target_h,
                        laplacian_variance=sharpness,
                        is_detector_available=detector_ready,
                    )
                    current_decisions[person.track_id] = decision

                    # 5. Automatic Event Capture
                    self.event_capturer.capture_event(
                        frame=frame,
                        person=person,
                        association=assoc,
                        decision=decision,
                        video_filename=input_p.name,
                        video_timestamp=timestamp_sec,
                        frame_number=frame_idx,
                    )

                # 6. Annotate and Write Frame
                annotated_frame = self.annotate_frame(
                    frame=frame,
                    people=tracked_people,
                    cards=detected_cards,
                    associations=associations,
                    decisions=current_decisions,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    sharpness=sharpness,
                )

                if writer.isOpened():
                    writer.write(annotated_frame)

                # 7. Progress Callback for Dashboard
                if progress_callback:
                    ratio = min(1.0, frame_idx / total_frames)
                    current_stats = {
                        "active_people": len(tracked_people),
                        "unique_people": len(unique_tracked_ids),
                        "total_events": len(self.event_capturer.get_events()),
                    }
                    progress_callback(ratio, frame_idx, total_frames, annotated_frame, current_stats)

        except Exception as e:
            logger.error("Error during video processing loop: %s", e, exc_info=True)
            return ProcessingResult(
                video_path=video_path,
                total_frames=total_frames,
                processed_frames=processed_count,
                elapsed_time_sec=time.time() - start_time,
                output_video_path=None,
                report_csv_path=None,
                events=self.event_capturer.get_events(),
                metrics=OperationalSummary(0, 0, 0, 0, 0, 0, 0.0, 0.0, False),
                success=False,
                error_message=f"Processing failed: {e}",
            )
        finally:
            cap.release()
            if writer.isOpened():
                writer.release()

        elapsed_sec = time.time() - start_time
        events = self.event_capturer.get_events()

        # Generate CSV report
        csv_filename = f"report_{input_p.stem}.csv"
        csv_path = self.report_generator.export_csv(events=events, filename=csv_filename)

        detector_ready = self.id_detector.is_available or getattr(self.id_detector, "has_vision_fallback", False)
        metrics = calculate_pipeline_metrics(
            events=events,
            unique_track_ids=len(unique_tracked_ids),
            total_frames=processed_count,
            elapsed_time_sec=elapsed_sec,
            is_detector_available=detector_ready,
        )

        # Validate that output video exists and is non-empty
        valid_video_path = str(out_video_path.as_posix()) if out_video_path.exists() and out_video_path.stat().st_size > 0 else None

        return ProcessingResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=processed_count,
            elapsed_time_sec=elapsed_sec,
            output_video_path=valid_video_path,
            report_csv_path=str(csv_path.as_posix()),
            events=events,
            metrics=metrics,
            success=True,
            error_message=None,
        )
