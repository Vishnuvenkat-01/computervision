"""
Unit tests for VideoProcessor in GateID Vision.

Tests end-to-end orchestration, invalid video handling, progress callbacks,
and user-triggered interruption with mocked components.
"""
from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import unittest
import numpy as np

from src.cv_compat import cv2
from src.association import AssociationEngine, BoundingBox, DetectedCard, TrackedPerson
from src.event_capture import EventCapturer
from src.id_card_detector import IDCardDetector
from src.person_tracker import PersonTracker
from src.report_generator import ReportGenerator
from src.status_classifier import StatusClassifier
from src.video_processor import VideoProcessor


class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            "processing": {
                "frame_stride": 1,
                "max_processing_resolution": [640, 480],
                "blur_threshold_variance": 50.0,
                "minimum_observation_frames": 2,
            },
            "output": {
                "reports_directory": f"{self.temp_dir.name}/reports",
                "processed_videos_directory": f"{self.temp_dir.name}/videos",
                "video_codec": "mp4v",
            },
            "event_capture": {
                "storage_directory": f"{self.temp_dir.name}/snapshots",
                "cooldown_seconds": 1.0,
                "max_snapshots_per_person": 5,
            },
            "models": {
                "id_card_detector": {
                    "model_path": "models/id_card_detector.pt",
                    "enabled_fallback": True,
                }
            },
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nonexistent_video_path(self):
        processor = VideoProcessor(config=self.config)
        result = processor.process_video("data/non_existent_file.mp4")

        self.assertFalse(result.success)
        self.assertIn("does not exist", result.error_message)

    def test_end_to_end_mocked_pipeline(self):
        # Create a valid synthetic video file using cv2.VideoWriter so native OpenCV can decode it
        dummy_video_file = Path(self.temp_dir.name) / "mock_entry.mp4"
        vw = cv2.VideoWriter(str(dummy_video_file), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (640, 480))
        for _ in range(5):
            vw.write(np.zeros((480, 640, 3), dtype=np.uint8))
        vw.release()

        # Mock person tracker to return a person
        mock_tracker = MagicMock(spec=PersonTracker)
        mock_person = TrackedPerson(1, BoundingBox(100, 100, 200, 300), 0.92)
        mock_tracker.track.return_value = [mock_person]

        # Mock card detector to return a visible card
        mock_detector = MagicMock(spec=IDCardDetector)
        mock_detector.is_available = True
        mock_card = DetectedCard(10, BoundingBox(120, 150, 160, 200), 0.88)
        mock_detector.detect.return_value = [mock_card]

        processor = VideoProcessor(
            config=self.config,
            person_tracker=mock_tracker,
            id_detector=mock_detector,
            association_engine=AssociationEngine(self.config),
            status_classifier=StatusClassifier(self.config),
            event_capturer=EventCapturer(self.config),
            report_generator=ReportGenerator(self.config),
        )

        progress_calls = []

        def on_progress(ratio, cur, total, frame, stats):
            progress_calls.append(cur)

        result = processor.process_video(
            video_path=str(dummy_video_file),
            progress_callback=on_progress,
        )

        self.assertTrue(result.success)
        self.assertGreater(result.processed_frames, 0)
        self.assertIsNotNone(result.report_csv_path)
        self.assertTrue(Path(result.report_csv_path).exists())
        self.assertGreater(len(progress_calls), 0)

    def test_stop_check_interruption(self):
        dummy_video_file = Path(self.temp_dir.name) / "mock_entry_stop.mp4"
        vw = cv2.VideoWriter(str(dummy_video_file), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (640, 480))
        for _ in range(5):
            vw.write(np.zeros((480, 640, 3), dtype=np.uint8))
        vw.release()

        processor = VideoProcessor(
            config=self.config,
            person_tracker=MagicMock(spec=PersonTracker, track=MagicMock(return_value=[])),
            id_detector=MagicMock(spec=IDCardDetector, is_available=False, detect=MagicMock(return_value=[])),
        )

        # Stop check returns True immediately on second iteration
        call_count = [0]

        def stop_fn():
            call_count[0] += 1
            return call_count[0] > 1

        result = processor.process_video(
            video_path=str(dummy_video_file),
            stop_check=stop_fn,
        )

        self.assertTrue(result.success)
        self.assertLessEqual(result.processed_frames, 2)


if __name__ == "__main__":
    unittest.main()
