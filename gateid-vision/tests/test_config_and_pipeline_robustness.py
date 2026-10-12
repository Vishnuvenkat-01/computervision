"""
Robustness and Edge-Case Tests for GateID Vision.

Tests configuration validation, invalid/corrupt inputs, resource release,
tracker failure resilience, zero-area geometry, and pipeline cleanup.
"""
from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import unittest
import numpy as np

from src.association import AssociationEngine, BoundingBox, DetectedCard, TrackedPerson
from src.config_loader import load_config
from src.event_capture import EventCapturer
from src.id_card_detector import IDCardDetector
from src.person_tracker import PersonTracker
from src.report_generator import ReportGenerator
from src.status_classifier import (
    STATUS_ID_VISIBLE,
    STATUS_UNCERTAIN,
    STATUS_VERIFICATION_REQUIRED,
    StatusClassifier,
)
from src.video_processor import VideoProcessor


class TestConfigAndPipelineRobustness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_config(self):
        config_path = Path("config.yaml")
        self.assertTrue(config_path.exists())
        cfg = load_config(str(config_path))
        self.assertIn("system", cfg)
        self.assertIn("models", cfg)
        self.assertIn("processing", cfg)
        self.assertIn("association", cfg)
        self.assertIn("event_capture", cfg)
        self.assertIn("output", cfg)

    def test_load_nonexistent_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_config_path_12345.yaml")

    def test_person_tracker_uninitialized_model_handling(self):
        # When model is None (e.g. ultralytics not installed), track() returns [] safely
        tracker = PersonTracker({}, model_override=None)
        tracker._model = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        people = tracker.track(frame)
        self.assertEqual(people, [])

    def test_zero_area_bounding_box_safety(self):
        # Edge case: degenerate bounding box with zero width or height
        zero_box = BoundingBox(10.0, 10.0, 10.0, 10.0)
        self.assertEqual(zero_box.width, 0.0)
        self.assertEqual(zero_box.height, 0.0)
        self.assertEqual(zero_box.area, 0.0)
        self.assertEqual(zero_box.iou(BoundingBox(10, 10, 20, 20)), 0.0)
        self.assertEqual(zero_box.containment_ratio(BoundingBox(10, 10, 20, 20)), 0.0)

    def test_multiple_cards_highest_confidence_selected(self):
        # When multiple cards fall within the same person's torso, highest confidence is selected
        engine = AssociationEngine()
        person = TrackedPerson(1, BoundingBox(100, 100, 200, 300), 0.95)
        card_low = DetectedCard(1, BoundingBox(120, 150, 160, 200), 0.60)
        card_high = DetectedCard(2, BoundingBox(130, 150, 170, 200), 0.92)

        matches = engine.associate([person], [card_low, card_high])
        self.assertIn(1, matches)
        self.assertEqual(matches[1].status, "associated")
        self.assertEqual(matches[1].card.card_id, 2)
        self.assertAlmostEqual(matches[1].card.confidence, 0.92)

    def test_status_classifier_reset_clears_history(self):
        classifier = StatusClassifier()
        person = TrackedPerson(1, BoundingBox(100, 100, 200, 300), 0.9)
        assoc = MagicMock(status="unassociated")

        classifier.classify(
            person=person,
            association=assoc,
            frame_number=1,
            frame_width=1280,
            frame_height=720,
            laplacian_variance=100.0,
        )
        self.assertIn(1, classifier.history)

        classifier.reset()
        self.assertEqual(len(classifier.history), 0)

    def test_event_capturer_reset_clears_all_records(self):
        capturer = EventCapturer({"event_capture": {"storage_directory": self.temp_dir.name}})
        person = TrackedPerson(1, BoundingBox(100, 100, 200, 300), 0.9)
        assoc = MagicMock(status="associated", card=DetectedCard(1, BoundingBox(120, 150, 160, 200), 0.88))
        decision = MagicMock(track_id=1, status="id_visible", confidence=0.88, reason="ID visible", frames_observed=5)

        capturer.capture_event(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            person=person,
            association=assoc,
            decision=decision,
            video_filename="test.mp4",
            video_timestamp=1.0,
            frame_number=1,
        )
        self.assertEqual(len(capturer.get_events()), 1)

        capturer.reset()
        self.assertEqual(len(capturer.get_events()), 0)
        self.assertEqual(len(capturer._last_capture_time), 0)
        self.assertEqual(len(capturer._snapshot_count), 0)

    def test_video_processor_handles_corrupt_empty_video(self):
        # Create an empty 0-byte file mimicking corrupted video header
        corrupt_video = Path(self.temp_dir.name) / "corrupt_video.mp4"
        corrupt_video.write_bytes(b"")

        processor = VideoProcessor({
            "output": {
                "reports_directory": f"{self.temp_dir.name}/reports",
                "processed_videos_directory": f"{self.temp_dir.name}/videos",
            }
        })
        result = processor.process_video(str(corrupt_video))

        # Must report failure gracefully without raising uncaught exceptions
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

    def test_report_generator_custom_output_path(self):
        custom_csv = Path(self.temp_dir.name) / "custom_sub" / "custom_report.csv"
        gen = ReportGenerator()
        out_path = gen.export_csv(events=[], custom_path=custom_csv)

        self.assertEqual(out_path, custom_csv)
        self.assertTrue(custom_csv.exists())


if __name__ == "__main__":
    unittest.main()
