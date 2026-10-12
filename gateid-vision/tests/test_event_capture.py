"""
Unit tests for the EventCapturer module in GateID Vision.

Tests snapshot cooldown rules, per-person snapshot caps, filesystem structure,
and metadata recording.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.association import AssociationMatch, BoundingBox, DetectedCard, TrackedPerson
from src.event_capture import EventCapturer
from src.status_classifier import ClassificationDecision


class TestEventCapturer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            "event_capture": {
                "cooldown_seconds": 3.0,
                "max_snapshots_per_person": 2,
                "storage_directory": self.temp_dir.name,
                "save_crop": True,
                "format": "jpg",
            }
        }
        self.capturer = EventCapturer(self.config)
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_directory_initialization(self):
        base = Path(self.temp_dir.name)
        self.assertTrue((base / "id_visible").exists())
        self.assertTrue((base / "verification_required").exists())
        self.assertTrue((base / "uncertain").exists())

    def test_snapshot_capture_and_cooldown(self):
        person = TrackedPerson(1, BoundingBox(100, 100, 200, 300), 0.9)
        assoc = AssociationMatch(1, "associated", DetectedCard(1, BoundingBox(120, 140, 160, 180), 0.85))
        decision = ClassificationDecision(
            track_id=1,
            status="id_visible",
            confidence=0.85,
            reason="ID card visible",
            is_new_status=True,
            frames_observed=5,
        )

        # 1. First event at t=1.0s should capture
        evt1 = self.capturer.capture_event(
            frame=self.dummy_frame,
            person=person,
            association=assoc,
            decision=decision,
            video_filename="test.mp4",
            video_timestamp=1.0,
            frame_number=25,
        )
        self.assertIsNotNone(evt1)
        self.assertEqual(len(self.capturer.get_events()), 1)

        # 2. Second event at t=2.0s (< 3.0s cooldown) should NOT capture
        evt2 = self.capturer.capture_event(
            frame=self.dummy_frame,
            person=person,
            association=assoc,
            decision=decision,
            video_filename="test.mp4",
            video_timestamp=2.0,
            frame_number=50,
        )
        self.assertIsNone(evt2)
        self.assertEqual(len(self.capturer.get_events()), 1)

        # 3. Third event at t=4.5s (> 3.0s cooldown) should capture
        evt3 = self.capturer.capture_event(
            frame=self.dummy_frame,
            person=person,
            association=assoc,
            decision=decision,
            video_filename="test.mp4",
            video_timestamp=4.5,
            frame_number=112,
        )
        self.assertIsNotNone(evt3)
        self.assertEqual(len(self.capturer.get_events()), 2)

        # 4. Fourth event at t=8.0s: max_snapshots_per_person is 2, so it must be rejected!
        evt4 = self.capturer.capture_event(
            frame=self.dummy_frame,
            person=person,
            association=assoc,
            decision=decision,
            video_filename="test.mp4",
            video_timestamp=8.0,
            frame_number=200,
        )
        self.assertIsNone(evt4)
        self.assertEqual(len(self.capturer.get_events()), 2)


if __name__ == "__main__":
    unittest.main()
