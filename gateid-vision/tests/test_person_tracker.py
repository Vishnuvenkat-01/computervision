"""
Unit tests for the PersonTracker in GateID Vision.

Tests person detection parsing, ByteTrack tracking ID extraction,
and empty/corrupt frame handling with mocked models.
"""
from unittest.mock import MagicMock
import unittest
import numpy as np

from src.person_tracker import PersonTracker


class TestPersonTracker(unittest.TestCase):
    def test_mocked_tracking_with_ids(self):
        mock_model = MagicMock()

        box1 = MagicMock()
        box1.xyxy = [[50.0, 50.0, 150.0, 300.0]]
        box1.conf = [0.92]
        box1.id = [1]

        box2 = MagicMock()
        box2.xyxy = [[200.0, 60.0, 320.0, 310.0]]
        box2.conf = [0.89]
        box2.id = [2]

        result = MagicMock()
        result.boxes = [box1, box2]
        mock_model.track.return_value = [result]

        tracker = PersonTracker({}, model_override=mock_model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        people = tracker.track(frame)

        self.assertEqual(len(people), 2)
        self.assertEqual(people[0].track_id, 1)
        self.assertEqual(people[1].track_id, 2)
        self.assertAlmostEqual(people[0].confidence, 0.92)

    def test_empty_or_none_frame(self):
        mock_model = MagicMock()
        tracker = PersonTracker({}, model_override=mock_model)

        self.assertEqual(len(tracker.track(None)), 0)
        self.assertEqual(len(tracker.track(np.array([]))), 0)

    def test_tracker_reset(self):
        tracker = PersonTracker({}, model_override=MagicMock())
        tracker._fallback_id_counter = 10
        tracker.reset()
        self.assertEqual(tracker._fallback_id_counter, 1)


if __name__ == "__main__":
    unittest.main()
