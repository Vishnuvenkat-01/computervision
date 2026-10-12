"""
Unit tests for the IDCardDetector in GateID Vision.

Tests graceful fallback handling when weights are absent, strict prohibition
of fabricated detections, and inference parsing with mocked models.
"""
from unittest.mock import MagicMock
import unittest
import numpy as np

from src.id_card_detector import IDCardDetector


class TestIDCardDetector(unittest.TestCase):
    def test_missing_weights_fallback_mode(self):
        config = {
            "models": {
                "id_card_detector": {
                    "model_path": "models/non_existent_weights.pt",
                    "enabled_fallback": True,
                }
            }
        }
        detector = IDCardDetector(config)

        self.assertFalse(detector.is_available)
        self.assertIn("fallback", detector.status_message.lower())

        # Must return empty list, not fake detections
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 0)

    def test_missing_weights_without_fallback_raises(self):
        config = {
            "models": {
                "id_card_detector": {
                    "model_path": "models/non_existent_weights.pt",
                    "enabled_fallback": False,
                }
            }
        }
        with self.assertRaises(FileNotFoundError):
            IDCardDetector(config)

    def test_mocked_model_inference(self):
        # Create a mock YOLO model
        mock_yolo = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = [[100.0, 150.0, 160.0, 220.0]]
        mock_box.conf = [0.88]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_yolo.predict.return_value = [mock_result]

        config = {
            "models": {
                "id_card_detector": {
                    "model_path": "models/id_card_detector.pt",
                    "confidence_threshold": 0.40,
                }
            }
        }
        detector = IDCardDetector(config, model_override=mock_yolo)

        self.assertTrue(detector.is_available)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cards = detector.detect(frame)

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].card_id, 1)
        self.assertAlmostEqual(cards[0].confidence, 0.88)
        self.assertEqual(cards[0].bbox.x1, 100.0)
        self.assertEqual(cards[0].bbox.y2, 220.0)

    def test_empty_or_none_frame(self):
        mock_yolo = MagicMock()
        detector = IDCardDetector({}, model_override=mock_yolo)

        self.assertEqual(len(detector.detect(None)), 0)
        self.assertEqual(len(detector.detect(np.array([]))), 0)


if __name__ == "__main__":
    unittest.main()
