"""
Unit tests for the StatusClassifier in GateID Vision.

Tests temporal state management, blur detection, edge clipping,
observation duration requirements, and transitions between
id_visible, verification_required, and uncertain.
"""
import unittest

from src.association import (
    AssociationMatch,
    BoundingBox,
    DetectedCard,
    TrackedPerson,
)
from src.status_classifier import (
    STATUS_ID_VISIBLE,
    STATUS_UNCERTAIN,
    STATUS_VERIFICATION_REQUIRED,
    StatusClassifier,
)


class TestStatusClassifier(unittest.TestCase):
    def setUp(self):
        self.config = {
            "processing": {
                "minimum_observation_frames": 10,
                "blur_threshold_variance": 60.0,
                "edge_boundary_margin": 0.05,
            }
        }
        self.classifier = StatusClassifier(self.config)
        self.person = TrackedPerson(
            track_id=1,
            bbox=BoundingBox(x1=200.0, y1=100.0, x2=400.0, y2=500.0),
            confidence=0.92,
        )

    def test_missing_detector_weights_uncertain(self):
        assoc = AssociationMatch(track_id=1, status="unassociated")
        decision = self.classifier.classify(
            person=self.person,
            association=assoc,
            frame_number=1,
            frame_width=1280,
            frame_height=720,
            laplacian_variance=120.0,
            is_detector_available=False,
        )
        self.assertEqual(decision.status, STATUS_UNCERTAIN)
        self.assertIn("unavailable", decision.reason.lower())

    def test_blurry_frame_uncertain(self):
        assoc = AssociationMatch(track_id=1, status="unassociated")
        decision = self.classifier.classify(
            person=self.person,
            association=assoc,
            frame_number=1,
            frame_width=1280,
            frame_height=720,
            laplacian_variance=30.0,  # Below threshold of 60.0
            is_detector_available=True,
        )
        self.assertEqual(decision.status, STATUS_UNCERTAIN)
        self.assertIn("blurry", decision.reason.lower())

    def test_edge_clipping_uncertain(self):
        # Person at very edge of frame (x1 = 10 on a 1280w frame, margin = 64)
        clipped_person = TrackedPerson(
            track_id=2,
            bbox=BoundingBox(x1=10.0, y1=100.0, x2=150.0, y2=500.0),
            confidence=0.88,
        )
        assoc = AssociationMatch(track_id=2, status="unassociated")
        decision = self.classifier.classify(
            person=clipped_person,
            association=assoc,
            frame_number=1,
            frame_width=1280,
            frame_height=720,
            laplacian_variance=120.0,
            is_detector_available=True,
        )
        self.assertEqual(decision.status, STATUS_UNCERTAIN)
        self.assertIn("clipped", decision.reason.lower())

    def test_ambiguous_crowd_overlap_uncertain(self):
        assoc = AssociationMatch(
            track_id=1,
            status="ambiguous_overlap",
            reason="Card overlaps multiple people",
        )
        decision = self.classifier.classify(
            person=self.person,
            association=assoc,
            frame_number=1,
            frame_width=1280,
            frame_height=720,
            laplacian_variance=120.0,
            is_detector_available=True,
        )
        self.assertEqual(decision.status, STATUS_UNCERTAIN)
        self.assertTrue(
            "multiple people" in decision.reason.lower() or "crowd" in decision.reason.lower()
        )

    def test_confirmed_id_visible(self):
        card = DetectedCard(
            card_id=1,
            bbox=BoundingBox(x1=260.0, y1=180.0, x2=320.0, y2=260.0),
            confidence=0.87,
        )
        assoc = AssociationMatch(
            track_id=1,
            status="associated",
            card=card,
            association_confidence=0.85,
        )
        decision = self.classifier.classify(
            person=self.person,
            association=assoc,
            frame_number=1,
            frame_width=1280,
            frame_height=720,
            laplacian_variance=120.0,
            is_detector_available=True,
        )
        self.assertEqual(decision.status, STATUS_ID_VISIBLE)
        self.assertEqual(decision.confidence, 0.85)

    def test_temporal_observation_to_verification_required(self):
        # Frame 1 to 9 (< 10 frames): must be UNCERTAIN (observing)
        assoc = AssociationMatch(track_id=1, status="unassociated")
        for f in range(1, 10):
            decision = self.classifier.classify(
                person=self.person,
                association=assoc,
                frame_number=f,
                frame_width=1280,
                frame_height=720,
                laplacian_variance=120.0,
                is_detector_available=True,
            )
            self.assertEqual(decision.status, STATUS_UNCERTAIN)
            self.assertIn("observing", decision.reason.lower())

        # Frame 10 (reaches minimum_observation_frames of 10):
        # Clear image, no card detected -> Must transition to VERIFICATION_REQUIRED
        decision10 = self.classifier.classify(
            person=self.person,
            association=assoc,
            frame_number=10,
            frame_width=1280,
            frame_height=720,
            laplacian_variance=120.0,
            is_detector_available=True,
        )
        self.assertEqual(decision10.status, STATUS_VERIFICATION_REQUIRED)
        self.assertIn("no id card detected", decision10.reason.lower())


if __name__ == "__main__":
    unittest.main()
