"""
Unit tests for the AssociationEngine in GateID Vision.

Tests spatial person-to-card association, anatomical torso boundary computation,
crowd overlap ambiguity resolution, and edge cases.
"""
import unittest

from src.association import (
    AssociationEngine,
    BoundingBox,
    DetectedCard,
    TrackedPerson,
)


class TestAssociationEngine(unittest.TestCase):
    def setUp(self):
        self.config = {
            "association": {
                "torso_y_min_ratio": 0.15,
                "torso_y_max_ratio": 0.65,
                "torso_x_margin_ratio": 0.10,
                "ambiguity_iou_threshold": 0.20,
            }
        }
        self.engine = AssociationEngine(self.config)

    def test_torso_region_computation(self):
        # Person from (100, 100) to (200, 300) -> width=100, height=200
        person_box = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
        torso = self.engine.compute_torso_region(person_box)

        # Expected:
        # margin_x = 100 * 0.10 = 10 -> x1 = 90, x2 = 210
        # y1 = 100 + (200 * 0.15) = 130
        # y2 = 100 + (200 * 0.65) = 230
        self.assertAlmostEqual(torso.x1, 90.0)
        self.assertAlmostEqual(torso.x2, 210.0)
        self.assertAlmostEqual(torso.y1, 130.0)
        self.assertAlmostEqual(torso.y2, 230.0)

    def test_single_person_associated_card(self):
        person = TrackedPerson(
            track_id=1,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0),
            confidence=0.90,
        )
        # Card squarely within torso region (x: 130-170, y: 150-190)
        card = DetectedCard(
            card_id=101,
            bbox=BoundingBox(x1=130.0, y1=150.0, x2=170.0, y2=190.0),
            confidence=0.85,
        )

        results = self.engine.associate([person], [card])

        self.assertIn(1, results)
        match = results[1]
        self.assertEqual(match.status, "associated")
        self.assertIsNotNone(match.card)
        self.assertEqual(match.card.card_id, 101)
        self.assertGreater(match.association_confidence, 0.5)

    def test_card_outside_torso_unassociated(self):
        person = TrackedPerson(
            track_id=2,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0),
            confidence=0.90,
        )
        # Card near feet (y: 280-320)
        card_at_feet = DetectedCard(
            card_id=102,
            bbox=BoundingBox(x1=130.0, y1=280.0, x2=170.0, y2=320.0),
            confidence=0.80,
        )

        results = self.engine.associate([person], [card_at_feet])

        self.assertIn(2, results)
        match = results[2]
        self.assertEqual(match.status, "unassociated")
        self.assertIsNone(match.card)

    def test_crowd_overlap_ambiguity(self):
        # Two people walking shoulder-to-shoulder with overlapping torsos
        person1 = TrackedPerson(
            track_id=1,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0),
            confidence=0.88,
        )
        person2 = TrackedPerson(
            track_id=2,
            bbox=BoundingBox(x1=140.0, y1=100.0, x2=240.0, y2=300.0),
            confidence=0.89,
        )
        # Card right in the overlapping corridor between person 1 and person 2
        card = DetectedCard(
            card_id=201,
            bbox=BoundingBox(x1=145.0, y1=160.0, x2=175.0, y2=200.0),
            confidence=0.85,
        )

        results = self.engine.associate([person1, person2], [card])

        self.assertIn(1, results)
        self.assertIn(2, results)
        # Crucial Crowd Mode requirement: must NOT make an arbitrary assignment
        self.assertEqual(results[1].status, "ambiguous_overlap")
        self.assertEqual(results[2].status, "ambiguous_overlap")
        self.assertIsNone(results[1].card)
        self.assertIsNone(results[2].card)

    def test_empty_people_or_cards(self):
        results_no_people = self.engine.associate([], [
            DetectedCard(1, BoundingBox(0, 0, 10, 10), 0.9)
        ])
        self.assertEqual(len(results_no_people), 0)

        person = TrackedPerson(1, BoundingBox(10, 10, 50, 50), 0.8)
        results_no_cards = self.engine.associate([person], [])
        self.assertIn(1, results_no_cards)
        self.assertEqual(results_no_cards[1].status, "unassociated")


if __name__ == "__main__":
    unittest.main()
