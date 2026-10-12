"""
Unit tests for the Metrics module in GateID Vision.

Tests operational metric aggregation and ground-truth validation evaluation.
"""
import unittest

from src.event_capture import CapturedEvent
from src.metrics import calculate_ground_truth_metrics, calculate_pipeline_metrics


class TestMetrics(unittest.TestCase):
    def test_pipeline_metrics_aggregation(self):
        events = [
            CapturedEvent("e1", "v.mp4", 1.0, 10, 1, "id_visible", 0.9, 0.8, "associated", "", ""),
            CapturedEvent("e2", "v.mp4", 2.0, 20, 2, "verification_required", 0.9, None, "unassociated", "", ""),
            CapturedEvent("e3", "v.mp4", 3.0, 30, 3, "uncertain", 0.8, None, "unassociated", "", ""),
        ]
        metrics = calculate_pipeline_metrics(
            events=events,
            unique_track_ids=3,
            total_frames=100,
            elapsed_time_sec=4.0,
            is_detector_available=True,
        )

        self.assertEqual(metrics.unique_tracked_people, 3)
        self.assertEqual(metrics.total_events_captured, 3)
        self.assertEqual(metrics.id_visible_count, 1)
        self.assertEqual(metrics.verification_required_count, 1)
        self.assertEqual(metrics.uncertain_count, 1)
        self.assertEqual(metrics.total_frames_processed, 100)
        self.assertAlmostEqual(metrics.approx_fps, 25.0)

    def test_ground_truth_metrics_without_gt_returns_unavailable(self):
        preds = [{"bbox": [10, 10, 50, 50]}]
        gt_result = calculate_ground_truth_metrics(preds, [])

        self.assertEqual(gt_result["status"], "unavailable")
        self.assertIsNone(gt_result["precision"])
        self.assertIsNone(gt_result["recall"])

    def test_ground_truth_metrics_with_annotations(self):
        preds = [
            {"bbox": [10, 10, 50, 50]},  # TP
            {"bbox": [100, 100, 150, 150]},  # FP
        ]
        ground_truth = [
            {"bbox": [12, 12, 52, 52]},  # Matches pred[0]
            {"bbox": [300, 300, 350, 350]},  # FN (missed)
        ]
        res = calculate_ground_truth_metrics(preds, ground_truth, iou_threshold=0.5)

        self.assertEqual(res["status"], "computed")
        self.assertEqual(res["true_positives"], 1)
        self.assertEqual(res["false_positives"], 1)
        self.assertEqual(res["false_negatives"], 1)
        self.assertAlmostEqual(res["precision"], 0.5)
        self.assertAlmostEqual(res["recall"], 0.5)
        self.assertAlmostEqual(res["f1_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
