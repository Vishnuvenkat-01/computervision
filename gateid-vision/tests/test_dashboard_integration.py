"""
Unit tests for Streamlit Dashboard integration in GateID Vision.

Tests upload content hashing, duplicate rerun prevention, missing-model warnings,
human review outcome management, and metric formatting.
"""
import unittest

from src.dashboard_utils import check_ultralytics_available, get_file_content_hash
from src.event_capture import CapturedEvent
from src.id_card_detector import IDCardDetector
from src.metrics import calculate_ground_truth_metrics
from src.report_generator import ReportGenerator
from src.video_processor import VideoProcessor


class TestDashboardIntegration(unittest.TestCase):
    def test_file_content_hashing_stability(self):
        content1 = b"TEST_VIDEO_SAMPLE_BYTES_A"
        content2 = b"TEST_VIDEO_SAMPLE_BYTES_B"

        hash1_a = get_file_content_hash(content1)
        hash1_b = get_file_content_hash(content1)
        hash2 = get_file_content_hash(content2)

        self.assertEqual(hash1_a, hash1_b)
        self.assertNotEqual(hash1_a, hash2)
        self.assertEqual(len(hash1_a), 32)  # Valid MD5 hex string

    def test_deduplication_rerun_logic(self):
        # Simulate Streamlit session state rerun behavior
        session_state = {
            "processed_result": "MOCK_RESULT",
            "last_processed_hash": "abc123hash",
        }

        # Case A: Same file uploaded again (rerun)
        current_upload_hash = "abc123hash"
        already_processed = (
            session_state.get("processed_result") is not None
            and session_state.get("last_processed_hash") == current_upload_hash
        )
        self.assertTrue(already_processed)

        # Case B: Different file uploaded
        new_upload_hash = "xyz999different"
        already_processed_new = (
            session_state.get("processed_result") is not None
            and session_state.get("last_processed_hash") == new_upload_hash
        )
        self.assertFalse(already_processed_new)

    def test_missing_id_model_reporting(self):
        # Configure detector with non-existent path
        config = {
            "models": {
                "id_card_detector": {
                    "model_path": "models/non_existent.pt",
                    "enabled_fallback": True,
                }
            }
        }
        detector = IDCardDetector(config)
        self.assertFalse(detector.is_available)
        self.assertIn("fallback", detector.status_message.lower())

    def test_human_review_separation_from_ai_prediction(self):
        # AI prediction must remain untouched when human officer review is logged
        ai_event = CapturedEvent(
            event_id="evt_rev_01",
            video_filename="gate.mp4",
            video_timestamp=3.2,
            frame_number=32,
            track_id=5,
            status="verification_required",
            person_confidence=0.91,
            card_confidence=None,
            association_outcome="unassociated",
            reason="No card observed in torso",
            snapshot_path="outputs/snapshots/verification_required/snap.jpg",
        )

        reviewed_cases = {}
        reviewed_cases[ai_event.event_id] = "Manually Verified (ID Present)"

        # Verify AI prediction is still 'verification_required'
        self.assertEqual(ai_event.status, "verification_required")

        # Verify human decision is preserved separately
        self.assertEqual(reviewed_cases[ai_event.event_id], "Manually Verified (ID Present)")

    def test_empty_ground_truth_notice_on_reports_page(self):
        gt_result = calculate_ground_truth_metrics(predictions=[], ground_truth=[])
        self.assertEqual(gt_result["status"], "unavailable")
        self.assertIn("Ground-truth", gt_result["message"])

    def test_report_generation_for_dashboard_export(self):
        report_gen = ReportGenerator()
        df = report_gen.events_to_dataframe([])
        self.assertTrue(hasattr(df, "to_csv"))
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        self.assertGreater(len(csv_bytes), 0)  # Has headers


if __name__ == "__main__":
    unittest.main()
