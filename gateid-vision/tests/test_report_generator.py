"""
Unit tests for the ReportGenerator in GateID Vision.

Tests DataFrame creation, CSV export, summary statistic calculation,
and empty event list handling.
"""
from pathlib import Path
import tempfile
import unittest

from src.event_capture import CapturedEvent
from src.report_generator import REPORT_COLUMNS, ReportGenerator


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            "output": {
                "reports_directory": self.temp_dir.name,
            }
        }
        self.generator = ReportGenerator(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_events_dataframe(self):
        df = self.generator.events_to_dataframe([])
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), REPORT_COLUMNS)

    def test_populated_events_dataframe(self):
        events = [
            CapturedEvent(
                event_id="evt_01",
                video_filename="test.mp4",
                video_timestamp=2.5,
                frame_number=25,
                track_id=1,
                status="id_visible",
                person_confidence=0.91,
                card_confidence=0.88,
                association_outcome="associated",
                reason="Card clearly visible",
                snapshot_path="outputs/snapshots/id_visible/snap1.jpg",
            ),
            CapturedEvent(
                event_id="evt_02",
                video_filename="test.mp4",
                video_timestamp=4.0,
                frame_number=40,
                track_id=2,
                status="verification_required",
                person_confidence=0.95,
                card_confidence=None,
                association_outcome="unassociated",
                reason="No card detected",
                snapshot_path="outputs/snapshots/verification_required/snap2.jpg",
            ),
        ]
        df = self.generator.events_to_dataframe(events)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["event_id"], "evt_01")
        self.assertEqual(df.iloc[1]["status"], "verification_required")

    def test_export_csv_creates_file(self):
        events = [
            CapturedEvent(
                event_id="evt_10",
                video_filename="demo.mp4",
                video_timestamp=1.0,
                frame_number=10,
                track_id=3,
                status="uncertain",
                person_confidence=0.85,
                card_confidence=None,
                association_outcome="unassociated",
                reason="Observing window",
                snapshot_path="outputs/snapshots/uncertain/snap3.jpg",
            )
        ]
        out_csv = self.generator.export_csv(events, filename="test_report.csv")
        self.assertTrue(out_csv.exists())
        self.assertGreater(out_csv.stat().st_size, 0)

        # Read back and verify content
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # Header + 1 data line
        self.assertIn("evt_10", lines[1])

    def test_export_empty_csv(self):
        out_csv = self.generator.export_csv([], filename="empty_report.csv")
        self.assertTrue(out_csv.exists())
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)  # Header only

    def test_summary_statistics(self):
        events = [
            CapturedEvent(
                event_id=f"e{i}",
                video_filename="v.mp4",
                video_timestamp=float(i),
                frame_number=i,
                track_id=1 if i < 2 else 2,
                status="id_visible" if i == 0 else "verification_required",
                person_confidence=0.9,
                card_confidence=0.8 if i == 0 else None,
                association_outcome="associated" if i == 0 else "unassociated",
                reason="",
                snapshot_path="",
            )
            for i in range(3)
        ]
        stats = self.generator.get_summary_statistics(events)
        self.assertEqual(stats["total_events"], 3)
        self.assertEqual(stats["id_visible"], 1)
        self.assertEqual(stats["verification_required"], 2)
        self.assertEqual(stats["unique_persons_flagged"], 2)


if __name__ == "__main__":
    unittest.main()
