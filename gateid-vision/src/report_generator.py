"""
Report generator module for GateID Vision.

Produces structured CSV event reports and pandas DataFrames from captured
monitoring events. Handles empty event datasets safely without errors.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import logging
import pandas as pd

from src.event_capture import CapturedEvent

logger = logging.getLogger(__name__)

REPORT_COLUMNS = [
    "event_id",
    "video_filename",
    "video_timestamp",
    "frame_number",
    "track_id",
    "status",
    "person_confidence",
    "card_confidence",
    "association_outcome",
    "reason",
    "snapshot_path",
    "crop_path",
    "manual_review_outcome",
]


class ReportGenerator:
    """
    Exports captured surveillance decision events into audit-ready CSV reports
    and analytical DataFrames.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("output", {})
        self.reports_dir = Path(cfg.get("reports_directory", "outputs/reports"))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def events_to_dataframe(self, events: List[CapturedEvent]) -> pd.DataFrame:
        """
        Converts a list of CapturedEvent instances into a pandas DataFrame.
        Returns an empty DataFrame with valid column names if events is empty.
        """
        if not events:
            return pd.DataFrame(columns=REPORT_COLUMNS)

        records = []
        for e in events:
            records.append({
                "event_id": e.event_id,
                "video_filename": e.video_filename,
                "video_timestamp": e.video_timestamp,
                "frame_number": e.frame_number,
                "track_id": e.track_id,
                "status": e.status,
                "person_confidence": e.person_confidence,
                "card_confidence": e.card_confidence if e.card_confidence is not None else "",
                "association_outcome": e.association_outcome,
                "reason": e.reason,
                "snapshot_path": e.snapshot_path,
                "crop_path": e.crop_path or "",
                "manual_review_outcome": e.manual_review_outcome or "pending",
            })

        return pd.DataFrame(records, columns=REPORT_COLUMNS)

    def export_csv(
        self,
        events: List[CapturedEvent],
        filename: Optional[str] = None,
        custom_path: Optional[Path] = None,
    ) -> Path:
        """
        Writes events to a CSV file.
        If events list is empty, writes header line to preserve report schema.
        """
        if custom_path is not None:
            out_file = Path(custom_path)
        else:
            fname = filename or "gateid_event_report.csv"
            out_file = self.reports_dir / fname

        out_file.parent.mkdir(parents=True, exist_ok=True)

        df = self.events_to_dataframe(events)
        df.to_csv(out_file, index=False, encoding="utf-8")
        logger.info("Successfully exported event report to %s (%d events)", out_file, len(events))

        return out_file

    def get_summary_statistics(self, events: List[CapturedEvent]) -> Dict[str, Any]:
        """
        Calculates category breakdown from recorded events.
        """
        total = len(events)
        if total == 0:
            return {
                "total_events": 0,
                "id_visible": 0,
                "verification_required": 0,
                "uncertain": 0,
                "unique_persons_flagged": 0,
            }

        counts = {
            "id_visible": sum(1 for e in events if e.status == "id_visible"),
            "verification_required": sum(1 for e in events if e.status == "verification_required"),
            "uncertain": sum(1 for e in events if e.status == "uncertain"),
        }
        unique_persons = len(set(e.track_id for e in events))

        return {
            "total_events": total,
            "id_visible": counts["id_visible"],
            "verification_required": counts["verification_required"],
            "uncertain": counts["uncertain"],
            "unique_persons_flagged": unique_persons,
        }
