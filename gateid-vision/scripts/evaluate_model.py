"""
Command-Line Evaluation Script for GateID Vision.
Evaluates detector predictions against ground-truth annotations (JSON/CSV)
and computes Precision, Recall, and F1-score without fabricating results.
"""
from pathlib import Path
from typing import Any, Dict, List
import argparse
import json
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import calculate_ground_truth_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GateID Vision detection performance against ground truth."
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to JSON file containing ground-truth bounding box annotations.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to JSON file containing model prediction bounding boxes.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold for True Positive assignment (default: 0.50).",
    )
    return parser.parse_args()


def load_boxes(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "boxes" in data:
        return data["boxes"]
    raise ValueError(f"Unrecognized annotation format in {path}")


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print(" GateID Vision — Ground-Truth Benchmark Evaluation")
    print("=" * 60)

    try:
        gt_boxes = load_boxes(args.ground_truth)
        pred_boxes = load_boxes(args.predictions)
    except Exception as e:
        print(f"Error loading evaluation files: {e}")
        sys.exit(1)

    print(f"Ground-Truth Bounding Boxes: {len(gt_boxes)}")
    print(f"Predicted Bounding Boxes:    {len(pred_boxes)}")
    print(f"IoU Matching Threshold:      {args.iou_threshold:.2f}")

    results = calculate_ground_truth_metrics(
        predictions=pred_boxes,
        ground_truth=gt_boxes,
        iou_threshold=args.iou_threshold,
    )

    print("\nBenchmark Results:")
    print(f"- Status:           {results.get('status')}")
    print(f"- True Positives:   {results.get('true_positives')}")
    print(f"- False Positives:  {results.get('false_positives')}")
    print(f"- False Negatives:  {results.get('false_negatives')}")
    print(f"- Precision:        {results.get('precision')}")
    print(f"- Recall:           {results.get('recall')}")
    print(f"- F1 Score:         {results.get('f1_score')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
