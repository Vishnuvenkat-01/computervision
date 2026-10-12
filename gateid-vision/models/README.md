# Models Directory

This directory holds the deep learning weights used by GateID Vision.

## Required Models

1. **Person Detector & Tracker (COCO Pretrained)**
   - Model: `yolov8n.pt` (or `yolov8s.pt`)
   - Source: Automatically downloaded from Ultralytics on first run.
   - Purpose: Detects persons (`class 0`) and assigns temporary tracking IDs via ByteTrack.

2. **Custom ID Card Detector**
   - Expected Path: `models/id_card_detector.pt`
   - Purpose: Specifically trained YOLOv8 model for detecting college ID cards (lanyards, pinned badges, cards held in hand).
   - Status: **Trained model weights must be placed here.**
   - Graceful Fallback: If `models/id_card_detector.pt` is missing, GateID Vision operates in **Person Tracking & Review Mode**, displaying a transparent warning badge on the dashboard. It does not fabricate detections or report fake card statuses.

## Training Your Custom Model

Refer to [`docs/MODEL_TRAINING.md`](../docs/MODEL_TRAINING.md) for full instructions on:
- Collecting crowd images with various lighting and occlusions.
- Annotating ID cards using tools such as CVAT or Roboflow.
- Fine-tuning YOLOv8 on your dataset.
- Exporting and saving the weights to this directory.
