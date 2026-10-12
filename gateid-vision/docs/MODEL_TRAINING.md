# Custom YOLO ID-Card Detector — Training & Annotation Guide
**GateID Vision — Campus Entry ID Card Monitoring**

> **Notice:** This document outlines the procedure for training a custom YOLOv8 model for college ID-card detection. In the current release of GateID Vision, custom weights are **not pre-packaged**, and the system operates in documented **Fallback Mode** until a dataset is collected and trained.

---

## 1. Overview & Objective

General-purpose object detection models (such as YOLOv8 trained on COCO) are trained on 80 broad categories (person, car, dog, etc.) and **cannot recognize college ID cards**.

To enable automated card detection in GateID Vision, you must train or fine-tune a YOLOv8 detector specifically on the visual appearance of campus credentials:
* Lanyards worn around the neck with plastic card holders.
* Badges clipped or pinned to shirts, pockets, or jackets.
* Cards held in hand during gate entry.

---

## 2. Responsible Dataset Collection & Ethics

### 2.1 Privacy Safeguards
1. **No Student PII Collection:** The detector is trained to recognize the **physical card object** (plastic pouch, lanyard strap, badge geometry), NOT student names, registration numbers, or phone numbers.
2. **Authorized Participant Staging:** Collect data using staged recordings with consenting student volunteers or synthetic proxies (dummy placeholder cards).
3. **Local Storage:** Keep all training imagery on secure local workstations; do not upload identifiable student portraits to public web services.

### 2.2 Dataset Diversity Requirements
To ensure the detector performs reliably across varied entrance environments, the dataset must include:
* **Distance Variations:** Close-ups (1–2 meters), medium distance (3–5 meters), and distant views (6–8 meters).
* **Lighting Conditions:** Bright morning sunlight, overcast afternoons, evening gate spotlights, and backlighting.
* **Occlusions:** Cards partially covered by backpack straps, jacket lapels, folded arms, or overlapping pedestrians.
* **Negative Samples:** People with no visible cards, people wearing scarves or headphones (to prevent false positive badge detections).
* **Card Orientations:** Cards hanging straight, tilted sideways, or flipped horizontally.

---

## 3. Annotation Protocol

### 3.1 Tools
Recommended annotation tools:
* **CVAT (Computer Vision Annotation Tool):** Open-source, web-based or local Docker container.
* **LabelImg / Label Studio:** Lightweight local tools for bounding box labeling.
* **Roboflow:** Online annotation platform with dataset export in YOLOv8 format.

### 3.2 Labeling Guidelines
* **Class Definition:** Single class: `id_card` (Class Index `0`).
* **Bounding Box Tightness:** Draw tight bounding boxes enclosing the plastic card holder and badge face.
* **Lanyard Handling:** If the lanyard is visible but the card is tucked into a jacket, label only if at least $30\%$ of the card body is visible.

---

## 4. Dataset Directory Structure

Structure your dataset in the standard Ultralytics YOLO format:

```text
id_card_dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── labels/
│       ├── img_001.txt
│       └── img_002.txt
├── val/
│   ├── images/
│   │   └── img_101.jpg
│   └── labels/
│       └── img_101.txt
└── test/
    ├── images/
    │   └── img_201.jpg
    └── labels/
        └── img_201.txt
```

### 4.1 Annotation Format (YOLO TXT)
Each label file contains one row per detected object:
```text
<class_id> <x_center> <y_center> <width> <height>
```
*(All coordinates normalized between 0.0 and 1.0).*

Example `train/labels/img_001.txt`:
```text
0 0.4852 0.4120 0.0820 0.1250
```

### 4.2 Dataset Configuration (`data.yaml`)
Create `data.yaml` inside your dataset directory:

```yaml
path: ../id_card_dataset # Dataset root path
train: train/images
val: val/images
test: test/images

names:
  0: id_card
```

---

## 5. Training the Model

### 5.1 Environment Setup
Ensure Ultralytics and PyTorch are installed:
```powershell
pip install ultralytics torch torchvision
```

### 5.2 Training Command (CLI)
Fine-tune YOLOv8 Nano (`yolov8n.pt`) for fast real-time edge performance:

```powershell
yolo detect train `
  data=id_card_dataset/data.yaml `
  model=yolov8n.pt `
  epochs=100 `
  imgsz=640 `
  batch=16 `
  device=0 `
  workers=4 `
  project=gateid_training `
  name=id_detector_run1
```

*(On Apple Silicon Mac, use `device=mps`. On CPU-only systems, use `device=cpu`).*

### 5.3 Training via Python Script
```python
from ultralytics import YOLO

# Load pretrained base model
model = YOLO("yolov8n.pt")

# Train on custom dataset
results = model.train(
    data="id_card_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="auto",
    project="gateid_training",
    name="id_detector_v1",
    patience=20,  # Early stopping if validation loss plateaus
)
```

---

## 6. Model Evaluation & Benchmark Metrics

Evaluate the trained checkpoint on the held-out test split:

```powershell
yolo detect val model=gateid_training/id_detector_v1/weights/best.pt data=id_card_dataset/data.yaml split=test
```

### Interpreting Evaluation Metrics
* **mAP@50 (Mean Average Precision at IoU=0.50):** Target $\ge 0.85$ for reliable gate surveillance.
* **Precision:** Ratio of true detected cards to all card detections. High precision prevents false alarms for lanyards without cards.
* **Recall:** Ratio of detected cards to actual cards present. Target $\ge 0.80$ to minimize missed badges.
* **Confusion Matrix:** Inspect background false positives (e.g., shirt logos or belt buckles mistaken for cards).

---

## 7. Deploying to GateID Vision

Once training is complete:

1. Copy the best weights checkpoint:
   ```powershell
   copy "gateid_training\id_detector_v1\weights\best.pt" "gateid-vision\models\id_card_detector.pt"
   ```
2. Verify `config.yaml` points to the weights file:
   ```yaml
   models:
     id_card_detector:
       model_path: "models/id_card_detector.pt"
       confidence_threshold: 0.40
       iou_threshold: 0.45
       enabled_fallback: true
   ```
3. Launch the dashboard (`streamlit run app.py`). The **System Status** page will transition from *Fallback Mode* to **🟢 Custom Model Loaded**.
