# GateID Vision — Campus Entry ID Card Monitoring System
**AI-Based Hostel Gate ID Card Compliance Monitoring System Using Computer Vision**  
*HackZen 2.0 Hackathon Submission*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63.0-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://ultralytics.com/)
[![Tests](https://img.shields.io/badge/Tests-46%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **Notice & Disclaimer:** This application is an objective **decision-support tool** for campus and hostel security staff. It identifies ambiguous cases that warrant manual verification. It **does not perform facial recognition**, does not extract student personal records, and does not verify credential authenticity. A missing detection does not equate to confirmed non-compliance.

---

## 1. Problem Statement & Motivation

During morning lectures and evening curfew entry hours, campus and hostel security guards manually inspect hundreds of entering students for physical ID cards. This manual process causes:
* **Entry Bottlenecks:** Crowded queues spill onto walkways and adjacent roadways.
* **Guard Fatigue:** Human error increases significantly across long monitoring shifts.
* **Lack of Visibility:** No verifiable record of peak entry traffic or compliance patterns.

**GateID Vision** solves this by analyzing recorded gate surveillance videos, tracking entering individuals across crowded scenes, checking anatomical chest/torso boundaries for visible ID badges, capturing timestamped evidence with rate-limited cooldowns, and presenting an intuitive Streamlit dashboard to assist guards in verifying cases.

---

## 2. Key Innovations: Crowd Mode & Decision Support

* **Crowd Mode Disambiguation:** Unlike single-person turnstile systems, GateID Vision is engineered for multiple pedestrians entering simultaneously. If an ID card falls in an overlapping corridor between two individuals walking shoulder-to-shoulder, the system marks the match as `ambiguous_overlap` rather than making an arbitrary attribution.
* **Anatomical Torso Modeling:** Restricts ID search to the upper body ($15\%\text{--}65\%$ of person height), preventing false associations with objects held near feet or on backpacks.
* **3-State Temporal Classification:** Decisions are accumulated across a minimum observation window ($\ge 15$ frames) to prevent flicker. States:
  - `id_visible`: High-confidence card localized in torso.
  - `verification_required`: Clear image, tracked for $\ge 15$ frames, but no card detected.
  - `uncertain`: Blurred frame ($\text{Laplacian variance} < 60.0$), edge boundary clipping, crowd overlap ambiguity, short observation duration, or missing custom detector weights.
* **Instant Auto-Monitoring on Upload:** Dropping a video into the dashboard triggers monitoring automatically without button friction, featuring rerun deduplication to prevent accidental reprocessing.
* **Transparent Fallback Mode:** Operates safely when custom model weights are absent without fabricating detections or crashing.

---

## 3. Computer Vision Pipeline

```text
Recorded Video ──▶ Frame Ingestion (Scaled / Stride 2)
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
Person Detection (YOLOv8)      ID-Card Detection (Custom)
ByteTrack (Temporary IDs)      (Transparent Fallback if Missing)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
            Torso Association Engine
            (Chest zone & crowd overlap check)
                        │
                        ▼
            Temporal Status Classifier
            (id_visible | verification_required | uncertain)
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
Rate-Limited Event Capture       Annotated Video Writer
(3s cooldown / max 2 per person)  (mp4v / H264 codec)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
           Streamlit Dashboard (app.py)
```

---

## 4. Repository Structure

```text
gateid-vision/
├── app.py                         # Streamlit Dashboard application
├── config.yaml                    # System configuration & threshold parameters
├── requirements.txt               # Pinned Python dependencies
├── run_tests.py                   # Universal test runner (pytest / unittest)
├── create_zip.py                  # Submission packaging script
├── README.md                      # Complete project documentation
├── .gitignore                     # Git exclusion rules
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── config_loader.py           # Configuration parser & validation
│   ├── cv_compat.py               # Native OpenCV wrapper with test shim
│   ├── dashboard_utils.py         # Upload hashing & environment checks
│   ├── person_tracker.py          # YOLOv8 + ByteTrack person tracking
│   ├── id_card_detector.py        # Custom ID detector wrapper & fallback
│   ├── association.py             # Torso geometry & crowd ambiguity resolver
│   ├── status_classifier.py       # 3-state temporal classifier
│   ├── event_capture.py           # Rate-limited snapshot capture
│   ├── metrics.py                 # Real operational metrics & GT evaluation
│   ├── report_generator.py        # Audit-ready CSV report generator
│   └── video_processor.py         # End-to-end video processing engine
│
├── tests/
│   ├── test_association.py        # Spatial matching & ambiguity tests
│   ├── test_status_classifier.py  # Temporal observation transition tests
│   ├── test_event_capture.py      # Snapshot cooldown & cap tests
│   ├── test_id_card_detector.py   # Missing weights & fallback tests
│   ├── test_person_tracker.py     # ByteTrack tracking ID tests
│   ├── test_metrics.py            # Operational metrics & IoU validation
│   ├── test_report_generator.py   # CSV formatting & empty export tests
│   ├── test_video_processor.py    # Mocked pipeline & cancellation tests
│   ├── test_dashboard_integration.py # Streamlit hashing & review tests
│   └── test_config_and_pipeline_robustness.py # Corrupt inputs & edge cases
│
├── models/
│   └── README.md                  # Model weights documentation
│
├── data/
│   ├── README.md                  # Video data guidelines & privacy notice
│   └── sample_videos/             # Directory for authorized test recordings
│
├── outputs/
│   ├── snapshots/                 # Captured evidence grouped by category
│   │   ├── id_visible/
│   │   ├── verification_required/
│   │   └── uncertain/
│   ├── reports/                   # Exported CSV event reports
│   └── processed_videos/          # Exported annotated MP4 videos
│
└── docs/
    ├── PROJECT_DOCUMENTATION.md   # Comprehensive technical paper
    ├── MODEL_TRAINING.md          # Custom YOLO training & annotation guide
    ├── EVALUATION.md              # Benchmarking methodology & failure modes
    └── PRIVACY_AND_RESPONSIBLE_USE.md # Ethical AI & privacy governance
```

---

## 5. Installation & Setup (Windows PowerShell)

### Prerequisites
* Windows 10/11 (or macOS / Linux).
* Python 3.10, 3.11, or compatible.
* Git.

### Step 1: Clone or Navigate to Project
```powershell
cd "C:\Users\vishn\Downloads\Computer Vision\gateid-vision"
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 6. Running the Application & Tests

### Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
*Open your browser to `http://localhost:8501`.*

### Run the Automated Test Suite
```powershell
# Run with universal runner:
python run_tests.py

# Or run with pytest (if installed):
pytest tests -v
```

---

## 7. Model Status & Custom Training

* **Pretrained Person Tracker:** Automatically loads `yolov8n.pt` on first native run.
* **Custom ID-Card Detector:** Expects trained weights at `models/id_card_detector.pt`.
* **Fallback Behavior:** If custom weights are missing, GateID Vision operates transparently in **Person Tracking Mode**, flagging compliance-dependent cards as `Uncertain`. It **does not fabricate fake detections**.
* **Training Instructions:** Refer to [`docs/MODEL_TRAINING.md`](docs/MODEL_TRAINING.md) for complete instructions on dataset collection, annotation via CVAT/Roboflow, and fine-tuning YOLOv8.

---

## 8. Current Implementation Status & Limitations

### Working and Verified
* 46 automated unit, integration, and robustness tests passing ($100\%$).
* Multi-object tracking and temporary ID assignment logic.
* Anatomical torso bounding geometry with crowd overlap ambiguity resolution.
* 3-state temporal classifier with blur variance and boundary clipping filters.
* Rate-limited snapshot capture ($3.0\text{s}$ cooldown, max $2$ per person).
* Full multi-page Streamlit dark UI with instant-start upload and review queue.
* CSV audit report and annotated video exporter.

### Known Limitations
* **Native Video Decoding:** Requires native `opencv-python` and `ultralytics` in the active Python environment.
* **Trained ID-Card Weights:** Custom model `models/id_card_detector.pt` must be trained on institution-specific lanyards and badges to achieve production detection.
* **Fixed 2D Camera Perspective:** Severe multi-person crowd depth occlusions cannot be completely resolved without multi-camera setups or overhead perspectives.
