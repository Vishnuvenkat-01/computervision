# GateID Vision — Technical Architecture & System Documentation
**HackZen 2.0 Hackathon Submission**  
**Project Title:** GateID Vision — AI-Based Campus Entry ID Card Compliance Monitoring System  

---

## 1. Abstract

Campus and hostel security teams face severe operational bottlenecks during peak entry hours when hundreds of students enter gates simultaneously. Manual checking creates entry congestion, increases staff cognitive fatigue, and leads to inconsistent compliance tracking. **GateID Vision** is a crowd-aware computer vision decision-support system designed to automate the visual monitoring of college ID cards from recorded entry video footage. 

By integrating YOLOv8 person detection with ByteTrack multi-object tracking, custom ID-card localization, anatomical torso bounding geometry, and a 3-state temporal classifier (`id_visible`, `verification_required`, and `uncertain`), GateID Vision flags ambiguous or unverified cases for manual security staff verification. Crucially, the system does not perform facial recognition, does not harvest student identities, and operates strictly as an objective decision-support assistant rather than an automated disciplinary mechanism.

---

## 2. Problem Statement & Operational Context

### 2.1 The Traditional Gate Check Challenge
In universities, colleges, and residential hostels, physical ID cards attached to lanyards or pinned badges serve as the primary credential for campus entry. During morning lecture rush hours and evening curfew entry windows, security guards manually inspect each entrant. This manual process suffers from:
1. **Severe Peak Congestion:** Physical inspections slow entry throughput to a crawl, creating pedestrian queues extending into public roads.
2. **Cognitive Fatigue & Inconsistency:** Guards inspecting 1,500+ entrants across a 45-minute window experience high error rates and selective enforcement.
3. **Absence of Auditability:** Paper logbooks or verbal checks leave no verifiable audit trail of entry trends or compliance bottlenecks.

### 2.2 Proposed Solution
GateID Vision provides a non-invasive, video-based monitoring workflow:
* Ingests standard recorded gate surveillance videos (MP4, AVI, MOV, MKV).
* Tracks multiple individuals simultaneously across crowded frames.
* Restricts card search to anatomical chest/torso boundaries.
* Distinguishes between crowd occlusion, image blur, boundary clipping, and genuine card absence.
* Automatically captures timestamped snapshots with evidence metadata.
* Provides an interactive Streamlit dashboard featuring an instant-trigger monitoring pipeline and human-in-the-loop review queue.

---

## 3. System Architecture & Component Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            GateID Vision Pipeline                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                [Recorded Video]
                                       │
                                       ▼
                       [cv2.VideoCapture Frame Ingestion]
                     (Resolution scale & frame stride: 2)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
  [PersonTracker (YOLOv8)]                              [IDCardDetector (Custom)]
  - Detects COCO class 0 (person)                       - Targeted card detection
  - ByteTrack assigns transient IDs                     - Transparent fallback if missing
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                          [AssociationEngine]
                          - Anatomical chest/torso zone calculation
                          - Containment & IoU evaluation
                          - Crowd overlap corridor ambiguity resolver
                                       │
                                       ▼
                          [StatusClassifier]
                          - Laplacian blur variance filter (< 60.0)
                          - Frame edge clipping detection (margin: 5%)
                          - Temporal observation window (>= 15 frames)
                          - States: 'id_visible' | 'verification_required' | 'uncertain'
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
    [EventCapturer]                                     [Frame Annotator & Writer]
    - Cooldown check (3.0s)                             - Color-coded bounding boxes
    - Per-person cap (max 2)                            - Anatomical torso guide
    - Snapshot & crop write                             - Header summary overlay
    - Structured metadata                               - Codec: mp4v / H264
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                        [Streamlit Dashboard (app.py)]
                        - Auto-monitoring on file drop
                        - Real-time progress visualization
                        - Human Verification Queue
                        - Audit Report Exporter (CSV & MP4)
```

---

## 4. Module Specifications

### 4.1 Person Detection & Multi-Object Tracking (`src/person_tracker.py`)
* **Detection Model:** Pretrained Ultralytics YOLOv8n (`yolov8n.pt`), restricted strictly to `class=0` (Person).
* **Multi-Object Tracking:** ByteTrack tracker (`bytetrack.yaml`). Associates detections across successive frames based on motion consistency and Kalman filtering.
* **Transient Track Identities:** Assigns temporary integer IDs (`Person 1`, `Person 2`). Tracking IDs are cleared upon video completion and never represent permanent identities.

### 4.2 Custom ID-Card Detection (`src/id_card_detector.py`)
* **Architecture:** Targeted YOLOv8 object detector trained specifically on college lanyards, hanging badges, and pocket cards.
* **Fallback Behavior:** If custom model weights (`models/id_card_detector.pt`) are absent, the detector does not crash or fabricate detections. Instead, it activates a documented **Fallback Mode**:
  - `is_available` returns `False`.
  - Detection queries return empty lists `[]`.
  - Status classification shifts card-dependent decisions to `Uncertain`, reporting that model weights are missing.

### 4.3 Spatial & Anatomical Association Engine (`src/association.py`)
* **Torso Boundary Modeling:** Rather than matching ID cards to an entire human bounding box (which causes false associations with cards at shoes or bags), the engine restricts association to the chest/torso zone:
  $$\text{Torso } Y_{min} = Y_1 + (H \times 0.15)$$
  $$\text{Torso } Y_{max} = Y_1 + (H \times 0.65)$$
  $$\text{Torso } X_{margin} = W \times 0.10$$
* **Crowd Overlap Ambiguity Handling:** When two individuals walk shoulder-to-shoulder, their 2D camera projections overlap. If a card lies inside the overlapping corridor, the association engine flags the match as `ambiguous_overlap` rather than guessing arbitrarily.

### 4.4 Temporal Status Classifier (`src/status_classifier.py`)
Decisions are never made on a single, isolated frame. A temporal state machine tracks history:
1. **`id_visible`**: Confirmed ID card detected inside a person's torso region with high detector confidence. Includes temporal smoothing against momentary card turns.
2. **`verification_required`**: Person tracked for at least $\ge 15$ frames (~1 second at 25-30 FPS) with clear illumination, sharp focus, unclipped torso, but no card detected. **This is a review prompt for staff, not proof of absence.**
3. **`uncertain`**: Image blur (Laplacian variance $< 60.0$), torso clipped near frame boundaries ($< 5\%$ edge margin), crowd overlap ambiguity, observation window $< 15$ frames, or missing custom model weights.

### 4.5 Event Capture & Rate Limiting (`src/event_capture.py`)
* Automatically captures full snapshots and localized crops to `outputs/snapshots/{status}/`.
* **Rate-Limiting Cooldown:** Requires a minimum $3.0\text{s}$ interval between snapshots of the same tracked individual in the same status.
* **Per-Person Cap:** Limits captures to a maximum of $2$ snapshots per individual to eliminate redundant disk writes.

### 4.6 Reporting & Operational Metrics (`src/report_generator.py` & `src/metrics.py`)
* Produces CSV event logs containing: `event_id`, `video_filename`, `video_timestamp`, `frame_number`, `track_id`, `status`, `person_confidence`, `card_confidence`, `association_outcome`, `reason`, `snapshot_path`, `crop_path`, and `manual_review_outcome`.
* Measures actual operational throughput (FPS) and distinct visitor counts.
* Enforces ground-truth validation separation: Precision, Recall, and F1 are only reported when verified annotations are provided.

---

## 5. Decision Logic & Mathematical Formulation

### 5.1 Image Sharpness Assessment
Frame sharpness is calculated using the variance of the 2D Laplacian operator:
$$\text{Sharpness} = \sigma^2 \left( \nabla^2 I_{gray} \right)$$
Frames with $\text{Sharpness} < 60.0$ are classified as `Uncertain (Motion Blur)`.

### 5.2 Torso Containment Ratio
For card bounding box $B_{card}$ and torso bounding box $B_{torso}$:
$$\text{Containment} = \frac{\text{Area}(B_{card} \cap B_{torso})}{\text{Area}(B_{card})}$$
A card is considered a candidate match if $\text{Containment} \ge 0.25$ or if its geometric center $(cx, cy) \in B_{torso}$.

---

## 6. Verification and Human-in-the-Loop Protocol

GateID Vision implements strict human-in-the-loop safeguards:
* Security officers inspect flagged cases in the **Verification Queue**.
* Officers can log review outcomes:
  1. *Confirmed Physically (ID Present)*
  2. *Flagged for Physical Gate Check*
  3. *Dismissed Observation*
* Officer reviews are stored separately from raw AI telemetry in session state, ensuring audit trail integrity.

---

## 7. Current Implementation Status vs. Roadmap

| Feature / Component | Status | Verification Method |
|---|:---:|---|
| Multi-Object Person Tracking | **Implemented** | Automated tests with ByteTrack integration |
| Anatomical Torso Association | **Implemented** | Geometric unit test suite (`test_association.py`) |
| Crowd Overlap Ambiguity Resolver | **Implemented** | Dual-candidate collision tests |
| 3-State Temporal Classifier | **Implemented** | Window transition tests (`test_status_classifier.py`) |
| Cooldown & Snapshot Event Capture | **Implemented** | Filesystem rate-limiting tests (`test_event_capture.py`) |
| Full Streamlit Dark UI Dashboard | **Implemented** | Tested with auto-trigger upload and rerun deduplication |
| Transparent Fallback Mode | **Implemented** | Tested against absent weights (`models/id_card_detector.pt`) |
| Custom ID-Card Model Weights | ⏳ *Pending Dataset* | Architecture ready; requires campus badge training |
| Real-Video Native Decoding | ⏳ *Pending Native Dep* | Requires `opencv-python` in active environment |

---

## 8. Conclusion

GateID Vision delivers a modern, robust, and ethical computer vision pipeline for college gate compliance monitoring. By emphasizing decision support over punitive automation and implementing multi-person crowd disambiguation, the system provides campus security teams with an actionable, reliable, and auditable tool.
