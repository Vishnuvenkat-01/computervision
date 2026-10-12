# Evaluation Methodology & Benchmark Guide
**GateID Vision — Campus Entry ID Card Monitoring**

---

## 1. Testing Framework Overview

The evaluation of GateID Vision is structured across three verification tiers:
1. **Tier 1: Unit & Regression Testing:** Automated unit testing of geometric, temporal, and reporting logic with mocked dependencies.
2. **Tier 2: Pipeline Integration Testing:** Multi-component verification covering frame pipelines, rate limiting, and dashboard session states.
3. **Tier 3: Empirical Ground-Truth Validation:** Scientific benchmarking against human-annotated video test sets.

---

## 2. Unit and Integration Test Results

The automated test suite contains **46 test cases** located in `tests/`:

| Test Module | Scope & Covered Functionality | Test Count | Result |
|---|---|:---:|:---:|
| `test_association.py` | Torso bounding box geometry, overlap containment, crowd ambiguity handling | 5 | **Passed** |
| `test_status_classifier.py` | Blur variance filter, edge clipping, temporal observation window, smooth transitions | 6 | **Passed** |
| `test_event_capture.py` | Filesystem setup, snapshot cooldowns, per-person capture caps | 2 | **Passed** |
| `test_id_card_detector.py` | Missing weights fallback, non-fabrication guarantees, mock inference parsing | 4 | **Passed** |
| `test_person_tracker.py` | YOLO ByteTrack ID parsing, track ID recovery, empty frame resilience | 3 | **Passed** |
| `test_metrics.py` | Operational count aggregation, ground-truth IoU evaluation | 3 | **Passed** |
| `test_report_generator.py` | Empty DataFrame safety, populated DataFrames, CSV formatting, summary counts | 5 | **Passed** |
| `test_video_processor.py` | Invalid paths, mocked end-to-end processing, stop/cancel callbacks | 3 | **Passed** |
| `test_dashboard_integration.py` | Upload MD5 hashing, rerun deduplication, human review separation, export | 6 | **Passed** |
| `test_config_and_pipeline_robustness.py` | Config YAML validation, 0-byte corrupt video handling, zero-area boxes, state resets | 9 | **Passed** |
| **Total** | | **46** | **100% Pass** |

To execute the full suite:
```powershell
python run_tests.py
```

---

## 3. Ground-Truth Benchmarking Protocol

### 3.1 Why Operational Counts Are Not Accuracy Metrics
A common flaw in hackathon submissions is treating observed detections (e.g., *"14 ID Visible / 20 Tracked Persons"*) as an $80\%$ model accuracy score.
* Detections represent **system outputs**, not ground-truth reality.
* To measure true scientific accuracy, predictions must be compared against **human-annotated ground-truth labels**.

### 3.2 Required Annotation Datasets for Formal Benchmark
To run a formal benchmark:
1. **Person Tracks Ground Truth:** Frame-by-frame bounding boxes $[x_1, y_1, x_2, y_2]$ and tracking ID labels for every visible entrant.
2. **ID-Card Presence Ground Truth:** Frame-by-frame bounding boxes for visible ID cards.
3. **Entrant Ground Truth Status:** Ground-truth classification label for each entrant:
   - `id_present_and_visible`
   - `id_present_but_occluded`
   - `no_id_worn`

### 3.3 Quantitative Metrics Definitions

#### A. ID Card Detection Metrics
* **Intersection over Union (IoU):**
  $$\text{IoU} = \frac{\text{Area}(B_{pred} \cap B_{gt})}{\text{Area}(B_{pred} \cup B_{gt})}$$
  A detection is considered a True Positive ($TP$) if $\text{IoU} \ge 0.50$.
* **Precision:**
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Recall:**
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* **F1 Score:**
  $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### B. Multi-Object Tracking Metrics
* **MOTA (Multiple Object Tracking Accuracy):** Evaluates tracker consistency considering false positives, missed targets, and track identity switches.
* **ID Switches (IDSW):** Count of times a tracked person's temporary ID changes between successive frames due to occlusion.

#### C. Association Accuracy
* **Torso Association Accuracy:** Percentage of correctly attributed ID cards to their true wearers in multi-person entrance scenes.

---

## 4. Current Metric Availability Status

| Metric | Status in GateID Vision | Reason |
|---|:---:|---|
| Unique Tracked Entrants | **Available** | Directly counted from active ByteTrack tracker IDs |
| Confirmed ID Visible Flags | **Available** | Directly counted from temporal classifier results |
| Verification Prompts | **Available** | Directly counted from temporal classifier results |
| Uncertain Observations | **Available** | Directly counted from blur, clipping, and fallback flags |
| Processing Speed (FPS) | **Available** | Directly measured from elapsed wall-clock processing time |
| Precision, Recall, F1 Score | ⏳ *Requires Ground Truth* | Cannot be fabricated; requires annotated test videos |
| MOTA / MOTP | ⏳ *Requires Ground Truth* | Requires annotated multi-object tracking ground truth |

---

## 5. Failure Modes & Edge Case Analysis

1. **Motion Blur:** Fast walking speeds or low camera shutter speeds blur small ID text and lanyards. *Mitigation: Laplacian variance filter shifts state to `Uncertain`.*
2. **Torso Edge Clipping:** Entrants crossing the camera edge have incomplete chests. *Mitigation: Frame boundary margin threshold ($5\%$) shifts state to `Uncertain`.*
3. **Shoulder-to-Shoulder Crowds:** Heavy overlap in 2D perspective causes one person's lanyard to fall visually within an adjacent person's torso. *Mitigation: Dual-candidate ambiguity detector flags `ambiguous_overlap` rather than attributing arbitrarily.*
4. **Card Reversal / Blank Backside:** Cards flipping backwards show blank plastic faces. *Mitigation: Custom detector training must include flipped card backs.*
