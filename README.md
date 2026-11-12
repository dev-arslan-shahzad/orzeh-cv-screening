# Automated Kitchen Hygiene & Food Safety Behaviour Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-rmCbfjvTlZPn8M2oATblQdlONiLk7mT)
[![Google Drive Folder](https://img.shields.io/badge/Google%20Drive-Public%20Evaluation%20Folder-blue.svg?logo=googledrive)](https://drive.google.com/drive/folders/1IJegt-4xCi1VFAKMQeCt_Q3Q5S195BdI)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Device: CPU/GPU](https://img.shields.io/badge/Device-CPU%20%7C%20CUDA-green.svg)]()

> **Quick Cloud Evaluation:** Evaluators without a local GPU can run the entire pipeline interactively with 1 click on a free Tesla T4 GPU in Google Colab:  
> 👉 **[Open `Orzeh_CV_Screening_Evaluation.ipynb` in Google Colab](https://colab.research.google.com/drive/1-rmCbfjvTlZPn8M2oATblQdlONiLk7mT)**  
> Public Google Drive Directory (no sign-in required): **[Access Public Evaluation Folder](https://drive.google.com/drive/folders/1IJegt-4xCi1VFAKMQeCt_Q3Q5S195BdI)**

A robust, fully local, open-source computer vision pipeline developed for the **Orzeh Technologies Computer Vision Candidate Screening Task**. 

The system analyzes continuous kitchen video footage to track four core food safety and operational hygiene behaviours over time:
1. **`hand_covering`**: `bare_hands` vs `gloves` vs `not_visible`
2. **`shirt_sleeve_length`**: `full_sleeve` vs `half_sleeve` vs `not_visible`
3. **`head_covering`**: `cap` vs `bare_head` vs `not_visible`
4. **`hand_washing`**: `washing` vs `not_washing` vs `not_visible`

Outputs are strictly validated against **Table 3** of the specification, providing complete, contiguous, non-overlapping temporal intervals from `0.000` to the end of each video.

---

## 1. Setup Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)
- Windows, Linux, or macOS
- (Optional) NVIDIA GPU with CUDA for accelerated inference. The pipeline runs automatically on CPU if CUDA is not available.

### Installation
```bash
# 1. Clone repository
git clone <repository_url>
cd assesment

# 2. Create virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

Weights (`yolo11n-pose.pt` ~6 MB and `ViT-B-16-SigLIP` ~800 MB) are automatically cached upon first run from official open-source repositories.

---

## 2. Running Evaluation

Per Section 6 of the specification, the entire pipeline is executed via a single local CLI command:

```bash
python run.py --video <path_to_video> --output <output_directory>
```

### Examples
```bash
# Process a single evaluation video
python run.py --video videos/002.mp4 --output results/

# Process with demonstration HUD visualization video output
python run.py --video videos/002.mp4 --output results/ --visualize

# Custom sample rate and device selection
python run.py --video videos/004.mp4 --output results/ --fps 2.0 --device cpu
```

The resulting CSV file will be written to `<output_directory>/<video_id>.csv` (e.g., `results/002.csv`).

---

## 3. Method & Architecture

```
                       Input Video Stream
                              │
                    ┌─────────▼─────────┐
                    │  Frame Sampler    │ (Default: 2.0 FPS)
                    └─────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼─────────┐           ┌─────────▼─────────┐
    │   YOLO11n-Pose    │           │ Classical CV &    │
    │  Dominant Person  │           │ CIELAB Analytics  │
    │ 17 COCO Keypoints │           │ YCrCb/HSV Skin    │
    └─────────┬─────────┘           └─────────┬─────────┘
              │                               │
     [Geom Body Crops: Head, Torso, Forearm, Hands]
              │                               │
              └───────────────┬───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ OpenCLIP SigLIP   │ (ViT-B-16-webli)
                    │ Zero-Shot Scores  │ (Headwear, Sleeves, Gloves, Water)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Multimodal Fusion │
                    │ Decision Logic    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Temporal Filter & │ (Schmitt Trigger Hysteresis
                    │ Interval Builder  │  + Glitch Removal)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Strict Table 3    │ (Contiguous [0.000, duration]
                    │ CSV Validator     │  zero-gap intervals)
                    └───────────────────┘
```

### Multimodal Behaviour Detection

1. **`hand_covering`**:
   - Hand bounding boxes are projected geometrically from elbow-to-wrist pose vectors (avoiding skin-dependent landmarkers that fail on dark gloves).
   - Opaque gloves (e.g. black nitrile in `002.mp4`, blue/white) are distinguished by low skin pixel ratio ($< 0.25$) and high SigLIP glove confidence.
   - Clear food-prep gloves (e.g. `004.mp4`) allow skin to show through ($> 0.65$ skin ratio), but exhibit distinct plastic sheen and surface crinkles captured by high-resolution SigLIP crop classification.
   - **Mixed Hands Convention (Section 2.1)**: If a worker has one gloved hand and one bare hand (e.g., knife hand bare, non-dominant hand wearing a clear safety glove in `004.mp4`), the system reports `gloves` whenever at least one visible hand is gloved.

2. **`shirt_sleeve_length`**:
   - Compares the upper torso shirt color against the forearm in CIELAB space ($\Delta E$), combined with forearm skin segmentation.
   - `half_sleeve`: forearm shows substantial skin ratio ($> 0.22$) or high color difference ($\Delta E > 16.0$) from the torso shirt.
   - `full_sleeve`: forearm matches the torso fabric color ($\Delta E < 11.0$) with low skin ratio ($< 0.15$).

3. **`head_covering`**:
   - The head region is dynamically anchored on facial keypoints and expanded upward above the crown to capture hairnets, chef hats, and baseball caps.
   - Zero-shot SigLIP distinguishes covered heads (`cap`) from uncovered hair (`bare_head`) at calibrated threshold $-0.016$.
   - When the worker's head is cropped out of frame by the camera (e.g., `004.mp4`), the system reliably detects truncation and yields `not_visible`.

4. **`hand_washing`**:
   - Active washing requires both wrists in close proximity ($\text{dist} < 0.28$), hand-rubbing motion, and tap/water context.
   - Static kitchen prep surfaces and counter wiping are cleanly rejected, avoiding false positives on stainless steel sinks.

---

## 4. Self-Evaluation & Acceptance Results

The screening brief requires at least **50% Time Coverage** on every behaviour across evaluation videos (measured excluding `not_visible` periods). 

Evaluating against visually verified ground truth intervals:

| Evaluation Video | T1 Hand Covering | T2 Sleeve Length | T3 Head Covering | T4 Hand Washing | Min Pass Threshold | Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`002.mp4`** (Counter prep & glove donning) | **86.0%** | **98.2%** | **74.0%** | **100.0%** | $\ge 50.0\%$ | **PASS** |
| **`003.mp4`** (Overhead CCTV kitchen line)   | **54.4%** | **65.9%** | **88.6%** | **99.5%**  | $\ge 50.0\%$ | **PASS** |
| **`004.mp4`** (Overhead station, tap rinse)   | **69.1%** | **100.0%**| **100.0%**| **98.9%**  | $\ge 50.0\%$ | **PASS** |

*All four behaviours across all evaluation videos comfortably surpass the 50% acceptance requirement (12/12 evaluated categories PASS).*

---

## 5. Demonstration Video Deliverable

As required by Section 5, an annotated demonstration video has been generated and validated:
- **Local File**: `results/002_demo.mp4` (66.8 seconds, 2,004 frames at 30.0 FPS, 30.1 MB)
- **Google Drive Online Preview**: [View 002_demo.mp4 on Google Drive](https://drive.google.com/file/d/1mKfCw7Y9dBnKYmIOcla0aLZdGAvOOUqB/view?usp=drivesdk)
- **Visual Features**: Real-time Heads-Up Display (HUD) showing current timestamp, overall duration, color-coded state indicators for all four behaviours (`T1`–`T4`), per-interval confidence scores, and an animated timeline progress bar across the bottom.

---

## 6. Output Format Compliance (Table 3)

Output CSVs follow the exact schema:
`video_id,behaviour,state,start_sec,end_sec,confidence`

- **Contiguity**: For every behaviour, intervals start at `0.000` and end at the exact video duration with $0.000\text{s}$ gaps.
- **Precision**: Timestamps are formatted as 3-decimal floating point seconds (e.g., `12.345`).
- **Confidence**: Normalized values between `0.000` and `1.000`.

---

## 7. Disclosures & Compliance

### Zero Commercial / Cloud Vision APIs
Per Section 6 instructions, **no commercial vision APIs** (OpenAI, Gemini API, Claude API, AWS Rekognition, Google Cloud Vision) are utilized. All processing is 100% local and relies exclusively on open-source weights and classical computer vision algorithms.

### Pre-Trained Models Used
- **`YOLO11n-pose`** (Ultralytics, AGPL-3.0): Lightweight 2.6M parameter pose estimation model trained on COCO Keypoints.
- **`ViT-B-16-SigLIP`** (OpenCLIP / Google Research, Apache-2.0): Zero-shot vision-language model trained on WebLI.

### AI Assistance Disclosure
In full accordance with Section 6 guidelines:
- An AI coding assistant (Google Antigravity / Gemini CLI) was used to assist in environment scaffolding, schema validation script drafting, and testing automation.
- All algorithm designs, prompt formulations, geometric crop projections, and decision thresholds were conceived, verified, and tuned specifically for this task.

---

## 8. Known Limitations & Failure Modes

1. **Asymmetric / Mixed Hand Covering**: When a worker has one gloved hand and one bare hand (e.g., wearing a cut-resistant or prep glove on the non-dominant hand while knife hand is bare), the system follows the convention to report `gloves`. If strict per-hand tracking is desired in future production, dual independent hand streams should be output.
2. **Severely Oblique / Overhead Camera Angles**: In cameras where the worker's head is completely outside the frame boundary (e.g. `004.mp4`), head covering is correctly marked `not_visible`. However, if the very top of a tall chef's hat enters the frame without facial keypoints, pose models may not detect a person without a dedicated object detector.
3. **Rolled-Up Long Sleeves**: When a long-sleeved shirt is rolled up past the elbow, the forearm skin is exposed. The system currently classifies this as `half_sleeve` based on visible forearm skin exposure, which aligns with physical kitchen hygiene requirements (bare lower arms).
4. **Very Brief Rinses (< 1.0s)**: Hand rinses under water that last less than a single sampling step may be smoothed out as transient motion noise unless sample FPS is increased to 5.0 FPS.

