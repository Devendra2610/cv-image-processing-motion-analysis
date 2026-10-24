# Computer Vision Practical Coding Assignment (CCA 3)
### Practical Implementation of Units 3, 4, and 5
**Student Name:** Devendra Agrawal  
**GitHub Profile:** [Devendra2610](https://github.com/Devendra2610)  
**Repository:** [cv-image-processing-motion-analysis](https://github.com/Devendra2610/cv-image-processing-motion-analysis)  
**Evaluator Access Granted:** `rahul.inchal.inchal24@gmail.com`, `rahul.inchal.vf@mitwpu.edu.in`  

---

## 📌 Project Overview
This repository contains the complete, reproducible implementation of the Practical Coding Assignment for Computer Vision (Units 3, 4, and 5). The project implements, analyzes, and compares core computer vision algorithms across four primary tasks:
1. **Unit 3 (Task 1): Image Restoration and Segmentation**
   - Gaussian & Salt-and-Pepper noise addition
   - Gaussian vs. Median spatial filtering with quantitative **PSNR and SSIM** benchmarks
   - Otsu & Adaptive Gaussian thresholding
   - Morphological operations (Erosion, Dilation, Opening, Closing)
   - Multi-threshold Canny edge detection sensitivity analysis
   - Marker-controlled Watershed segmentation with Euclidean Distance Transform
2. **Unit 4 (Task 2): Feature Extraction and Image Matching**
   - Histogram of Oriented Gradients (HOG) visualization and descriptor distribution
   - Scale-Invariant Feature Transform (SIFT) keypoint detection
   - Feature matching with David Lowe's Ratio Test ($d_1 / d_2 < 0.75$)
   - Invariance evaluation under $45^\circ$ image rotation and $0.6\times$ spatial downscaling
3. **Unit 5 (Task 3): Object Recognition Using Viola–Jones and YOLO**
   - Classical Viola-Jones Haar Cascade face detection vs. Ultralytics YOLOv8 deep learning
   - Inference latency, bounding box precision, and multi-class comparison
   - Edge/Failure case analysis: detector breakdown on rotated viewpoint ($45^\circ$)
4. **Unit 5 (Task 4): Video Motion Estimation (`video_1.mp4`)**
   - Sequential video frame extraction from original mobile recording
   - Consecutive frame differencing ($|I_t - I_{t-1}| > T$)
   - Gaussian Mixture Model (GMM-MOG2) background subtraction with shadow suppression
   - Sparse Lucas–Kanade optical flow kinematic velocity vector tracking
   - Comprehensive comparative evaluation table

---

## 📂 Repository Directory Structure
```text
CV/
├── data/                                # Input datasets
│   ├── Image_org.jpeg                   # Primary original photograph (899x1599)
│   ├── Image_2.jpeg                     # Paired photograph with perspective shift
│   └── video_1.mp4                      # Original self-recorded video (1080x1920, 284 frames)
├── outputs/                             # Generated figures and quantitative metrics
│   ├── task1_restoration/               # PSNR/SSIM charts, noisy & restored images
│   ├── task1_segmentation/              # Thresholding, morphology, Canny, Watershed
│   ├── task2_features/                  # HOG vector field, SIFT keypoints, matching
│   ├── task3_recognition/               # Viola-Jones & YOLOv8 bounding boxes, failure case
│   └── task4_motion/                    # Differencing, GMM mask, Optical Flow vectors on video_1.mp4
├── task1_restoration_segmentation.py    # Unit 3 implementation script
├── task2_feature_matching.py            # Unit 4 implementation script
├── task3_object_recognition.py          # Unit 5 object recognition script
├── task4_motion_estimation.py           # Unit 5 video motion estimation script
├── run_all.py                           # Master pipeline runner (runs all tasks end-to-end)
├── generate_report.py                   # Compiles REPORT.md and REPORT.html
├── REPORT.md                            # Comprehensive technical report (Markdown)
├── REPORT.html                          # High-fidelity printable technical report
├── presentation_script.md               # 5-8 minute project video presentation script
├── requirements.txt                     # Python library dependencies
└── README.md                            # Project documentation & execution guide
```

---

## 🛠️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.8 to 3.13 installed.
- Git installed.

### 2. Clone the Repository
```bash
git clone https://github.com/Devendra2610/cv-image-processing-motion-analysis.git
cd cv-image-processing-motion-analysis
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

*Required packages include:*
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `scikit-image>=0.21.0`
- `ultralytics>=8.0.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`

---

## 🚀 Execution Guide

### Option A: Run the Entire Pipeline with a Single Command
To execute all four tasks sequentially, compute all metrics, and save all visualization figures:
```bash
python run_all.py
```

### Option B: Execute Individual Tasks

#### Task 1: Image Restoration and Segmentation
```bash
python task1_restoration_segmentation.py --image data/Image_org.jpeg
```
*Outputs saved to `outputs/task1_restoration/` and `outputs/task1_segmentation/`.*

#### Task 2: Feature Extraction and Image Matching
```bash
python task2_feature_matching.py --image1 data/Image_org.jpeg --image2 data/Image_2.jpeg
```
*Outputs saved to `outputs/task2_features/`.*

#### Task 3: Object Recognition (Viola-Jones vs YOLOv8)
```bash
python task3_object_recognition.py --image data/Image_org.jpeg
```
*Outputs saved to `outputs/task3_recognition/`.*

#### Task 4: Video Motion Estimation
```bash
python task4_motion_estimation.py --video data/video_1.mp4
```
*Outputs saved to `outputs/task4_motion/`.*

#### Generate / Update Technical Documentation
```bash
python generate_report.py
```
*Builds both `REPORT.md` and styled `REPORT.html`.*

---

## 📊 Summary of Experimental Results

### 1. Image Restoration (PSNR / SSIM Benchmark)
| Filter / Degradation | Gaussian Noise ($\sigma=25$) | Salt & Pepper Noise ($p=0.06$) |
| :--- | :---: | :---: |
| **Noisy Baseline** | 20.23 dB / 0.1670 | 17.89 dB / 0.1926 |
| **Gaussian Filter (5x5)** | **31.65 dB / 0.7493 [BEST]** | 29.45 dB / 0.6689 |
| **Median Filter (5x5)** | 30.56 dB / 0.6922 | **36.85 dB / 0.9350 [BEST]** |

*Takeaway:* Non-linear median filtering is vastly superior for impulse noise (+7.4 dB gain over Gaussian filter), whereas linear Gaussian filtering excels for additive normally distributed noise.

### 2. SIFT Invariance Evaluation
- **Baseline Matches:** 470 inliers out of 674 raw matches (**69.73% retention**).
- **Rotation ($45^\circ$):** 441 inliers out of 674 raw matches (**65.43% retention**), proving mathematical rotation invariance via canonical gradient assignment.
- **Scaling ($0.6\times$):** 245 inliers out of 674 raw matches (**36.35% retention**), demonstrating scale-space pyramid matching under resolution downsampling.

### 3. Object Recognition: Viola-Jones vs. YOLOv8
- **Speed:** Viola-Jones operates at ~321 ms on CPU; YOLOv8 performs deep CNN inference in ~3.0 s on CPU.
- **Failure Mode:** When tested on a $45^\circ$ rotated face, **Viola-Jones produced 0 detections (complete failure)** due to rigid axis-aligned Haar features. In contrast, **YOLOv8 successfully detected the person and face** due to deep translation/rotation-tolerant feature hierarchies.

### 4. Video Motion Estimation Comparison (`video_1.mp4`)
- **Frame Differencing:** Real-time (>100 FPS), detects head/jaw movements immediately, but creates double-ghost artifacts and fails completely when motion stops.
- **GMM Background Subtraction:** Produces robust silhouette masks, filters shadows, and preserves stopped objects.
- **Lucas-Kanade Optical Flow:** Computes continuous kinematic 2D velocity vectors with explicit direction and magnitude (155 motion vectors tracking facial features and head turn).

---

## 🎥 Project Explanation Video
A detailed spoken video presentation script is available in [`presentation_script.md`](presentation_script.md).  
- **Video Link:** `[Google Drive Video Link - Enter URL Here]`  
*(Sharing set to: Anyone with link can view / access given to rahul.inchal.inchal24@gmail.com and rahul.inchal.vf@mitwpu.edu.in)*

---

## 👤 Author & Academic Integrity
- **Student:** Devendra Agrawal
- **GitHub:** [Devendra2610](https://github.com/Devendra2610)
- **Assignment:** Computer Vision CCA 3 (Units 3, 4, and 5)
