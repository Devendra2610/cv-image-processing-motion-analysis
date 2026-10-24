# 5–8 Minute Project Explanation Video Presentation Script
**Course:** Computer Vision (CCA 3) — Practical Coding Assignment (Units 3, 4 & 5)  
**Presenter / Student Name:** Devendra Agrawal  
**GitHub Username:** Devendra2610  
**Submission Evaluator Access:** rahul.inchal.inchal24@gmail.com, rahul.inchal.vf@mitwpu.edu.in  

---

## Video Recording Checklist & Guidelines
- **Target Duration:** 5 to 8 minutes.
- **Recording Mode:** Screen recording with web-camera feed (picture-in-picture) showing you speaking.
- **Tools You Can Use:** OBS Studio, Windows Game Bar (`Win + Alt + R`), or Google Meet recording.
- **Structure:**
  1. Introduction & Objectives (~1 min)
  2. Dataset Description (~45 sec)
  3. Task 1: Image Restoration & Segmentation (~1.5 min)
  4. Task 2: HOG & SIFT Feature Matching (~1.5 min)
  5. Task 3: Object Recognition (Viola-Jones vs. YOLOv8 & Failure Case) (~1.5 min)
  6. Task 4: Video Motion Estimation & Tracking on `video_1.mp4` (~1.5 min)
  7. Conclusion & Wrap-up (~30 sec)

---

## Detailed Spoken Script (Timeline Walkthrough)

### [0:00 - 0:50] 1. Introduction and Objectives
**What to show on screen:** Open the title slide or show the project directory in VS Code / IDE with `README.md`.

> *"Hello everyone and respected Professor Rahul. My name is Devendra Agrawal, and today I will be presenting my practical coding assignment for Computer Vision, covering Units 3, 4, and 5.*
> 
> *The objective of this assignment is to implement, evaluate, and critically compare fundamental techniques in computer vision. Specifically, we explore:*
> 1. *Image restoration using linear Gaussian filtering versus non-linear median filtering under Gaussian and Salt-and-Pepper noise, followed by morphological operations, Canny edge detection, and marker-controlled watershed segmentation.*
> 2. *Feature extraction using HOG descriptors and SIFT keypoint detection to evaluate scale and rotation invariance.*
> 3. *Object recognition comparing the classical Viola-Jones Haar Cascade detector with modern deep-learning YOLOv8, analyzing their speed and failure modes.*
> 4. *Video motion estimation comparing Frame Differencing, GMM background subtraction, and Lucas-Kanade optical flow on an original video.*
> 
> *All implementations are written in Python using OpenCV, NumPy, Matplotlib, scikit-image, and Ultralytics YOLO.*

---

### [0:50 - 1:30] 2. Dataset Description
**What to show on screen:** Open `data/Image_org.jpeg`, `data/Image_2.jpeg`, and play `data/video_1.mp4`.

> *"For the dataset, I used my own photograph, `Image_org.jpeg`, captured indoors at 899 by 1599 resolution. It contains clear structural details, facial geometry, textured clothing, and background architectural lines. I also generated a paired second image, `Image_2.jpeg`, with perspective and lighting adjustments to benchmark matching invariance.*
> 
> *For the video tasks, I used my self-recorded video, `video_1.mp4`, recorded in Full HD (1080 by 1920) at 30 FPS. It captures a Golden Retriever dog in an indoor living space turning its head, opening its mouth, and shifting its gaze against a stable room background. This provides rich natural motion dynamics and fine textured features for motion tracking."*

---

### [1:30 - 2:50] 3. Unit 3: Image Restoration & Segmentation (Task 1)
**What to show on screen:** Open `outputs/task1_restoration/restoration_comparison.png` and `outputs/task1_segmentation/watershed_segmentation.png`. Briefly show `task1_restoration_segmentation.py`.

> *"Moving to Task 1: Image Restoration. I added zero-mean Gaussian noise with sigma 25 and Salt-and-Pepper noise with 6% corrupted pixels to my photograph. I then applied both Gaussian filtering and Median filtering to both noisy images.*
> 
> *Looking at our quantitative results in the comparison table:*
> - *For Gaussian noise, the Gaussian filter gave the best reconstruction, achieving a PSNR of 31.65 dB and SSIM of 0.75, compared to 30.56 dB for the Median filter.*
> - *However, for Salt-and-Pepper noise, the Median filter performed dramatically better, reaching an outstanding PSNR of 36.85 dB and SSIM of 0.935, compared to only 29.45 dB for Gaussian filtering.*
> 
> *The theoretical reason is clear: Gaussian noise is additive and normally distributed, so linear spatial averaging effectively cancels the noise variance. In contrast, Salt-and-Pepper noise consists of extreme outlier impulse spikes of 0 and 255. A Gaussian filter averages these spikes with neighboring pixels, creating blurred smudges. The median filter, being a non-linear rank-order filter, replaces each pixel with the statistical median, cleanly removing the impulses while preserving sharp edges.*
> 
> *In Part B, I implemented Otsu thresholding and morphological operations: erosion, dilation, opening for noise removal, and closing for gap filling. I performed Canny edge detection across multiple thresholds, showing that low thresholds (30, 100) capture fine fabric textures, while higher thresholds (50, 150) isolate clear structural boundaries.*
> 
> *Finally, I implemented marker-controlled watershed segmentation using the Euclidean Distance Transform. By identifying sure foreground seeds and sure background, we completely eliminated the classical oversegmentation problem and achieved clean regional boundaries."*

---

### [2:50 - 4:10] 4. Unit 4: Feature Extraction and Matching (Task 2)
**What to show on screen:** Open `outputs/task2_features/hog_visualization.png` and `outputs/task2_features/feature_matching_comparison.png`.

> *"In Task 2, we focus on feature extraction and matching.*
> 
> *First, I extracted Histogram of Oriented Gradients (HOG) features using 9 orientation bins and 16x16 pixel cells. As seen in the visualization, HOG effectively captures the structural outline of the shoulders, face, and background door frame, yielding a compact 29,484-dimensional descriptor vector.*
> 
> *Next, I used SIFT—Scale Invariant Feature Transform—to detect keypoints and compute 128-dimensional descriptors. I matched keypoints between Image 1 and Image 2 using a Brute-Force Matcher combined with David Lowe's ratio test at 0.75.*
> 
> *To rigorously test SIFT's invariance properties, I conducted two experiments:*
> 1. *First, I rotated Image 2 by 45 degrees. SIFT retained 441 good inlier matches—a match retention ratio of 65.4%. This empirically demonstrates SIFT's rotational invariance, achieved because SIFT assigns a canonical orientation based on local gradient peaks to each keypoint before computing descriptors.*
> 2. *Second, I downscaled Image 2 by 0.6x. Here, SIFT maintained 245 good matches (36.4% retention). As expected, reducing resolution discards high-frequency details, yet SIFT's Gaussian scale-space pyramid successfully matches the remaining scale-normalized octave features."*

---

### [4:10 - 5:40] 5. Unit 5: Object Recognition (Task 3)
**What to show on screen:** Open `outputs/task3_recognition/detector_comparison_normal.png` and `outputs/task3_recognition/detector_failure_case_analysis.png`.

> *"Now in Task 3, we compare the classical Viola-Jones Haar Cascade classifier with the modern deep learning detector, YOLOv8.*
> 
> *On the clean photograph:*
> - *Viola-Jones executes very quickly on CPU (around 321 ms) and successfully detects faces using rectangular Haar features and an AdaBoost reject cascade.*
> - *YOLOv8 runs a complete CNN forward pass and detects the person with high confidence, providing multi-class detection out of the box.*
> 
> *Next, we evaluated an important edge case and detector failure mode:*
> *When we test the detectors on a 45-degree rotated photograph:*
> - *Viola-Jones completely fails, producing 0 detections. Why? Because Haar rectangular features are strictly axis-aligned. The classifier assumes horizontal eye alignment and a vertical nose ridge. When the face is rotated, the integral image calculations fail to trigger the cascade thresholds.*
> - *In contrast, YOLOv8 still successfully recognizes and localizes the person! Deep convolutional neural networks learn hierarchical, rotation-tolerant spatial representations across multiple receptive fields, making them vastly more robust to real-world orientation and pose changes."*

---

### [5:40 - 6:50] 6. Unit 5: Video Motion Estimation on `video_1.mp4` (Task 4)
**What to show on screen:** Open `outputs/task4_motion/motion_estimation_comparison.png`.

> *"Finally, in Task 4, we examine video motion estimation applied to our self-recorded video `video_1.mp4` across three distinct techniques:*
> 1. *Frame Differencing: By subtracting consecutive frames ($|I_t - I_{t-1}| > T$), it immediately detects moving regions such as the dog's head turning, mouth panting, and ear shifts with virtually zero computational overhead (>100 FPS). However, it creates double ghosts (where the head moved from and where it moved to), and when motion pauses, the region vanishes.*
> 2. *GMM Background Subtraction (MOG2): By modeling each pixel as an adaptive mixture of Gaussians over time, it establishes a continuous background model. It generates clean foreground silhouettes of the dog, suppresses subtle lighting shifts and shadows, and preserves stopped objects.*
> 3. *Lucas-Kanade Optical Flow: Unlike binary spatial masks, Lucas-Kanade solves the optical flow constraint equation under brightness constancy to compute continuous 2D motion velocity vectors ($dx, dy$). As seen on screen, 155 motion vectors track key feature points around the dog's snout, nose, eyes, and ears, displaying both the direction and magnitude of motion, making it ideal for kinematic trajectory tracking."*

---

### [6:50 - 7:30] 7. Conclusion & Git Repository
**What to show on screen:** Show the Git repository page on GitHub or VS Code with `git status` showing clean commit history, and display the technical report `REPORT.html`.

> *"In conclusion, this assignment provided hands-on experience bridging classical computer vision techniques with modern deep learning paradigms. We proved why non-linear filtering is essential for impulse noise, verified SIFT's mathematical scale and rotation invariance, demonstrated the rotational failure point of Viola-Jones compared to YOLO, and contrasted spatial foreground masks with kinematic optical flow vectors on real self-recorded video.*
> 
> *The entire executable source code, requirements, dataset instructions, and documentation are committed and pushed to my Git repository at github.com/Devendra2610/cv-image-processing-motion-analysis.*
> *Access has been granted to rahul.inchal.inchal24@gmail.com and rahul.inchal.vf@mitwpu.edu.in.*
> 
> *Thank you very much for your time and guidance!"*

---

## Tips for Recording
1. **Pacing:** Speak at a calm, confident pace.
2. **Visual Transitions:** Switch between the code files and the generated figures in `outputs/` smoothly as you speak.
3. **Upload:** After recording, upload your `.mp4` video to Google Drive, set sharing permission to *"Anyone with the link can view"* (or add `rahul.inchal.inchal24@gmail.com`), and paste the link into `REPORT.md` and `README.md`.
