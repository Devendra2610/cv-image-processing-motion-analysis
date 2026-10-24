"""
Task 4: Video Motion Estimation
Unit 5 Practical Assignment
Author: Devendra Agrawal
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def run_motion_estimation(video_path: str, out_dir: str, num_frames_to_process: int = 150):
    """
    Process video using:
    1. Consecutive frame extraction
    2. Frame Differencing
    3. GMM-based Background Subtraction (MOG2)
    4. Lucas-Kanade Optical Flow
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Task 4] Video Info: {total_frames} frames, {fps:.1f} FPS, {w}x{h}")

    # GMM Background Subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)

    # LK optical flow parameters
    feature_params = dict(maxCorners=150, qualityLevel=0.03, minDistance=12, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frames from video.")

    # Downscale for consistent fast processing if needed
    if max(h, w) > 960:
        scale = 960 / max(h, w)
        prev_frame = cv2.resize(prev_frame, (int(w * scale), int(h * scale)))
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Storage for selected comparison frame
    comparison_snapshot = None
    target_sample_frame = min(75, total_frames // 2)

    frame_idx = 0
    flow_mask_track = np.zeros_like(prev_frame)

    while cap.isOpened() and frame_idx < min(total_frames - 1, num_frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        if max(h, w) > 960:
            scale = 960 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Frame Differencing
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        diff_clean = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
        diff_clean = cv2.morphologyEx(diff_clean, cv2.MORPH_DILATE, kernel)

        # 2. GMM Background Subtraction
        fg_mask_raw = bg_subtractor.apply(frame)
        # Remove shadows (value 127 in MOG2) and keep only foreground (value 255)
        _, gmm_fg = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)
        gmm_clean = cv2.morphologyEx(gmm_fg, cv2.MORPH_OPEN, kernel)
        gmm_clean = cv2.morphologyEx(gmm_clean, cv2.MORPH_DILATE, kernel)

        # 3. Lucas-Kanade Optical Flow
        lk_vis = frame.copy()
        if p0 is None or len(p0) < 15:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        motion_vectors = []
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mag = np.hypot(a - c, b - d)
                    if mag > 1.0: # filter stationary points
                        motion_vectors.append(((c, d), (a, b), mag))
                        # Draw vector arrow
                        cv2.arrowedLine(lk_vis, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                        cv2.circle(lk_vis, (int(a), int(b)), 4, (0, 255, 0), -1)

                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = None
        else:
            p0 = None

        # Capture comparison snapshot at representative motion instant
        if frame_idx == target_sample_frame:
            # Draw bounding boxes from GMM foreground
            contours, _ = cv2.findContours(gmm_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox_vis = frame.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 400:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    cv2.rectangle(bbox_vis, (x, y), (x + bw, y + bh), (0, 255, 255), 3)
                    cv2.putText(bbox_vis, "Moving Object", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            comparison_snapshot = {
                "frame_prev": prev_frame.copy(),
                "frame_curr": frame.copy(),
                "diff_mask": diff_clean.copy(),
                "gmm_mask": gmm_clean.copy(),
                "optical_flow": lk_vis.copy(),
                "tracked_bbox": bbox_vis.copy(),
                "frame_idx": frame_idx,
                "vector_count": len(motion_vectors)
            }

        prev_gray = curr_gray.copy()
        prev_frame = frame.copy()

    cap.release()

    if comparison_snapshot is None:
        raise RuntimeError("No comparison snapshot could be generated from video.")

    # Save individual snapshot files
    cv2.imwrite(os.path.join(out_dir, "frame_previous.png"), comparison_snapshot["frame_prev"])
    cv2.imwrite(os.path.join(out_dir, "frame_current.png"), comparison_snapshot["frame_curr"])
    cv2.imwrite(os.path.join(out_dir, "diff_foreground_mask.png"), comparison_snapshot["diff_mask"])
    cv2.imwrite(os.path.join(out_dir, "gmm_foreground_mask.png"), comparison_snapshot["gmm_mask"])
    cv2.imwrite(os.path.join(out_dir, "optical_flow_vectors.png"), comparison_snapshot["optical_flow"])
    cv2.imwrite(os.path.join(out_dir, "motion_bounding_boxes.png"), comparison_snapshot["tracked_bbox"])

    # Multi-panel visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Task 4: Video Motion Estimation & Tracking Analysis (Frame #{comparison_snapshot['frame_idx']})",
                 fontsize=16, fontweight='bold')

    axes[0, 0].imshow(cv2.cvtColor(comparison_snapshot["frame_prev"], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Previous Frame $I_{t-1}$")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(comparison_snapshot["frame_curr"], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. Current Frame $I_t$")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(comparison_snapshot["diff_mask"], cmap='gray')
    axes[0, 2].set_title("3. Frame Differencing Mask\n($|I_t - I_{t-1}| > T$)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(comparison_snapshot["gmm_mask"], cmap='gray')
    axes[1, 0].set_title("4. GMM Background Subtraction (MOG2)\nAdaptive Mixture Foreground")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(comparison_snapshot["optical_flow"], cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"5. Lucas-Kanade Optical Flow\n({comparison_snapshot['vector_count']} Motion Vectors)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cv2.cvtColor(comparison_snapshot["tracked_bbox"], cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Motion Tracking & Bounding Box\n(Spatial Object Localization)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "motion_estimation_comparison.png"), dpi=300)
    plt.close()

    # Detailed technical comparison table data
    comparison_table = [
        {
            "Technique": "Frame Differencing (|I_t - I_{t-1}|)",
            "Underlying Principle": "Pixel-by-pixel intensity subtraction between consecutive frames with thresholding",
            "Computational Complexity": "Very Low (O(N) single subtraction per pixel, >100 FPS)",
            "Camera Shake Sensitivity": "Extremely High (Any camera tremor lights up entire mask as motion)",
            "Stationary Object Behavior": "Fails (Moving object becomes invisible the moment it stops moving)",
            "Ghosting Artifacts": "Double-ghost effect: marks both where object left and where it entered",
            "Output Representation": "Binary spatial mask of pixel changes"
        },
        {
            "Technique": "GMM Background Subtraction (MOG2)",
            "Underlying Principle": "Models pixel color distributions over time as a mixture of Gaussians",
            "Computational Complexity": "Moderate (~40-60 FPS, updates adaptive Gaussian parameters)",
            "Camera Shake Sensitivity": "Moderate (Handles subtle background motion like leaves, tremors)",
            "Stationary Object Behavior": "Good (Preserves stopped foreground for several frames before slow assimilation)",
            "Ghosting Artifacts": "Minimal (Initial ghost clears quickly as background model converges)",
            "Output Representation": "Clean binary foreground segmentation mask with shadow suppression"
        },
        {
            "Technique": "Lucas-Kanade Optical Flow",
            "Underlying Principle": "Solves optical flow equation under brightness constancy & spatial coherence",
            "Computational Complexity": "Moderate-High (Calculates spatial image gradients & 2x2 matrix inversions)",
            "Camera Shake Sensitivity": "Shows global vector field (Global shift vs local object vectors)",
            "Stationary Object Behavior": "Vectors drop to zero magnitude (Preserves tracked feature locations)",
            "Ghosting Artifacts": "None (No background model or ghost silhouettes generated)",
            "Output Representation": "Continuous 2D velocity vectors (dx, dy) with direction and magnitude"
        }
    ]

    with open(os.path.join(out_dir, "motion_estimation_comparison.json"), "w") as f:
        json.dump(comparison_table, f, indent=4)

    print("[Task 4] Video motion estimation complete. Saved outputs to:", out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 4: Video Motion Estimation")
    parser.add_argument("--video", default="data/sample_video.mp4", help="Input video path")
    parser.add_argument("--out_dir", default="outputs/task4_motion", help="Output directory")
    args = parser.parse_args()

    print("==================================================")
    print("RUNNING TASK 4: VIDEO MOTION ESTIMATION")
    print("==================================================")
    run_motion_estimation(args.video, args.out_dir)
    print("Task 4 finished successfully!\n")
