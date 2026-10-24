"""
Task 2: Feature Extraction and Image Matching
Unit 4 Practical Assignment
Author: Devendra Agrawal
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


def extract_and_visualize_hog(img_bgr: np.ndarray, out_dir: str):
    """Extract and visualize Histogram of Oriented Gradients (HOG)."""
    # Resize for standard HOG visualization
    h, w = img_bgr.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Compute HOG features and visualization image
    fd, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Histogram of Oriented Gradients (HOG) Feature Representation", fontsize=15, fontweight='bold')

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Input Image ({img_bgr.shape[1]}x{img_bgr.shape[0]})")
    axes[0].axis("off")

    axes[1].imshow(hog_image_rescaled, cmap='magma')
    axes[1].set_title(f"HOG Gradient Orientation Field\n({len(fd):,} feature dimensions)")
    axes[1].axis("off")

    # Plot sample feature descriptor distribution
    axes[2].hist(fd[:1000], bins=30, color='royalblue', edgecolor='black', alpha=0.7)
    axes[2].set_title("HOG Normalized Descriptor Distribution\n(First 1,000 bins)")
    axes[2].set_xlabel("Gradient Energy Value")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hog_visualization.png"), dpi=300)
    plt.close()

    print(f"[Task 2] HOG extracted. Total descriptor length: {len(fd):,}")
    return len(fd)


def match_sift_features(img1_rgb: np.ndarray, img2_rgb: np.ndarray, ratio_thresh: float = 0.75):
    """Detect SIFT keypoints and match using BFMatcher with Lowe's ratio test."""
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return kp1, kp2, [], []

    # Brute-Force Matcher with L2 norm for SIFT descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Apply David Lowe's ratio test
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return kp1, kp2, raw_matches, good_matches


def run_feature_matching_pipeline(img1_path: str, img2_path: str, out_dir: str):
    """Full Task 2 pipeline: HOG, SIFT, Matching, Rotation, and Resizing experiments."""
    os.makedirs(out_dir, exist_ok=True)

    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)
    if img1_bgr is None or img2_bgr is None:
        raise FileNotFoundError(f"Images not found: {img1_path} or {img2_path}")

    # Standardize scale for efficient processing
    h1, w1 = img1_bgr.shape[:2]
    max_dim = 800
    if max(h1, w1) > max_dim:
        scale = max_dim / max(h1, w1)
        img1_bgr = cv2.resize(img1_bgr, (int(w1 * scale), int(h1 * scale)))
        img2_bgr = cv2.resize(img2_bgr, (int(img2_bgr.shape[1] * scale), int(img2_bgr.shape[0] * scale)))

    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    # 1. HOG Feature Extraction
    extract_and_visualize_hog(img1_bgr, out_dir)

    # 2. Baseline SIFT Matching between Image 1 and Image 2
    kp1, kp2, raw_baseline, good_baseline = match_sift_features(img1_rgb, img2_rgb)

    # Draw rich keypoints
    kp_img1 = cv2.drawKeypoints(img1_rgb, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_img2 = cv2.drawKeypoints(img2_rgb, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(out_dir, "sift_keypoints_img1.png"), cv2.cvtColor(kp_img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "sift_keypoints_img2.png"), cv2.cvtColor(kp_img2, cv2.COLOR_RGB2BGR))

    # Draw matches
    match_img_baseline = cv2.drawMatches(
        img1_rgb, kp1, img2_rgb, kp2, good_baseline, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(os.path.join(out_dir, "sift_matches_baseline.png"), cv2.cvtColor(match_img_baseline, cv2.COLOR_RGB2BGR))

    # 3. Rotation Invariance Experiment: Rotate Image 2 by 45 degrees
    h2, w2 = img2_rgb.shape[:2]
    center = (w2 // 2, h2 // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h2 * sin) + (w2 * cos))
    new_h = int((h2 * cos) + (w2 * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    img2_rot_rgb = cv2.warpAffine(img2_rgb, rot_mat, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)

    kp1_rot, kp2_rot, raw_rot, good_rot = match_sift_features(img1_rgb, img2_rot_rgb)
    match_img_rot = cv2.drawMatches(
        img1_rgb, kp1_rot, img2_rot_rgb, kp2_rot, good_rot, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(os.path.join(out_dir, "sift_matches_rotated_45.png"), cv2.cvtColor(match_img_rot, cv2.COLOR_RGB2BGR))

    # 4. Scale Invariance Experiment: Resize Image 2 by 0.6x
    img2_scaled_rgb = cv2.resize(img2_rgb, (0, 0), fx=0.6, fy=0.6)
    kp1_scl, kp2_scl, raw_scl, good_scl = match_sift_features(img1_rgb, img2_scaled_rgb)
    match_img_scl = cv2.drawMatches(
        img1_rgb, kp1_scl, img2_scaled_rgb, kp2_scl, good_scl, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(os.path.join(out_dir, "sift_matches_resized_06x.png"), cv2.cvtColor(match_img_scl, cv2.COLOR_RGB2BGR))

    # 5. Compile quantitative matching statistics
    stats = {
        "Baseline": {
            "condition": "Original Pair (Image 1 vs Image 2)",
            "keypoints_img1": len(kp1),
            "keypoints_img2": len(kp2),
            "raw_matches": len(raw_baseline),
            "good_matches": len(good_baseline),
            "match_ratio_pct": round(len(good_baseline) / max(1, len(raw_baseline)) * 100, 2)
        },
        "Rotated_45deg": {
            "condition": "Rotated Image 2 (45 degrees)",
            "keypoints_img1": len(kp1_rot),
            "keypoints_img2": len(kp2_rot),
            "raw_matches": len(raw_rot),
            "good_matches": len(good_rot),
            "match_ratio_pct": round(len(good_rot) / max(1, len(raw_rot)) * 100, 2)
        },
        "Resized_0.6x": {
            "condition": "Rescaled Image 2 (0.6x downscaled)",
            "keypoints_img1": len(kp1_scl),
            "keypoints_img2": len(kp2_scl),
            "raw_matches": len(raw_scl),
            "good_matches": len(good_scl),
            "match_ratio_pct": round(len(good_scl) / max(1, len(raw_scl)) * 100, 2)
        }
    }

    with open(os.path.join(out_dir, "feature_matching_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # 6. Combined Multi-Panel Visualization Figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(kp_img1)
    ax0.set_title(f"Image 1 SIFT Keypoints ({len(kp1)} detected)", fontsize=13, fontweight='bold')
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(kp_img2)
    ax1.set_title(f"Image 2 SIFT Keypoints ({len(kp2)} detected)", fontsize=13, fontweight='bold')
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.imshow(match_img_baseline)
    ax2.set_title(
        f"Baseline Pair Matching (Lowe's Ratio < 0.75): {len(good_baseline)} Inlier Matches ({stats['Baseline']['match_ratio_pct']}%)",
        fontsize=13, fontweight='bold', color='darkgreen'
    )
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.imshow(match_img_rot)
    ax3.set_title(
        f"Rotation Invariance (45° Rotated): {len(good_rot)} Matches ({stats['Rotated_45deg']['match_ratio_pct']}%)",
        fontsize=12, fontweight='bold'
    )
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.imshow(match_img_scl)
    ax4.set_title(
        f"Scale Invariance (0.6x Scaled): {len(good_scl)} Matches ({stats['Resized_0.6x']['match_ratio_pct']}%)",
        fontsize=12, fontweight='bold'
    )
    ax4.axis("off")

    plt.suptitle("Task 2: SIFT Feature Extraction, Lowe's Ratio Matching & Invariance Evaluation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_matching_comparison.png"), dpi=300)
    plt.close()

    print("[Task 2] SIFT Feature matching complete.")
    print("Matching Statistics Summary:")
    for k, v in stats.items():
        print(f"  {k:15s}: {v['good_matches']} good matches / {v['raw_matches']} raw ({v['match_ratio_pct']}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 2: Feature Extraction and Image Matching")
    parser.add_argument("--image1", default="data/Image_org.jpeg", help="Path to primary photograph")
    parser.add_argument("--image2", default="data/Image_2.jpeg", help="Path to second photograph")
    parser.add_argument("--out_dir", default="outputs/task2_features", help="Output directory")
    args = parser.parse_args()

    print("==================================================")
    print("RUNNING TASK 2: FEATURE EXTRACTION & IMAGE MATCHING")
    print("==================================================")
    run_feature_matching_pipeline(args.image1, args.image2, args.out_dir)
    print("Task 2 finished successfully!\n")
