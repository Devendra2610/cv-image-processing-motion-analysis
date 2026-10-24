"""
Task 1: Image Restoration and Segmentation
Units 3 Practical Assignment
Author: Devendra Agrawal
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25.0) -> np.ndarray:
    """Add additive zero-mean Gaussian noise to an image."""
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy


def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> np.ndarray:
    """Add salt-and-pepper (impulse) noise to an image."""
    noisy = image.copy()
    num_salt = np.ceil(salt_prob * image.shape[0] * image.shape[1])
    num_pepper = np.ceil(pepper_prob * image.shape[0] * image.shape[1])

    # Salt (white pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy[tuple(coords)] = 255

    # Pepper (black pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy[tuple(coords)] = 0
    return noisy


def compute_metrics(clean: np.ndarray, test_img: np.ndarray):
    """Compute PSNR and SSIM between clean ground truth and test image."""
    p_val = psnr(clean, test_img)
    s_val = ssim(clean, test_img, channel_axis=2)
    return round(float(p_val), 2), round(float(s_val), 4)


def run_image_restoration(image_path: str, out_dir: str):
    """Perform Gaussian & Salt-and-Pepper noise addition and filtering."""
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Add noise
    np.random.seed(42)
    noisy_gauss = add_gaussian_noise(img_rgb, mean=0, sigma=25.0)
    noisy_sp = add_salt_and_pepper_noise(img_rgb, salt_prob=0.03, pepper_prob=0.03)

    # 2. Filter images
    # Restore Gaussian noise
    gauss_median = cv2.medianBlur(noisy_gauss, 5)
    gauss_gaussian = cv2.GaussianBlur(noisy_gauss, (5, 5), sigmaX=1.5)

    # Restore Salt-and-Pepper noise
    sp_median = cv2.medianBlur(noisy_sp, 5)
    sp_gaussian = cv2.GaussianBlur(noisy_sp, (5, 5), sigmaX=1.5)

    # 3. Calculate PSNR and SSIM metrics
    metrics = {
        "Gaussian_Noisy": compute_metrics(img_rgb, noisy_gauss),
        "Gaussian_restored_GaussianFilter": compute_metrics(img_rgb, gauss_gaussian),
        "Gaussian_restored_MedianFilter": compute_metrics(img_rgb, gauss_median),
        "SaltPepper_Noisy": compute_metrics(img_rgb, noisy_sp),
        "SaltPepper_restored_GaussianFilter": compute_metrics(img_rgb, sp_gaussian),
        "SaltPepper_restored_MedianFilter": compute_metrics(img_rgb, sp_median),
    }

    # Save individual results
    cv2.imwrite(os.path.join(out_dir, "clean_original.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "noise_gaussian.png"), cv2.cvtColor(noisy_gauss, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "noise_salt_pepper.png"), cv2.cvtColor(noisy_sp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "gauss_restored_gaussian.png"), cv2.cvtColor(gauss_gaussian, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "gauss_restored_median.png"), cv2.cvtColor(gauss_median, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "sp_restored_gaussian.png"), cv2.cvtColor(sp_gaussian, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "sp_restored_median.png"), cv2.cvtColor(sp_median, cv2.COLOR_RGB2BGR))

    # Save summary JSON
    with open(os.path.join(out_dir, "restoration_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Create visualization comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Image Restoration Analysis: Gaussian Noise vs. Salt & Pepper Noise", fontsize=16, fontweight='bold')

    # Row 1: Gaussian noise & restorations
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Clean Image\n(Ground Truth)")
    axes[0, 0].axis("off")

    p, s = metrics["Gaussian_Noisy"]
    axes[0, 1].imshow(noisy_gauss)
    axes[0, 1].set_title(f"Gaussian Noise (σ=25)\nPSNR: {p} dB | SSIM: {s}")
    axes[0, 1].axis("off")

    p, s = metrics["Gaussian_restored_GaussianFilter"]
    axes[0, 2].imshow(gauss_gaussian)
    axes[0, 2].set_title(f"Gaussian Filter (5x5, σ=1.5)\nPSNR: {p} dB | SSIM: {s}")
    axes[0, 2].axis("off")

    p, s = metrics["Gaussian_restored_MedianFilter"]
    axes[0, 3].imshow(gauss_median)
    axes[0, 3].set_title(f"Median Filter (5x5)\nPSNR: {p} dB | SSIM: {s}")
    axes[0, 3].axis("off")

    # Row 2: Salt & Pepper noise & restorations
    axes[1, 0].imshow(img_rgb)
    axes[1, 0].set_title("Original Clean Image\n(Ground Truth)")
    axes[1, 0].axis("off")

    p, s = metrics["SaltPepper_Noisy"]
    axes[1, 1].imshow(noisy_sp)
    axes[1, 1].set_title(f"Salt & Pepper Noise (p=0.06)\nPSNR: {p} dB | SSIM: {s}")
    axes[1, 1].axis("off")

    p, s = metrics["SaltPepper_restored_GaussianFilter"]
    axes[1, 2].imshow(sp_gaussian)
    axes[1, 2].set_title(f"Gaussian Filter (5x5, σ=1.5)\nPSNR: {p} dB | SSIM: {s}")
    axes[1, 2].axis("off")

    p, s = metrics["SaltPepper_restored_MedianFilter"]
    axes[1, 3].imshow(sp_median)
    axes[1, 3].set_title(f"Median Filter (5x5) [BEST]\nPSNR: {p} dB | SSIM: {s}", color='darkgreen', fontweight='bold')
    axes[1, 3].axis("off")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "restoration_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("[Task 1A] Restoration complete. Results saved to:", out_dir)
    print("PSNR and SSIM Metrics:")
    for k, v in metrics.items():
        print(f"  {k:35s}: PSNR = {v[0]:5.2f} dB | SSIM = {v[1]:.4f}")

    return metrics


def run_morphological_segmentation(image_path: str, out_dir: str):
    """Perform thresholding, morphological operations, Canny edge detection, and Watershed segmentation."""
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Resize if large for consistent processing
    h, w = img_bgr.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Thresholding: Otsu vs Adaptive Gaussian
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_thresh = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    cv2.imwrite(os.path.join(out_dir, "otsu_threshold.png"), otsu_thresh)
    cv2.imwrite(os.path.join(out_dir, "adaptive_threshold.png"), adaptive_thresh)

    # 2. Morphological Operations: Erosion, Dilation, Opening, Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(otsu_thresh, kernel, iterations=2)
    dilated = cv2.dilate(otsu_thresh, kernel, iterations=2)
    opened = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite(os.path.join(out_dir, "morph_erosion.png"), eroded)
    cv2.imwrite(os.path.join(out_dir, "morph_dilation.png"), dilated)
    cv2.imwrite(os.path.join(out_dir, "morph_opening.png"), opened)
    cv2.imwrite(os.path.join(out_dir, "morph_closing.png"), closed)

    # Visual grid for morphology
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Thresholding and Morphological Operations", fontsize=16, fontweight='bold')
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title("Grayscale Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(otsu_thresh, cmap='gray')
    axes[0, 1].set_title("Otsu's Thresholding (Binary Mask)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(adaptive_thresh, cmap='gray')
    axes[0, 2].set_title("Adaptive Gaussian Thresholding")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(eroded, cmap='gray')
    axes[1, 0].set_title("Erosion (2 iters, 5x5 Rect)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(dilated, cmap='gray')
    axes[1, 1].set_title("Dilation (2 iters, 5x5 Rect)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(opened, cmap='gray')
    axes[1, 2].set_title("Opening (Noise Removal)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "morphological_operations_grid.png"), dpi=300)
    plt.close()

    # 3. Canny Edge Detection (Comparing Thresholds)
    canny_low = cv2.Canny(blurred_gray, 30, 100)
    canny_mid = cv2.Canny(blurred_gray, 50, 150)
    canny_high = cv2.Canny(blurred_gray, 100, 200)

    cv2.imwrite(os.path.join(out_dir, "canny_edges_50_150.png"), canny_mid)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Canny Edge Detection Multi-Threshold Sensitivity Analysis", fontsize=15, fontweight='bold')
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(canny_low, cmap='gray')
    axes[1].set_title("Canny (Low: 30, High: 100)\nHigh Sensitivity / More Edges")
    axes[1].axis("off")

    axes[2].imshow(canny_mid, cmap='gray')
    axes[2].set_title("Canny (Low: 50, High: 150) [Optimal]\nClean Structural Boundaries")
    axes[2].axis("off")

    axes[3].imshow(canny_high, cmap='gray')
    axes[3].set_title("Canny (Low: 100, High: 200)\nSelective Strong Edges Only")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "canny_edge_detection.png"), dpi=300)
    plt.close()

    # 4. Watershed Segmentation
    # Preprocessing: morphological opening on thresholded image to remove tiny noise
    opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Finding sure background area by dilating
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area using Euclidean Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Normalize distance transform for visualization
    dist_vis = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region = sure_bg - sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Mark unknown region as 0
    markers[unknown == 255] = 0

    # Apply Watershed
    watershed_img = img_bgr.copy()
    markers = cv2.watershed(watershed_img, markers)

    # Mark boundaries in red on RGB image
    watershed_vis = img_rgb.copy()
    watershed_vis[markers == -1] = [255, 0, 0]

    # Create distinct colored segmentation mask
    num_labels = np.max(markers)
    color_mask = np.zeros_like(img_rgb)
    np.random.seed(42)
    colors = np.random.randint(40, 255, size=(num_labels + 2, 3), dtype=np.uint8)
    for lbl in range(2, num_labels + 1):
        color_mask[markers == lbl] = colors[lbl]

    blended_seg = cv2.addWeighted(img_rgb, 0.65, color_mask, 0.35, 0)
    blended_seg[markers == -1] = [255, 0, 0]

    cv2.imwrite(os.path.join(out_dir, "watershed_boundaries.png"), cv2.cvtColor(watershed_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "watershed_color_segmented.png"), cv2.cvtColor(blended_seg, cv2.COLOR_RGB2BGR))

    # Watershed visualization figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle("Marker-Controlled Watershed Segmentation Pipeline", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(opening, cmap='gray')
    axes[0, 1].set_title("2. Morphological Opening (Clean Mask)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(dist_vis, cmap='jet')
    axes[0, 2].set_title("3. Euclidean Distance Transform")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(sure_fg, cmap='gray')
    axes[1, 0].set_title("4. Sure Foreground (Peaks)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(markers, cmap='tab20')
    axes[1, 1].set_title("5. Watershed Markers / Basins")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(blended_seg)
    axes[1, 2].set_title("6. Final Watershed Segmented Regions\n(Boundaries in Red)", fontweight='bold')
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "watershed_segmentation.png"), dpi=300)
    plt.close()

    print("[Task 1B] Morphological processing and watershed segmentation complete.")
    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 1: Image Restoration and Segmentation")
    parser.add_argument("--image", default="data/Image_org.jpeg", help="Path to input original photograph")
    parser.add_argument("--out_restoration", default="outputs/task1_restoration", help="Output directory for restoration")
    parser.add_argument("--out_segmentation", default="outputs/task1_segmentation", help="Output directory for segmentation")
    args = parser.parse_args()

    print("==================================================")
    print("RUNNING TASK 1: IMAGE RESTORATION AND SEGMENTATION")
    print("==================================================")
    run_image_restoration(args.image, args.out_restoration)
    run_morphological_segmentation(args.image, args.out_segmentation)
    print("Task 1 finished successfully!\n")
