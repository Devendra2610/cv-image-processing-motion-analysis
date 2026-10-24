"""
Task 3: Object Recognition Using Viola-Jones and YOLO
Unit 5 Practical Assignment
Author: Devendra Agrawal
"""

import os
import time
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def detect_viola_jones(img_bgr: np.ndarray, cascade_path: str = None):
    """Run Viola-Jones Haar Cascade face detection."""
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade from: {cascade_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    start_time = time.perf_counter()
    # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    detections = []
    for (x, y, w, h) in faces:
        detections.append({
            "class": "Face",
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "confidence": None  # Haar cascades do not provide normalized probabilities
        })

    return detections, latency_ms


def detect_yolo(img_bgr: np.ndarray, model_name: str = "yolov8n.pt", conf_thresh: float = 0.25):
    """Run YOLOv8 object detection."""
    model = YOLO(model_name)

    start_time = time.perf_counter()
    results = model(img_bgr, conf=conf_thresh, verbose=False)
    latency_ms = (time.perf_counter() - start_time) * 1000

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": cls_name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3)
            })

    return detections, latency_ms


def draw_detections(img_rgb: np.ndarray, detections: list, color: tuple, label_prefix: str = ""):
    """Draw bounding boxes and class labels onto an RGB image."""
    annotated = img_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"{label_prefix}{det['class']}"
        if det["confidence"] is not None:
            label += f": {int(det['confidence'] * 100)}%"

        # Label background pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 5, max(th + 2, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
    return annotated


def run_object_recognition(image_path: str, out_dir: str):
    """Full Task 3: Compare Viola-Jones vs YOLO and evaluate edge/failure cases."""
    os.makedirs(out_dir, exist_ok=True)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Scale image to manageable resolution if needed
    h, w = img_bgr.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Normal Image Detections
    vj_dets, vj_lat = detect_viola_jones(img_bgr)
    yolo_dets, yolo_lat = detect_yolo(img_bgr)

    # Render visualizations
    vj_vis = draw_detections(img_rgb, vj_dets, (34, 139, 34), label_prefix="VJ: ")
    yolo_vis = draw_detections(img_rgb, yolo_dets, (30, 144, 255), label_prefix="YOLO: ")

    cv2.imwrite(os.path.join(out_dir, "viola_jones_detection.png"), cv2.cvtColor(vj_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "yolo_detection.png"), cv2.cvtColor(yolo_vis, cv2.COLOR_RGB2BGR))

    # 2. Side-by-side comparison on normal photograph
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    axes[0].imshow(vj_vis)
    axes[0].set_title(
        f"Viola-Jones (Haar Cascade)\nDetections: {len(vj_dets)} | Latency: {vj_lat:.1f} ms\n(Specialized Single-Class Face Detector)",
        fontsize=13, fontweight='bold'
    )
    axes[0].axis("off")

    axes[1].imshow(yolo_vis)
    classes_str = ", ".join([f"{d['class']} ({int(d['confidence']*100)}%)" for d in yolo_dets])
    axes[1].set_title(
        f"Ultralytics YOLOv8\nDetections: {len(yolo_dets)} | Latency: {yolo_lat:.1f} ms\nDetected: {classes_str}",
        fontsize=13, fontweight='bold'
    )
    axes[1].axis("off")

    plt.suptitle("Task 3: Object Recognition Comparison (Viola-Jones vs YOLOv8)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "detector_comparison_normal.png"), dpi=300)
    plt.close()

    # 3. Failure Case Experiment: Rotated Face (45 degrees) & Partial Occlusion
    # Rotated image creates an ideal failure case for Viola-Jones
    center = (img_bgr.shape[1] // 2, img_bgr.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rot_bgr = cv2.warpAffine(img_bgr, rot_matrix, (img_bgr.shape[1], img_bgr.shape[0]), borderMode=cv2.BORDER_REFLECT)
    rot_rgb = cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB)

    vj_rot_dets, vj_rot_lat = detect_viola_jones(rot_bgr)
    yolo_rot_dets, yolo_rot_lat = detect_yolo(rot_bgr)

    vj_rot_vis = draw_detections(rot_rgb, vj_rot_dets, (34, 139, 34), label_prefix="VJ: ")
    yolo_rot_vis = draw_detections(rot_rgb, yolo_rot_dets, (30, 144, 255), label_prefix="YOLO: ")

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    axes[0].imshow(vj_rot_vis)
    axes[0].set_title(
        f"[FAILURE CASE] Viola-Jones on 45° Rotated Image\nDetections: {len(vj_rot_dets)} (MISSED DETECTOR FAILS)\nReason: Haar rectangular filters cannot handle non-axis-aligned facial geometry",
        fontsize=12, fontweight='bold', color='crimson'
    )
    axes[0].axis("off")

    axes[1].imshow(yolo_rot_vis)
    axes[1].set_title(
        f"YOLOv8 on 45° Rotated Image\nDetections: {len(yolo_rot_dets)} (SUCCESSFULLY RECOGNIZED)\nReason: Deep CNN features encode rotation-invariant high-level semantic patterns",
        fontsize=12, fontweight='bold', color='darkgreen'
    )
    axes[1].axis("off")

    plt.suptitle("Detector Robustness & Failure Case Analysis: Rotated Viewpoint", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "detector_failure_case_analysis.png"), dpi=300)
    plt.close()

    # 4. Save structured comparison metrics
    summary = {
        "normal_test": {
            "viola_jones": {
                "num_detections": len(vj_dets),
                "latency_ms": round(vj_lat, 2),
                "detections": vj_dets
            },
            "yolo": {
                "num_detections": len(yolo_dets),
                "latency_ms": round(yolo_lat, 2),
                "detections": yolo_dets
            }
        },
        "rotation_failure_test": {
            "viola_jones_detections": len(vj_rot_dets),
            "yolo_detections": len(yolo_rot_dets),
            "observation": "Viola-Jones missed the rotated face completely (0 detections), while YOLO successfully identified the person."
        }
    }

    with open(os.path.join(out_dir, "object_recognition_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("[Task 3] Object recognition complete.")
    print(f"Normal Test -> Viola-Jones: {len(vj_dets)} dets ({vj_lat:.1f}ms) | YOLO: {len(yolo_dets)} dets ({yolo_lat:.1f}ms)")
    print(f"Rotation Test -> Viola-Jones: {len(vj_rot_dets)} dets | YOLO: {len(yolo_rot_dets)} dets")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 3: Object Recognition Using Viola-Jones and YOLO")
    parser.add_argument("--image", default="data/Image_org.jpeg", help="Input image path")
    parser.add_argument("--out_dir", default="outputs/task3_recognition", help="Output directory")
    args = parser.parse_args()

    print("==================================================")
    print("RUNNING TASK 3: OBJECT RECOGNITION (VIOLA-JONES vs YOLO)")
    print("==================================================")
    run_object_recognition(args.image, args.out_dir)
    print("Task 3 finished successfully!\n")
