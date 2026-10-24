"""
Master Pipeline Runner: Executes All Computer Vision Tasks End-to-End
Practical Coding Assignment — Units 3, 4, and 5
Author: Devendra Agrawal
"""

import sys
import time
import os

from task1_restoration_segmentation import run_image_restoration, run_morphological_segmentation
from task2_feature_matching import run_feature_matching_pipeline
from task3_object_recognition import run_object_recognition
from task4_motion_estimation import run_motion_estimation, find_default_video


def main():
    print("=" * 70)
    print("  COMPUTER VISION PRACTICAL ASSIGNMENT (CCA-3): UNITS 3, 4 & 5")
    print("  Student Name: Devendra Agrawal")
    print("=" * 70)

    start_total = time.time()

    img1_path = "data/Image_org.jpeg"
    img2_path = "data/Image_2.jpeg"
    video_path = find_default_video()

    # Fallback checks
    if not os.path.exists(img1_path) and os.path.exists("Image_org.jpeg"):
        img1_path = "Image_org.jpeg"

    print(f"\n[Data Check]")
    print(f"  Primary Image:  {img1_path} (Exists: {os.path.exists(img1_path)})")
    print(f"  Second Image:   {img2_path} (Exists: {os.path.exists(img2_path)})")
    print(f"  Video Input:    {video_path} (Exists: {os.path.exists(video_path)})")

    # ----------------------------------------------------
    # TASK 1: Image Restoration and Segmentation
    # ----------------------------------------------------
    print("\n" + "#" * 70)
    print("  STAGE 1: IMAGE RESTORATION AND SEGMENTATION (UNIT 3)")
    print("#" * 70)
    t1_start = time.time()
    run_image_restoration(img1_path, "outputs/task1_restoration")
    run_morphological_segmentation(img1_path, "outputs/task1_segmentation")
    print(f">> Task 1 completed in {time.time() - t1_start:.2f} seconds.")

    # ----------------------------------------------------
    # TASK 2: Feature Extraction and Image Matching
    # ----------------------------------------------------
    print("\n" + "#" * 70)
    print("  STAGE 2: FEATURE EXTRACTION & SIFT MATCHING (UNIT 4)")
    print("#" * 70)
    t2_start = time.time()
    run_feature_matching_pipeline(img1_path, img2_path, "outputs/task2_features")
    print(f">> Task 2 completed in {time.time() - t2_start:.2f} seconds.")

    # ----------------------------------------------------
    # TASK 3: Object Recognition (Viola-Jones vs YOLOv8)
    # ----------------------------------------------------
    print("\n" + "#" * 70)
    print("  STAGE 3: OBJECT RECOGNITION USING VIOLA-JONES & YOLO (UNIT 5)")
    print("#" * 70)
    t3_start = time.time()
    run_object_recognition(img1_path, "outputs/task3_recognition")
    print(f">> Task 3 completed in {time.time() - t3_start:.2f} seconds.")

    # ----------------------------------------------------
    # TASK 4: Video Motion Estimation
    # ----------------------------------------------------
    print("\n" + "#" * 70)
    print("  STAGE 4: VIDEO MOTION ESTIMATION & TRACKING (UNIT 5)")
    print("#" * 70)
    t4_start = time.time()
    run_motion_estimation(video_path, "outputs/task4_motion")
    print(f">> Task 4 completed in {time.time() - t4_start:.2f} seconds.")

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"  ALL TASKS SUCCESSFULLY COMPLETED IN {total_time:.2f} SECONDS!")
    print(f"  All generated visualizations and metrics are available in:")
    print(f"    - outputs/task1_restoration/")
    print(f"    - outputs/task1_segmentation/")
    print(f"    - outputs/task2_features/")
    print(f"    - outputs/task3_recognition/")
    print(f"    - outputs/task4_motion/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
