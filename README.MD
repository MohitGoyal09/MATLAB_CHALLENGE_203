# Automated Video Segmentation for MATLAB Labeler Apps

_Submission for the MATLAB AI Challenge 2025_

## Project Overview

Manually creating pixel-perfect segmentation masks for video is slow and tedious. This project implements automation algorithms for the MATLAB Video Labeler app to accelerate labeling. The final system combines deep-learning-based detection with classical computer-vision refinement to automatically generate high-quality pixel masks for multiple classes, significantly reducing manual labeling time.

## ✨ Key Features

- **Fully automatic**: Detects and segments objects without human intervention.
- **Multi-class**: Supports `car`, `truck`, `bus`, `person`, and `bicycle`.
- **High-quality masks**: Multi-stage refinement (thresholding, morphology, active contours) yields clean masks.
- **Built for Video Labeler**: Drop-in automation algorithms recognized by the app.

## 🚀 Demo

The hybrid segmentation algorithm detecting multiple objects and producing temporally consistent pixel masks.

![Algorithm Demo GIF](media/demo.gif)

## 🛠️ Algorithms in This Repo

Located under `+vision/+labeler/`:

1. `PropagateWithFlow.m` — Optical-flow-based label propagation (baseline).
2. `VehicleSegmentationAlgorithm.m` — Vehicle-focused segmentation using detection + refinement.
3. `HybridSegmentationAlgorithm.m` — Hybrid detector + classical refinement for per-frame masks.
4. `MultiHybridSegmentationAlgorithm.m` — Multi-class extension (cars, trucks, buses, people, bicycles).

## ⚙️ Setup and Usage

### Requirements

- MATLAB R2025a (or newer)
- Computer Vision Toolbox™
- Deep Learning Toolbox™
- Image Processing Toolbox™
- Optional: Deep Learning Toolbox Model for YOLO v4 CSP-Darknet53-COCO (if using YOLO-based variants)

### Install

1. Clone the repository.
2. Open MATLAB and set the current folder to the repo root.
3. Add the project to the MATLAB path:

    ```matlab
    addpath(genpath(pwd));
    savepath;
    ```

### Run in Video Labeler

1. Launch the app:

    ```matlab
    videoLabeler
    ```

2. In the app:
   - Load a video (e.g., `visiontraffic.avi`, `viptrain.avi`).
   - Create `PixelLabel` definitions for desired classes (e.g., "car", "person"). Names must match exactly.
   - Open the **Automate** tab, click **Select Algorithm ▸ Refresh list**.
   - Choose an algorithm (e.g., "Hybrid Segmentation" or "Multi-Class Hybrid Segmentation").
   - Select a time range and click **Run**.

## 📊 Results

If you benchmark labeling speed, record manual vs. automated times here to quantify time saved.

| Method | Time to Label 30s Clip | Time Saved |
| :--- | :--- | :--- |
| Manual Labeling | — | — |
| Automated Labeling | — | — |

For a full technical discussion, see the [project report](REPORT.MD).

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
