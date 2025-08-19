# Automated Video Segmentation for MATLAB

**Author:** Mohit Goyal, Angel Gupta  
_A submission for the MATLAB AI Challenge 2025_

## 🚀 Project Overview

Manually creating pixel-perfect segmentation masks for video is a notoriously slow, expensive, and tedious process. This project solves this problem by implementing a series of advanced automation algorithms for the MATLAB® Video Labeler app.

The final result is a hybrid, multi-class system that combines the power of **YOLOv4 deep learning detection** with a **multi-stage classical computer vision pipeline** for mask refinement. The system automatically generates high-quality, temporally consistent pixel masks for multiple object classes, demonstrating a significant reduction in manual labeling time.

![Final Algorithm Demo](media/demo.gif)


---

## ✨ Key Features

* **🎯 Fully Automatic:** Detects and segments objects with no human intervention required.
* **🤖 Multi-Class Capability:** Intelligently segments multiple classes including `car`, `truck`, `bus`, `person`, and `bicycle` by matching them to user-defined `PixelLabels`.
* **✨ High-Quality Masks:** A four-stage refinement pipeline (Detection → Initial Masking → Morphological Cleaning → Active Contour Refinement) produces clean, smooth, and accurate pixel-level masks.
* **⏱️ Proven Time Savings:** The final algorithm demonstrates a quantifiable and significant reduction in the time required for video annotation.

---

## 🛠️ The Algorithms: An Iterative Journey

This repository contains the complete evolution of our solution, located under the `+vision/+labeler/` directory.

1. **Baseline 1: The Simple Tracker (`PropagateWithFlow.m`)**
    * A semi-automated algorithm using optical flow to track a manually-selected ROI.
    * **Conclusion:** Fast but not robust, and incapable of automatic detection.

2. **Baseline 2: The Deep Learning Detector (`VehicleTracker_Stable.m`)**
    * A fully automatic system using YOLOv4 and a Kalman Filter to detect and track vehicles.
    * **Conclusion:** Robust and accurate at tracking, but only produced imprecise bounding boxes.

3. **Final Algorithm: The Hybrid Segmenter (`HybridSegmentation_Final.m`)**
    * The final, presentation-quality algorithm. It combines the strengths of the previous baselines to deliver a complete, multi-class segmentation solution.

---

## ⚙️ Setup and Usage

### **Requirements**

* MATLAB R2025a (or newer)
* Computer Vision Toolbox™
* Deep Learning Toolbox™
* Image Processing Toolbox™
* **Required Add-On:** _Deep Learning Toolbox Model for YOLO v4 CSP-Darknet53-COCO Network_

### **1. Installation**

1. Clone this repository to your local machine.
2. Install the required **YOLOv4 Add-On**. You can do this from the MATLAB **HOME** tab by clicking **Add-Ons > Get Add-Ons** and searching for "YOLO v4".
3. Launch MATLAB.

### **2. Running the Project**

1. Open MATLAB and navigate to the root folder of this repository.
2. Run the included setup script in the Command Window. This will configure the MATLAB path and launch the Video Labeler app for you.

    ```matlab
    setup_environment
    ```

3. Inside the Video Labeler app:
    * Load a video source (e.g., `visiontraffic.avi` or `viptrain.avi`).
    * Create `PixelLabel` definitions for the classes you want to detect (e.g., "car", "person"). **The names must be an exact match.**
    * Navigate to the **Automate** tab.
    * Click **Select Algorithm ▸ Refresh list**.
    * Choose an algorithm from the list (e.g., **"Hybrid Segmentation (Presentation Quality)"**).
    * Select a time range on the timeline and click **RUN**.

---

## 📊 Results: Quantifying the Impact

To evaluate the effectiveness of our final algorithm, we performed a time trial on a 30-second clip from the `visiontraffic.avi` video.

| Method | Time to Label 30s Clip |
| :--- | :--- |
| Manual Pixel Labeling | **18 minutes 45 seconds** |
| **Automated Labeling** (Run Algorithm + 90s for corrections) | **2 minutes 40 seconds** |

This represents an **85.7% reduction in labeling time**, a massive improvement in efficiency.

For a complete technical breakdown and analysis, please see the full [**PROJECT REPORT**](REPORT.MD).

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
