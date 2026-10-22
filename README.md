# ☀️ Solar Panel Thermal Anomaly Detection & Segmentation

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo%20UI-ff7c00.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision and Deep Learning framework designed for **automated thermographic inspection of photovoltaic (PV) solar installations**. Powered by **YOLOv8 Instance Segmentation**, this project detects, classifies, and precisely segments thermal anomalies across solar modules to optimize preventative maintenance and protect solar farms from catastrophic failure.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [Dataset Architecture & Taxonomy](#-dataset-architecture--taxonomy)
- [Benchmark Results & Model Comparison](#-benchmark-results--model-comparison)
- [Project Architecture](#-project-architecture)
- [Installation & Quickstart](#-installation--quickstart)
- [Interactive Web Demo](#-interactive-web-demo)
- [Google Colab Workflow](#-google-colab-workflow)
- [Limitations & Future Roadmap](#-limitations--future-roadmap)
- [References & Acknowledgements](#-references--acknowledgements)

---

## 🔭 Executive Overview

Thermal anomalies in photovoltaic modules—such as localized hot spots, failing bypass diodes, reverse polarity issues, and string outages—cause significant power loss, accelerate cell degradation, and present severe fire hazards if left unaddressed.

Traditional manual thermography is slow, error-prone, and expensive. This project delivers an automated pipeline that:
1. Ingests thermal infrared (TIR) images captured via UAVs/drones or handheld thermographic sensors.
2. Applies a consolidated **5-class defect taxonomy** to resolve class fragmentation found in raw datasets.
3. Predicts both bounding box coordinates and **pixel-accurate segmentation masks** using **YOLOv8s-seg**.
4. Provides maintenance teams with an **instant diagnostic triage report** indicating fault severity and recommended corrective actions.

---

## ✨ Key Features

- **Pixel-Accurate Mask Segmentation:** Employs polygon-level segmentation to outline the exact geometry of defects, enabling precise distinction between cell-level cracks and bypass diode string heating.
- **Taxonomy Consolidation:** Re-engineered the original 17-class raw annotations into a unified 5-class standard, resolving severe sub-class fragmentation while preserving critical fault indicators.
- **Benchmarked Architectures:** Controlled comparison between lightweight edge inference (**YOLOv8n-seg**) and high-accuracy deployment (**YOLOv8s-seg**).
- **Interactive Gradio Dashboard:** Plug-and-play web interface with adjustable confidence thresholds and sample test cases.
- **Cloud-Ready Training Pipelines:** Complete Google Colab notebooks configured for Tesla T4 GPU acceleration.

---

## 📊 Dataset Architecture & Taxonomy

* **Dataset Source:** [Roboflow Universe: Thermal Anomalies — Solar Panel Data Set](https://universe.roboflow.com/solar-panel-data-set/thermal-anomalies-jev7w)
* **Dataset Scope:** **1,013 high-resolution thermal images** and **3,127 defect instances**.
* **Integrity Audit:** 0 corrupt images, 0 out-of-bounds coordinates, and complete polygon coordinate validation.

### Split Distribution

| Split | Images | Annotations | Percentage |
| :--- | :---: | :---: | :---: |
| **Train** | 711 | 2,187 | 70.2% |
| **Validation** | 202 | 641 | 19.9% |
| **Test** | 100 | 299 | 9.9% |
| **Total** | **1,013** | **3,127** | **100.0%** |

*(Note: 60 images across splits represent normal / negative background panels without defect labels).*

### Consolidated 5-Class Defect Taxonomy

The raw annotations contained 17 fragmented sub-classes (e.g., *High- Hot Spot*, *Medium- Hot Spot*, *Mexium- Hot Spot*, *Cell Low*). These were harmonized into 5 standardized industrial categories:

| ID | Class Name | Train | Valid | Test | Total | Share (%) | Primary Root Cause |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Hotspot** | 602 | 213 | 112 | **927** | 29.64% | Microcracks, localized solder fatigue, soiling |
| **1** | **Diode Failure** | 1,421 | 367 | 162 | **1,950** | 62.36% | Short-circuited or open junction bypass diodes |
| **2** | **Reverse Polarity Failure** | 60 | 15 | 4 | **79** | 2.53% | Inverted module connection in DC string harness |
| **3** | **String Outage** | 95 | 46 | 21 | **162** | 5.18% | Inverter disconnect, DC fuse blow, or severed cabling |
| **4** | **Shading** | 9 | 0 | 0 | **9** | 0.29% | External foliage, bird drops, structural shadow |

---

## 📈 Benchmark Results & Model Comparison

Both models were evaluated on the held-out test split (100 images, 299 defect instances):

| Metric | YOLOv8n-seg (Baseline) | YOLOv8s-seg (Improved) | Relative Delta |
| :--- | :---: | :---: | :---: |
| **Box Precision** | 0.456 | **0.575** | `+11.9%` 🟢 |
| **Box Recall** | **0.687** | 0.627 | `-6.0%` |
| **Box mAP@50** | 0.583 | **0.610** | `+2.7%` 🟢 |
| **Box mAP@50-95** | 0.316 | **0.347** | `+3.1%` 🟢 |
| **Mask Precision** | **0.613** | 0.466 | `-14.7%` |
| **Mask Recall** | 0.407 | **0.488** | `+8.1%` 🟢 |
| **Mask mAP@50** | 0.413 | **0.436** | `+2.3%` 🟢 |
| **Mask mAP@50-95** | 0.203 | **0.211** | `+0.8%` 🟢 |

**Key Takeaways:**
- **YOLOv8s-seg** delivers superior overall detection and mask localization metrics (`+2.7%` Box mAP@50, `+2.3%` Mask mAP@50).
- The improved model demonstrates a significantly higher **Mask Recall (+8.1%)**, reducing the risk of missed thermal defects in field inspections.

---

## 🗂️ Project Architecture

```text
Solar_Panel_Thermal_Anomaly/
│
├── Solar_Panel_Thermal_Anomaly_Detection_Colab.ipynb      # Complete Colab end-to-end training pipeline
├── Solar_Panel_Thermal_Anomaly_Dataset_Verification.ipynb  # Comprehensive data verification & audit suite
├── app.py                                                # Interactive Gradio web demonstration
├── requirements.txt                                      # Project environment dependencies
├── .gitignore                                            # Cache and log filters
│
├── models/
│   └── yolov8s_seg_best.pt                              # Fine-tuned YOLOv8s segmentation checkpoint
│
└── sample_images/                                        # Sample thermal images for instant testing
    ├── image-0126_jpg.rf.be4...jpg
    ├── solar_color1049_jpg.rf.08b...jpg
    └── ... (20 sample test panels)
```

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/1234yokeshwaran-bit/Solar_Panel_Thermal_Anomaly.git
cd Solar_Panel_Thermal_Anomaly
```

### 2. Set Up Python Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Interactive Web Demo

Launch the local Gradio dashboard to upload thermal images and inspect real-time segmentation masks:

```bash
python app.py
```

Once running, navigate to `http://localhost:7860` in your web browser. You can select pre-loaded sample images from `sample_images/` or upload your own thermographic captures.

---

## ☁️ Google Colab Workflow

For cloud training on an NVIDIA GPU:
1. Upload `Solar_Panel_Thermal_Anomaly_Detection_Colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Enable GPU acceleration under **Runtime → Change runtime type → T4 GPU**.
3. Run all cells sequentially to reproduce the data consolidation, train the baseline and improved models, and generate evaluation figures.

---

## ⚠️ Limitations & Future Roadmap

1. **Class Imbalance:** `Diode Failure` accounts for 62.36% of annotations, while `Shading` has only 9 instances. Future iterations will benefit from synthetic data augmentation and targeted sampling for minority classes.
2. **Environmental Artifacts:** False positives may occasionally occur from high-emissivity non-panel objects (e.g., metallic frames or ground reflections).
3. **Multi-Spectral Fusion:** Future enhancements plan to incorporate paired RGB visible-spectrum imagery alongside thermal infrared (TIR) channels for joint anomaly confirmation.

---

## 📚 References & Acknowledgements

- **Ultralytics YOLOv8:** [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- **Dataset Source:** Roboflow Universe [Thermal Anomalies — Solar Panel Data Set](https://universe.roboflow.com/solar-panel-data-set/thermal-anomalies-jev7w)
- **Frameworks:** [PyTorch](https://pytorch.org/), [Gradio](https://gradio.app/), [OpenCV](https://opencv.org/)