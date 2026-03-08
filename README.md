# Multi-Object Apparel Detection and Instance Segmentation

A visual recognition system for multi-label clothing classification, detection, and instance segmentation on the DeepFashion2 dataset.

## Overview

This project builds a multi-task visual recognition pipeline for apparel images, addressing three core problems:

| Task | Description |
|------|-------------|
| **Classification** | Multi-label image-level prediction of clothing categories |
| **Detection** | Bounding box localization and category classification per instance |
| **Segmentation** | Pixel-level instance masks for each clothing item |

All models are trained on the **top 5 most frequent clothing categories** from the DeepFashion2 dataset, using both **training from scratch** and **transfer learning** strategies.

---

## Dataset

**DeepFashion2** — a large-scale person-centric clothing dataset with item-level annotations.

### Category IDs

| ID | Category |
|----|----------|
| 1 | Short Sleeve Top |
| 2 | Long Sleeve Top |
| 3 | Short Sleeve Outwear |
| 4 | Long Sleeve Outwear |
| 5 | Vest |
| 6 | Sling |
| 7 | Shorts |
| 8 | Trousers |
| 9 | Skirt |
| 10 | Short Sleeve Dress |
| 11 | Long Sleeve Dress |
| 12 | Vest Dress |
| 13 | Sling Dress |

Each annotation JSON provides: `category_id`, `bounding_box`, `segmentation` polygons, `landmarks`, `scale`, `occlusion`, `zoom_in`, and `viewpoint`.

---

## Project Structure

```
project/
├── preprocessing.py              # Dataset scanning, filtering, splitting
├── processed_dataset/
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── label_map.json
├── classification/
│   ├── resnet50.ipynb               # ResNet-50 (scratch + transfer)
│   ├── efficientnet.ipynb           # EfficientNet-B0 (scratch + transfer)
│   └── mobilenet.ipynb              # MobileNetV3 (scratch + transfer)
└── detection/
    ├── yolo.ipynb                   # YOLOv8 (detection + segmentation)
    ├── maskrcnn.ipynb               # Mask R-CNN with ResNet-50 + FPN
    └── unet.ipynb                   # U-Net with connected component post-processing
```

---

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```
---

## Preprocessing

The preprocessing pipeline (`preprocessing.py`) performs the following steps:

1. **Scan** all annotation files across train / validation / test splits
2. **Count** category frequencies and select the **top 5** most common categories
3. **Filter** images containing at least one top-5 category
4. **Build** multi-label binary vectors per image
5. **Balanced subset sampling** — per-class quota sampling to address class imbalance (40% of full dataset)
6. **Split** into train / val / test (70% / 15% / 15%)
7. **Save** splits as JSON files in `processed_dataset/`

```bash
python preprocessing.py
```

Output:
```
processed_dataset/
├── train.json
├── val.json
├── test.json
└── label_map.json
```

---

## Task 3.1 — Classification

Multi-label classification using **sigmoid output** and **BCEWithLogitsLoss**. Each model is trained under two strategies:

| Model | Scratch | Transfer Learning |
|-------|---------|-------------------|
| ResNet-50 | ✔ | ✔ |
| EfficientNet-B0 | ✔ | ✔ |
| MobileNetV3 | ✔ | ✔ |

**Total: 6 experiments**

### Training

```bash
# ResNet-50
python classification/resnet50.py --strategy scratch
python classification/resnet50.py --strategy transfer

# EfficientNet-B0
python classification/efficientnet.py --strategy scratch
python classification/efficientnet.py --strategy transfer

# MobileNetV3
python classification/mobilenet.py --strategy scratch
python classification/mobilenet.py --strategy transfer
```

### Architecture Notes

- **Output layer**: `Linear → Sigmoid` (one logit per class)
- **Loss**: `BCEWithLogitsLoss`
- **Transfer learning**: pretrained ImageNet weights, initial layers frozen

---

## Task 3.2 — Detection & Segmentation

Given an image, predict bounding boxes, category labels, and pixel-level segmentation masks for each clothing instance.

| Model | Backbone | Outputs |
|-------|----------|---------|
| **YOLOv8** | CSPDarknet + FPN/PAN | Boxes, labels, instance masks |
| **Mask R-CNN** | ResNet-50 + FPN | Boxes, labels, instance masks |
| **U-Net** | CNN encoder (ResNet-34 optional) | Semantic masks → instances via connected components |

### Training

```bash
# YOLOv8
python detection/yolo.py

# Mask R-CNN
python detection/maskrcnn.py

# U-Net
python detection/unet.py
```

### U-Net Post-Processing

After generating semantic segmentation maps, instance-level results are extracted via:
1. **Connected component analysis** to separate individual items
2. **Bounding box extraction** from each component
3. **Category assignment** via majority voting over the segmentation map

---

## Evaluation Metrics

### Classification (Task 3.1)
- Per-class **Precision**, **Recall**, **F1-score**
- **Macro-averaged** and **Micro-averaged** F1
- Per-class **ROC curves** and **AUC**

### Segmentation (Task 3.2)
- **Mean IoU (mIoU)** — per class and macro-averaged
- **Dice coefficient** (segmentation F1)

### Detection (Task 3.2)
- **COCO-style mAP@[0.5:0.95]**
- Per-class **ROC curves**, **AUC**, and **F1-score**

---

## Results

### Classification

| Model | Strategy | Micro F1 | Macro F1 | AUC (avg) |
|-------|----------|----------|----------|-----------|
| ResNet-50 | Scratch | — | — | — |
| ResNet-50 | Transfer | — | — | — |
| EfficientNet-B0 | Scratch | — | — | — |
| EfficientNet-B0 | Transfer | — | — | — |
| MobileNetV3 | Scratch | — | — | — |
| MobileNetV3 | Transfer | — | — | — |

### Detection & Segmentation

| Model | mAP@[0.5:0.95] | mIoU | Dice |
|-------|----------------|------|------|
| YOLOv8 | — | — | — |
| Mask R-CNN | — | — | — |
| U-Net | — | — | — |

> Results will be populated after training runs are complete.

---

## Compute

All experiments are designed to run on **Kaggle** or **Google Colab** (single GPU). Large models exceeding 7B parameters are explicitly excluded per project constraints.
