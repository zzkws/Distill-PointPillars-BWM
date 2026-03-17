# Distill-PointPillars-BWM
A lightweight 3D object detection framework using cross-architecture knowledge distillation and Beam-Wise Mixing (BWM) based on OpenPCDet.

# Consistency-Regularized 3D Object Detection via BWM and Knowledge Distillation

[![Framework](https://img.shields.io/badge/Framework-OpenPCDet-blue.svg)](https://github.com/open-mmlab/OpenPCDet)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes-green.svg)](https://www.nuscenes.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Introduction
[cite_start]This repository contains the official implementation of the paper/project: **CONSISTENCY-REGULARIZED MULTI-MODAL 3D OBJECT DETECTION FOR AUTONOMOUS DRIVING VIA BEAM-WISE MIXING AND CROSS-SENSOR KNOWLEDGE DISTILLATION**[cite: 1, 2, 9, 10, 11]. 

[cite_start]This project aims to solve the "impossible triangle" of robust perception in autonomous driving[cite: 101]. [cite_start]We propose a Teacher-Student architecture distillation training framework based on OpenPCDet[cite: 135, 136]. [cite_start]By distilling knowledge from a complex Teacher model (CenterPoint) to a lightweight Student model (PointPillar), we achieve significant performance improvements without increasing the inference computational cost of the Student model[cite: 139, 303, 304, 306].

## ✨ Core Highlights
[cite_start]Our proposed training framework consists of three main parts[cite: 57]:
1. [cite_start]**Cross-Architecture Knowledge Distillation**: Efficient feature-level distillation allowing the anchor-based PointPillar to learn spatial geometric intuition from the anchor-free CenterPoint[cite: 138, 304, 306, 318].
2. [cite_start]**Feature-Level Beam-Wise Mixing (BWM)**: Inspired by LiDAR scanning properties, this module interleaves feature maps row-wise to create adversarial interference, forcing robust learning[cite: 141, 142, 308, 309, 310].
3. [cite_start]**Consistency Regularization**: Ensures the student model focuses on geometric features rather than rote memorizing context, maintaining consistent predictions under BWM interference[cite: 144, 314, 333].

## 📊 Experimental Results (nuScenes Dataset)
[cite_start]The experimental results on the nuScenes dataset demonstrate the effectiveness of our proposed framework[cite: 58, 386]. 

| Model / Configuration | mAP ↑ | NDS ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | mAVE ↓ | mAAE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (PointPillar) | 0.3461 | 0.4951 | 0.3843 | 0.2695 | 0.4295 | 0.4947 | 0.2009 |
| Ours (Distillation) | 0.3994 | 0.5391 | 0.3721 | 0.2663 | 0.4257 | 0.3444 | 0.1973 |
| **Ours (Distillation + BWM)** | **0.4070** | **0.5468** | **0.3670** | **0.2625** | **0.3884** | 0.3504 | 0.1987 |
| *Teacher (CenterPoint)* | *0.5003* | *0.6071* | *0.3113* | *0.2604* | *0.4288* | *0.2389* | *0.1914* |

[cite_start]*(Note: Results are evaluated on the NuScenes validation set[cite: 357, 359].)*

## 🚀 Pre-trained Models & Weights
You can download the trained model weights from the link below:
* [Google Drive Link to Model Weights](https://drive.google.com/drive/folders/1EB3W9-JAxQtWEb7H3HVTq44JakdIRht_?usp=sharing)

## 🛠️ Quick Start
### 1. Environment Setup
[cite_start]This code is built upon the OpenPCDet toolbox[cite: 435]. Please follow their official [installation guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) to set up the environment.
* [cite_start]Python 3.10+ [cite: 363]
* [cite_start]PyTorch 2.1.2 [cite: 363]
* [cite_start]CUDA 11.8 [cite: 363]

### 2. Dataset Preparation
Please download the [nuScenes dataset](https://www.nuscenes.org/nuscenes) and organize it according to the OpenPCDet dataset structure guidelines.

### 3. Usage
*(Add brief instructions here on how to run your `distill_pointpillar.py` or training YAML configs)*
```bash
# Example command to run the distillation training:
python tools/train.py --cfg_file configs/distill_pointpillar.yaml