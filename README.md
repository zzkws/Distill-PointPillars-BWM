# Consistency-Regularized 3D Object Detection via BWM and Knowledge Distillation

[![Framework](https://img.shields.io/badge/Framework-OpenPCDet-blue.svg)](https://github.com/open-mmlab/OpenPCDet)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes-green.svg)](https://www.nuscenes.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Introduction
This repository contains the official implementation of the paper/project: **CONSISTENCY-REGULARIZED MULTI-MODAL 3D OBJECT DETECTION FOR AUTONOMOUS DRIVING VIA BEAM-WISE MIXING AND CROSS-SENSOR KNOWLEDGE DISTILLATION**. 

This project aims to solve the "impossible triangle" of robust perception in autonomous driving. We propose a Teacher-Student architecture distillation training framework based on OpenPCDet. By distilling knowledge from a complex Teacher model (CenterPoint) to a lightweight Student model (PointPillar), we achieve significant performance improvements without increasing the inference computational cost of the Student model.

## ✨ Core Highlights
Our proposed training framework consists of three main parts:
1. **Cross-Architecture Knowledge Distillation**: Efficient feature-level distillation allowing the anchor-based PointPillar to learn spatial geometric intuition from the anchor-free CenterPoint.
2. **Feature-Level Beam-Wise Mixing (BWM)**: Inspired by LiDAR scanning properties, this module interleaves feature maps row-wise to create adversarial interference, forcing robust learning.
3. **Consistency Regularization**: Ensures the student model focuses on geometric features rather than rote memorizing context, maintaining consistent predictions under BWM interference.

## 📊 Experimental Results (nuScenes Dataset)
The experimental results on the nuScenes dataset demonstrate the effectiveness of our proposed framework. 

| Model / Configuration | mAP ↑ | NDS ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | mAVE ↓ | mAAE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (PointPillar) | 0.3461 | 0.4951 | 0.3843 | 0.2695 | 0.4295 | 0.4947 | 0.2009 |
| Ours (Distillation) | 0.3994 | 0.5391 | 0.3721 | 0.2663 | 0.4257 | 0.3444 | 0.1973 |
| **Ours (Distillation + BWM)** | **0.4070** | **0.5468** | **0.3670** | **0.2625** | **0.3884** | 0.3504 | 0.1987 |
| *Teacher (CenterPoint)* | *0.5003* | *0.6071* | *0.3113* | *0.2604* | *0.4288* | *0.2389* | *0.1914* |

*(Note: Results are evaluated on the NuScenes validation set.)*

## 🚀 Pre-trained Models & Weights
You can download the trained model weights from the link below:
* [Google Drive Link to Model Weights](https://drive.google.com/drive/folders/1EB3W9-JAxQtWEb7H3HVTq44JakdIRht_?usp=sharing)

## 🛠️ Quick Start

### 1. Environment & Dataset Setup
This project is built upon the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) toolbox. 
1. Please follow their official [Installation Guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) to set up the Python environment (Python 3.10, PyTorch 2.1.2, CUDA 11.8).
2. Download the [nuScenes dataset](https://www.nuscenes.org/) and organize it according to the OpenPCDet dataset structure guidelines.

### 2. Code Integration
Integrate the files from this repository into your local OpenPCDet workspace:

1. **Model Scripts**: Copy `distill_pointpillar.py` and `consistency_distill_pointpillar.py` into the `pcdet/models/detectors/` directory of your OpenPCDet project.
2. **Register Models**: Open `pcdet/models/detectors/__init__.py` and import the new detector classes so the framework can recognize them:
   ```python
   # Add these lines to __init__.py
   from .distill_pointpillar import DistillPointPillar
   from .consistency_distill_pointpillar import ConsistencyDistillPP
   
   __all__ = {
       ...
       'DistillPointPillar': DistillPointPillar,
       'ConsistencyDistillPP': ConsistencyDistillPP,
   }
   ```
3. **Configurations**: Copy the YAML configuration files (`pointpillar.yaml`, `distill_pointpillar.yaml`, `consistency_distill_pointpillar.yaml`) into the `tools/cfgs/nuscenes_models/` directory.

### 3. Training
Navigate to the `tools` directory in your OpenPCDet workspace and start training.

**For Baseline PointPillar:**
```bash
python train.py --cfg_file cfgs/nuscenes_models/pointpillar.yaml
```

**For Teacher-Student Distillation:**
```bash
python train.py --cfg_file cfgs/nuscenes_models/distill_pointpillar.yaml
```

**For Distillation + Beam-Wise Mixing (Ours):**
```bash
python train.py --cfg_file cfgs/nuscenes_models/consistency_distill_pointpillar.yaml
```

## ✒️ Author
* **Zhou Zikang** 

## 📜 License
This project is released under the [MIT License](LICENSE).
