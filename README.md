# A Coarse‑to‑Fine Human Pose Estimation Method based on Two‑stage Distillation and Progressive Graph Neural Network

[![arXiv](https://img.shields.io/badge/arXiv-2508.11212-b31b1b.svg)](https://arxiv.org/abs/2508.11212)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation for our two‑stage coarse‑to‑fine human pose estimation framework.

<p align="center">
<img src="https://github.com/user-attachments/assets/8390410d-f28a-46ee-98a2-3dadabbcdb17" width="920"/>
</p>
<p align="center"><b>Figure 1:</b> Overview of our two‑stage coarse‑to‑fine pose estimation framework.</p>

> This work introduces a two‑stage knowledge distillation framework for human pose estimation. We combine skeleton‑aware distillation loss and Image‑Guided Progressive Graph Convolutional Network (IGP‑GCN) to fully exploit human joint structural priors, achieving notable performance gains especially under crowded and occlusion scenarios.

## 📌 Highlights
- 🧠 **Two‑stage distillation pipeline**: Feature distillation + skeleton‑aware structure distillation in stage‑1; IGP‑GCN progressive pose refinement in stage‑2.
- 🦴 **Skeleton‑aware loss**: Explicitly models human joint topology during knowledge distillation.
- 🔍 **IGP‑GCN module**: Image‑guided progressive graph convolution for refining occluded / inaccurate keypoints.
- 📊 **Consistent improvements**: SOTA results on COCO Keypoints and CrowdPose benchmarks, superior performance for crowded scenes.
- ⚡ **Support latency & memory profiling**: Automatically compute inference latency, end‑to‑end runtime, FPS and peak GPU memory.

## 📋 Table of Contents
- [Environment](#environment)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Weights](#pretrained-weights)
- [Training](#training)
  - [Stage‑1 Distillation](#stage‑1-distillation)
  - [Stage‑2 IGP‑GCN Refinement](#stage‑2-igp-gcn-refinement)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Main Results](#main-results)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Environment
```bash
# create conda environment
conda create -n XXX python=3.8
conda activate XXX

# install dependencies
pip install torch torchvision opencv-python pyyaml numpy

```
## Dataset Preparation
```bash


