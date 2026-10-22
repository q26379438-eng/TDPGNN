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
We conduct experiments on MS‑COCO Keypoints 2017 and CrowdPose.
Please download the datasets and organize them as follows:
data/
├── coco/
│   ├── annotations/
│   ├── train2017/
│   └── val2017/
└── crowdpose/
Modify the dataset root path in the corresponding yaml config files according to your local environment.

```
## Training

### First stage: feature distillation + skeleton‑aware pose distillation.
```bash
python train_pose.py --cfg xxx.yaml
Total epochs: 210
```
### Stage‑2 IGP‑GCN Refinement
Freeze the student backbone trained in Stage‑1, and only optimize the IGP‑GCN module with multi‑level progressive distillation supervision.
```bash
#python train_pose.py --cfg xxx.yaml
Training epochs: 30
Initial learning rate: 1e‑3

```
## Inference
```bash
Single‑image inference with built‑in latency, FPS and memory profiling.
# Ours (Two‑stage distillation + IGP‑GCN)
python infer_pose.py \
--cfg xxx.yaml \
--weight xxx.pth \
--img test.jpg

# Baseline: SimCC HRNet‑W32
python infer_pose.py \
--cfg xxx.ymal \
--weight xxx.pth \
--img test.jpg
```
### Output
- `xxx.jpg` — visualized pose skeleton image
- `xxx.json` — predicted 17 keypoint coordinates
### Profiling Metrics
- **Inference Latency (ms)**: Pure model forward pass only. Excludes image IO, preprocessing, postprocessing and visualization. 
- **End‑to‑end Runtime (ms)**: Full pipeline from image reading to result saving.
- **Peak GPU Memory (MB)**: Peak allocated memory occupied by model weights and intermediate feature maps.
## Evaluation
Evaluate on COCO val2017 or CrowdPose test set:
```bash
python xxx.py --cfg xxx.yaml --weight xxx.pth
```
## Main Results

| Method | Backbone | Input Size | Params (M) | GFLOPs | COCO val2017 AP (%) | CrowdPose‑test AP (%) |
|:-------|:---------|:-----------|:-----------|:-------|:--------------------|:----------------------|
| SimCC  | HRNet‑W32 | 256×192 | 28.5 | 7.8 | 75.3 | 66.7 |
| Ours   | HRNet‑W32 | 256×192 | 31.0 | 8.1 | **76.2** | **68.7** |

> More detailed metrics and ablation studies are provided in our paper.
## Citation
If you find this work useful in your research, please cite:
@article{ji2025coarsetofine,
  title   = {A Coarse-to-Fine Human Pose Estimation Method based on Two-stage Distillation and Progressive Graph Neural Network},
  author  = {Ji, Zhangjian and Zhang, Wenjin and Qiao, Shaotong and Feng, Kai and Qian, Yuhua},
  journal = {arXiv preprint arXiv:2508.11212},
  year    = {2025}
}
## Acknowledgement
This code is built upon SimCC. We thank the authors for their excellent open‑source implementation.
## License
This project is released under the MIT License. See LICENSE for details.
