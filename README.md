# A Coarse‑to‑Fine Human Pose Estimation Method based on Two‑stage Distillation and Progressive Graph Neural Network
Official PyTorch implementation for our coarse‑to‑fine two‑stage knowledge distillation human pose estimation framework.

> Human pose estimation is a fundamental task in computer vision. Existing knowledge‑distillation‑based pose methods rarely exploit the inherent structural topology among human joints.
We propose a two‑stage distillation framework built upon SimCC:
1. **First‑stage distillation**: Feature distillation + skeleton‑aware structure loss, transferring high‑level semantic knowledge from a strong teacher model (SimCC‑HRNet‑W48) to the student model (SimCC‑HRNet‑W32). A weight‑decay strategy is adopted for distillation loss to improve generalization.
2. **Second‑stage distillation**: Image‑Guided Progressive Graph Convolutional Network (**IGP‑GCN**) refines coarse pose outputs from stage‑1. Multi‑level progressive supervision from teacher final pose is applied to optimize the GCN module.

Our method achieves consistent gains on COCO Keypoints and CrowdPose datasets, and obtains larger improvements under crowded / heavy‑occlusion scenarios.

## Environment
```bash
conda create -n pose_env python=3.8
conda activate pose_env
pip install torch torchvision opencv-python pyyaml numpy
