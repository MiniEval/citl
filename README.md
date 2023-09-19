# Continous Intermediate Token Learning - PyTorch implementation

Official Pytorch implementation of the CVPR2023 paper: [Continuous Intermediate Token Learning with Implicit Motion Manifold for Keyframe Based Motion Interpolation](https://arxiv.org/abs/2303.14926)

# Prerequisites
- Python 3.10
- [PyTorch 1.13.1 with CUDA](https://github.com/pytorch/pytorch)
- [NumPy v1.24.2](https://github.com/numpy/numpy)
- [transforms3d 0.4.1](https://github.com/matthew-brett/transforms3d)

# Usage

train.py:
```python train.py ["cmu"/"lafan"] [dataset path]```

eval.py:
```python eval.py ["cmu"/"lafan"] [dataset path] [ckpt]```

A pre-trained CMU model can be found [here](https://drive.google.com/file/d/1T-Wus9Z1Mejo2Ywcv6WnS8dAsDEdwwlh/view?usp=sharing).