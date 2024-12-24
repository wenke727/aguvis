# AGUVIS

<p align="center">
        📑 <a  href="https://huggingface.co/papers/2412.04454" target="_blank">Paper</a> &nbsp&nbsp  </a> | &nbsp&nbsp 🌐 <a href="https://aguvis-project.github.io/" target="_blank">Project Page</a> &nbsp&nbsp | &nbsp&nbsp 💾 <a href="https://huggingface.co/collections/xlangai/aguvis-agent-6764ae6d343c62af95b0a742" target="_blank"> AGUVIS Data Collection</a> &nbsp&nbsp
<br>

<p align="center">
    <img src="https://aguvis-project.github.io/static/images/overview.jpg" type="image/jpg"/>
<p>

## Introduction

AGUVIS is a unified pure vision-based framework for autonomous GUI agents that can operate across various platforms (web, desktop, mobile). Unlike previous approaches that rely on textual representations, AGUVIS leverages unified purely vision-based observations and a consistent action space to ensure better generalization across different platforms.

### Key Features & Contributions

- 🔍 **Pure Vision Framework**: First fully autonomous pure vision GUI agent capable of performing tasks independently without relying on closed-source models
- 🔄 **Cross-Platform Unification**: Unified action space and plugin system that works consistently across different GUI environments
- 📊 **Comprehensive Dataset**: Large-scale dataset of GUI agent trajectories with multimodal grounding and reasoning
- 🧠 **Two-Stage Training**: Novel training pipeline focusing on GUI grounding followed by planning and reasoning
- 💭 **Inner Monologue**: Explicit planning and reasoning capabilities integrated into the model training

Our framework demonstrates state-of-the-art performance in both offline and real-world online scenarios, offering a more efficient and generalizable approach to GUI automation.

https://github.com/user-attachments/assets/83f2c281-961c-4e2d-90dd-8cb1857adfb6

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone git@github.com:xlang-ai/aguvis.git
cd aguvis
```

2. Create and activate a conda environment:
```bash
conda create -n aguvis python=3.10
conda activate aguvis
```

3. Install PyTorch and dependencies:
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .
```

### Data Preparation

1. **Stage 1: Grounding**
   - Download the dataset from [aguvis-stage1](https://huggingface.co/datasets/xlangai/aguvis-stage1)
   - Place the data according to the structure defined in [`data/stage1.yaml`](./data/stage1.yaml)

2. **Stage 2: Planning and Reasoning**
   - Download the dataset from [aguvis-stage2](https://huggingface.co/datasets/xlangai/aguvis-stage2)
   - Place the data according to the structure defined in [`data/stage2.yaml`](./data/stage2.yaml)

### Training

1. Configure your training settings:
   - Open `scripts/train.sh`
   - Set the `SFT_TASK` variable to specify your training stage

2. Start training:
```bash
bash scripts/train.sh
```

## Checklist

- **Data**
  - ✅ Stage 1: Grounding Dataset
  - ✅ Stage 2: Planning and Reasoning Trajectories
- **Code**
  - ✅ Training Pipeline
  - 🚧 Model Weights and Configurations
  - 🚧 Inference Scripts
  - 🚧 Evaluation Toolkit

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{xu2024aguvis,
  title={Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction},
  author={Yiheng Xu and Zekun Wang and Junli Wang and Dunjie Lu and Tianbao Xie and Amrita Saha and Doyen Sahoo and Tao Yu and Caiming Xiong},
  year={2024},
  url={https://arxiv.org/abs/2412.04454}
}
```
