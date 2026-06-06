<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-4.41%2B-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
</p>

<h1 align="center">Chest X-Ray AI</h1>
<p align="center"><b>Multi-Label Thoracic Disease Detection with Vision Transformers</b></p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#results">Results</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

This repository contains the complete implementation of a **production-ready AI system** for automated multi-label classification of thoracic diseases from frontal chest X-ray images. Built on the **NIH ChestX-ray14** dataset (112,120 images), the system leverages a **Vision Transformer (ViT-B/16)** backbone pre-trained on ImageNet-21k to achieve robust generalization across 14 disease categories.

**Key Achievement:** Mean test ROC-AUC of **0.8301** across all 14 pathology classes, demonstrating clinical-grade discriminative performance.

> Developed as a Final Year Project in Computer Science / Artificial Intelligence.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔬 **ViT-B/16 Backbone** | Fine-tuned Vision Transformer with ImageNet-21k pre-training |
| 🏷️ **14-Class Multi-Label** | Simultaneous detection of Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax |
| 🎯 **Optimal Thresholding** | Per-class threshold optimization for maximum F1-score |
| 🌐 **Flask Web Interface** | Professional dark-themed UI with drag-and-drop image upload |
| 📊 **Confidence Visualization** | Animated bar charts with severity indicators |
| 📄 **PDF Report Generation** | Automated clinical report export via ReportLab |
| ⚡ **Lightweight Deployment** | Optimized for CPU inference with ~1GB model footprint |

---


### Model Configuration
- **Backbone:** `google/vit-base-patch16-224-in21k`
- **Hidden Dimension:** 768
- **Classifier:** 768 → 512 → 14 (with Dropout 0.3)
- **Loss Function:** Asymmetric Loss (ASL) with γ⁺=0, γ⁻=4
- **Optimizer:** AdamW with Layer-wise Learning Rate Decay
- **EMA:** Exponential Moving Average (decay=0.9999)

---

## Results

### Aggregate Performance
| Split | Mean AUC | Loss | Best Epoch |
|-------|----------|------|------------|
| **Test** | **0.8301** | 0.1880 | 8 |
| Validation | 0.8280 | — | 5 (Phase 2) |

### Per-Class Test AUC
| Pathology | AUC | Pathology | AUC |
|-----------|-----|-----------|-----|
| Atelectasis | 0.79 | Infiltration | 0.71 |
| Cardiomegaly | 0.90 | Mass | 0.86 |
| Consolidation | 0.79 | Nodule | 0.78 |
| Edema | 0.88 | Pleural Thickening | 0.79 |
| Effusion | 0.89 | Pneumonia | 0.75 |
| Emphysema | 0.93 | Pneumothorax | 0.88 |
| Fibrosis | 0.80 | Hernia | 0.91 |

*Full evaluation metrics, confusion matrices, and ROC curves are available in `docs/`.*

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/chestxray-ai.git
cd chestxray-ai/chestxray_frontend
pip install -r requirements.txt