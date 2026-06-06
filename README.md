# Chest X-Ray Disease Classification

[![Dashboard](https://img.shields.io/badge/Live%20Dashboard-View%20Now-5b8dee?style=for-the-badge&logo=html5&logoColor=white)](https://muhammedhasif7127-png.github.io/chestxray-classifier/dashboard.html)
[![Model Weights](https://img.shields.io/badge/Model%20Weights-Google%20Drive-4caf7d?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1NUDfRd2Xu7zE6_ZS5kIeQ6sYmucFgcy5/view?usp=drive_link)
[![Notebook](https://img.shields.io/badge/Train%20Notebook-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

Multi-label detection of **14 thoracic diseases** from chest X-rays using **Vision Transformer (ViT-B/16)** and two-phase transfer learning — Final Year Project by **Mohammed Hasif, June 2026**.

---

## Live Dashboard

> **[→ Open Dashboard](https://muhammedhasif7127-png.github.io/chestxray-classifier/dashboard.html)**

The dashboard is a fully standalone HTML file — **no server, no install, no backend needed**. Open it directly in any browser.

**What it shows:**
- Per-class AUC scores across all 14 disease classes
- Training curve across Phase 1 + Phase 2 (8 epochs)
- Dataset class frequency distribution
- Live inference demo — drag and drop any chest X-ray image
- Full model config, hyperparameters, and architecture

**Deploy for a permanent public URL (GitHub Pages — free):**

```bash
# Step 1 — make sure dashboard.html is in your repo root
git add dashboard.html
git commit -m "add dashboard"
git push

# Step 2 — enable GitHub Pages
# Go to: Settings → Pages → Branch: main → Save

# Step 3 — your live URL will be:
# https://muhammedhasif7127-png.github.io/chestxray-classifier/dashboard.html
```

Dashboard link is live at the URL above once GitHub Pages is enabled.

**Other hosting options:**

| Option | How |
|---|---|
| Netlify | Go to netlify.com → drag and drop `dashboard.html` → instant URL |
| Vercel | `npx vercel dashboard.html` |
| Local | Double-click `dashboard.html` — opens in browser, no server needed |

---

## Results

| Metric | Value |
|---|---|
| Mean ROC-AUC | **0.847** |
| Classes achieving ≥ 0.80 AUC | **11 / 14** |
| Best performing class | Hernia — 0.941 |
| Total parameters | 86M backbone + 10,766 classifier head |
| Training time | ~45 min on Google Colab T4 |

---

## Features

- Vision Transformer (ViT-B/16) backbone pretrained on ImageNet-21k
- Two-phase transfer learning (frozen backbone → full fine-tune)
- Weighted Binary Cross-Entropy for class imbalance
- Test-Time Augmentation (TTA × 4 passes)
- Mixed precision training (fp16 AMP)
- Early stopping with patience 3
- Flask frontend for local inference

---

## Diseases Detected

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

---

## Project Structure

```
chestxray-classifier/
├── dashboard.html              ← standalone dashboard (open in any browser)
├── app.py                      ← Flask inference server
├── model/
│   └── model_weights.pt        ← place downloaded weights here
├── templates/
│   └── index.html
├── static/
└── notebooks/
    └── chestxray_training.ipynb
```

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/muhammedhasif7127-png/chestxray-classifier.git
cd chestxray-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model Weights

Due to GitHub's 100MB file limit, model weights (~337MB) are hosted externally.

**Download:** [Google Drive (ZIP)](https://drive.google.com/file/d/1NUDfRd2Xu7zE6_ZS5kIeQ6sYmucFgcy5/view?usp=drive_link)

After downloading, extract and place `model_weights.pt` inside `model/`:

```
chestxray-classifier/model/model_weights.pt
```

> Alternatively, train from scratch using the Colab notebook in `notebooks/`.

### 4. Launch Application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser.

---

## Training

Open `notebooks/chestxray_training.ipynb` in Google Colab. The notebook covers:

- Kaggle API dataset download (NIH ChestX-ray14, 224×224 resized, ~2.3 GB)
- Data cleaning, label encoding, class imbalance analysis
- Phase 1 — frozen backbone, classifier head only, LR=1e-3, 3 epochs (~5 min/epoch)
- Phase 2 — full fine-tune, LR=2e-5, weight decay=0.01, 5 epochs (~10 min/epoch)
- Per-class AUC evaluation with early stopping

---

## Model Details

| Component | Detail |
|---|---|
| Backbone | ViT-Base/16 (ImageNet-21k pretrained) |
| Patch size | 16 × 16 |
| Hidden dim | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| Classifier head | Linear(768 → 14) + Sigmoid |
| Loss function | Weighted Binary Cross-Entropy |
| Phase 1 optimizer | Adam · LR 1e-3 |
| Phase 2 optimizer | AdamW · LR 2e-5 · WD 0.01 |
| LR scheduler | ReduceLROnPlateau |
| TTA passes | 4 |
| Target metric | Mean AUC ≥ 0.80 ✅ |

---

## Author

**Mohammed Hasif** · Final Year Project · June 2026

---

## License

MIT License. See `LICENSE` for details.
