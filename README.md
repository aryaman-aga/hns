# Dual-Granularity Explainability for Medical Image Diagnosis

A research-grade medical imaging system implementing the paper:

> *"Enhancing Clinical Trust: A Multi-Modal Deep Learning Framework with Dual-Granularity Explainability for Medical Image Diagnosis"*  
> Aryaman Agarwal, Kanik Chawla, Sparsh Kalia — NSUT

**Two tasks, one architecture:**

| Task | Dataset | Classes | Baseline | Ours (SE-ResNet-18) |
|---|---|---|---|---|
| Chest X-Ray | PneumoniaMNIST | Normal / Pneumonia | 92.56% | TBD after training |
| Breast Ultrasound | BreastMNIST | Benign / Malignant | 88.46% | TBD after training |

**Architecture improvements over the paper:**
- SE (Squeeze-Excitation) attention after every residual block
- Cosine Annealing LR + Mixup augmentation + Label smoothing
- Class-balanced WeightedRandomSampler
- Gradient clipping for training stability

---

## Project Structure

```
hns/
├── src/
│   ├── dataset.py           # MedMNIST v2 loader + augmentations
│   ├── model.py             # SE-ResNet-18 architecture
│   ├── train.py             # Training loop (mixup, cosine LR, early stopping)
│   ├── evaluate.py          # Metrics + plots (AUC-ROC, F1, confusion matrix)
│   ├── explainability.py    # Grad-CAM + Integrated Gradients
│   └── generate_report.py  # PDF architecture report generator
├── .github/
│   ├── workflows/ci.yml     # GitHub Actions CI (forward-pass smoke test)
│   └── PULL_REQUEST_TEMPLATE.md
├── models/
│   ├── best_pneumonia.pth   # Trained weights — PneumoniaMNIST
│   ├── best_breast.pth      # Trained weights — BreastMNIST
│   └── test_metrics_*.json  # Per-task test results
├── data/                    # MedMNIST downloads (auto-created, gitignored)
├── test/                    # Sample images for the app gallery
├── app.py                   # Streamlit frontend (dual-task)
├── Montgomery.ipynb         # Research notebook
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Quick Start

### Step 1 — Clone / unzip the project

```bash
unzip lung-tb-detection.zip
cd hns
```

### Step 2 — Create virtual environment & install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **Apple Silicon (MPS)**: the CPU wheel above works on M-series Macs — MPS is detected automatically.

### Step 3 — Train the models

MedMNIST **downloads automatically** (~200MB total). No dataset setup needed.

```bash
source venv/bin/activate

# Train on PneumoniaMNIST (~5-10 min on MPS/GPU)
python -m src.train --task pneumonia --epochs 50 --batch_size 64

# Train on BreastMNIST (~2-3 min)
python -m src.train --task breast --epochs 50 --batch_size 32
```

Both models save to `models/best_<task>.pth`.

### Step 4 — Run the Streamlit app

```bash
source venv/bin/activate
streamlit run app.py
```

Open **http://localhost:8501** — select task, upload an image, choose XAI method.

### Step 5 — Generate architecture PDF

```bash
python -m src.generate_report
# → architecture_report.pdf
```

---

## Re-training the Model

```bash
# PneumoniaMNIST
python -m src.train --task pneumonia --epochs 50 --batch_size 64 --lr 3e-4

# BreastMNIST
python -m src.train --task breast --epochs 50 --batch_size 32 --lr 3e-4
```

---

## Evaluation

```bash
source venv/bin/activate
python -m src.evaluate
```

Outputs: AUC-ROC, F1, Precision, Recall, Dice coefficient, IoU, confusion matrix, and curve plots saved to `models/eval_plots/`.

---

## Research Notebook

Open `Montgomery.ipynb` for the full research pipeline:

```bash
source venv/bin/activate
jupyter notebook Montgomery.ipynb
```

The notebook covers:
1. Exploratory Data Analysis (class balance, age/gender distributions, sample images)
2. DataLoader construction
3. Model architecture overview + forward pass shapes
4. Training
5. Training curves
6. Test set evaluation with all metrics
7. Grad-CAM visualisations
8. Integrated Gradients visualisations
9. Side-by-side dual-granularity comparison figure (for paper)

---

## Architecture

```
Input (1×224×224 grayscale)
       │
  Modified Conv1  (7×7, s=2, 64ch — weight-averaged from ImageNet)
       │
  Layer1 — BasicBlock×2 + SE  →  64ch,  56×56
  Layer2 — BasicBlock×2 + SE  → 128ch,  28×28
  Layer3 — BasicBlock×2 + SE  → 256ch,  14×14
  Layer4 — BasicBlock×2 + SE  → 512ch,   7×7   ← Grad-CAM target
       │
  AdaptiveAvgPool2d → (512-d)
       │
  Dropout(0.3) → FC(128) → ReLU → Dropout(0.15) → FC(2)
       │
  Softmax → Prediction
```

**SE Block detail** (after every residual block):
```
Feature map (C×H×W)
  → GlobalAvgPool → (C,)
  → FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid → scale weights
  → Feature map ⊙ weights
```

---

## Explainability Methods

| Method | Granularity | Speed | Description |
|---|---|---|---|
| **Grad-CAM** | Coarse ROI | ~1s | Gradient-weighted activation map from denseblock4 |
| **Integrated Gradients** | Pixel-level PLI | ~5-15s | Captum IG with NoiseTunnel (SmoothGrad²) |

---

## Results

| Task | Dataset | Baseline (paper) | SE-ResNet-18 (ours) |
|---|---|---|---|
| Pneumonia detection | PneumoniaMNIST | 92.56% | After training |
| Breast malignancy | BreastMNIST | 88.46% | After training |

Run `python -m src.train` and check `models/test_metrics_*.json` for actual numbers.

---

## Citation

If you use this code or model in your research, please cite:
```
[Your paper citation here]
```

Dataset citation:
```
Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G.
Two public chest X-ray datasets for computer-aided screening of pulmonary diseases.
Quant Imaging Med Surg. 2014 Dec;4(6):475-7.
```
