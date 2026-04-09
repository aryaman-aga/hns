# MedXAI — Dual-Granularity Explainability for Medical Image Diagnosis

<p align="center">
  <strong>🧠 SE-ResNet-18 · 🔬 Grad-CAM & Integrated Gradients · 🤖 Ollama AI Explanations</strong>
</p>

A research-grade medical imaging system with a premium glassmorphism web interface, implementing the paper:

> *"Enhancing Clinical Trust: A Multi-Modal Deep Learning Framework with Dual-Granularity Explainability for Medical Image Diagnosis"*  
> Aryaman Agarwal, Kanik Chawla, Sparsh Kalia — NSUT

---

## ✨ Key Features

- **Dual-Granularity XAI** — Grad-CAM (region-level ROI) + Integrated Gradients (pixel-level attribution)
- **Role-Based Dashboards** — Separate interfaces for Medical Practitioners and Patients
- **AI Clinical Explanations** — Ollama-powered LLM generates human-readable analysis
- **Severity Assessment** — Automatic Critical / Moderate / Low risk classification
- **Patient Guidance** — "Consult Doctor" or "Low Risk" verdict with downloadable doctor brief
- **Premium UI** — Glassmorphism design, Framer Motion animations, responsive layout

### Supported Disease Models

| Task | Dataset | Classes | Baseline | Ours (SE-ResNet-18) |
|---|---|---|---|---|
| Chest X-Ray | PneumoniaMNIST | Normal / Pneumonia | 92.56% | TBD after training |
| Breast Ultrasound | BreastMNIST | Benign / Malignant | 88.46% | TBD after training |
| Chest X-Ray (14 Disease) | NIH Chest-14 | 14 Pathologies | — | Coming Soon |

### Architecture Improvements

- SE (Squeeze-Excitation) attention after every residual block
- Cosine Annealing LR + Mixup augmentation + Label smoothing
- Class-balanced WeightedRandomSampler
- Gradient clipping for training stability

---

## 📁 Project Structure

```
hns/
├── frontend/                    # React web application (Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx       # Glassmorphism navigation bar
│   │   │   ├── HeroSection.jsx  # Animated hero with floating cards
│   │   │   ├── HowItWorks.jsx   # 3-step connected timeline
│   │   │   ├── Features.jsx     # 6-card feature grid
│   │   │   ├── GlassCard.jsx    # Reusable glass effect card
│   │   │   ├── ScanUploader.jsx # Drag & drop with task/XAI selectors
│   │   │   ├── ResultsPanel.jsx # Dual-mode results (doctor/patient)
│   │   │   └── Footer.jsx       # Paper citation + NSUT branding
│   │   ├── pages/
│   │   │   ├── Landing.jsx      # Full landing page
│   │   │   ├── Auth.jsx         # Sign in / Sign up (dual role)
│   │   │   ├── PractitionerDashboard.jsx  # Doctor analysis view
│   │   │   └── PatientDashboard.jsx       # Patient guidance view
│   │   ├── context/
│   │   │   └── AuthContext.jsx  # Auth state management
│   │   ├── App.jsx              # Router + layout
│   │   ├── main.jsx             # Entry point
│   │   └── index.css            # Design system (glassmorphism tokens)
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── src/                         # ML backend modules
│   ├── dataset.py               # MedMNIST v2 loader + augmentations
│   ├── model.py                 # SE-ResNet-18 architecture
│   ├── train.py                 # Training loop (mixup, cosine LR, early stopping)
│   ├── evaluate.py              # Metrics + plots (AUC-ROC, F1, confusion matrix)
│   ├── explainability.py        # Grad-CAM + Integrated Gradients
│   └── generate_report.py       # PDF architecture report generator
│
├── models/                      # Trained model weights
│   ├── best_pneumonia.pth
│   ├── best_breast.pth
│   └── test_metrics_*.json
│
├── api.py                       # Flask REST API (serves ML to React frontend)
├── app.py                       # Streamlit frontend (legacy, still functional)
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Clone & install Python dependencies

```bash
git clone <repo-url>
cd hns

python -m venv venv
venv\Scripts\activate                # Linux/Mac: source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install flask flask-cors
```

### Step 2 — Train the models

MedMNIST downloads automatically (~200MB). No dataset setup needed.

```bash
# Pneumonia detection (~5-10 min on GPU)
python -m src.train --task pneumonia --epochs 50 --batch_size 64

# Breast cancer screening (~2-3 min)
python -m src.train --task breast --epochs 50 --batch_size 32
```

Both models save to `models/best_<task>.pth`.

### Step 3 — Install Ollama (AI Explanations)

```powershell
# Windows (PowerShell)
irm https://ollama.com/install.ps1 | iex

# Then pull a model
ollama run llama3
```

### Step 4 — Install frontend dependencies

```bash
cd frontend
npm install
```

### Step 5 — Run everything

You need **3 terminals** running simultaneously:

```bash
# Terminal 1: Flask API
cd hns
venv\Scripts\activate
python api.py
# → http://localhost:5000

# Terminal 2: React Frontend
cd hns/frontend
npm run dev
# → http://localhost:5173

# Terminal 3: Ollama (for AI explanations)
ollama run llama3
# → http://localhost:11434
```

Open **http://localhost:5173** in your browser.

---

## 🖥️ Web Application

### Design System

| Token | Value |
|---|---|
| Background | `#F0F4F8` (Alice Blue) |
| Accent | `#B2DFDB` (Light Teal) |
| Glass Surface | `rgba(255,255,255,0.20)` + `backdrop-filter: blur(15px)` |
| Font | Inter (Google Fonts) |
| Letter Spacing | `0.025em` |
| Animations | Framer Motion (floating cards, fade-in, staggered reveals) |

### Pages

| Page | Route | Description |
|---|---|---|
| **Landing** | `/` | Hero, How It Works, Features, Architecture, CTA |
| **Auth** | `/auth` | Sign In / Sign Up with Practitioner / Patient toggle |
| **Practitioner Dashboard** | `/practitioner` | Full analysis with severity, heatmaps, AI clinical report |
| **Patient Dashboard** | `/patient` | Simplified guidance with doctor consultation advice |

### User Flows

**Practitioner:**
1. Sign up → Select disease model + XAI method → Upload scan
2. Get severity assessment (Critical / Moderate / Low)
3. View Grad-CAM (region) + Integrated Gradients (pixel) heatmaps
4. Generate detailed AI clinical explanation via Ollama

**Patient:**
1. Sign up → Select disease model → Upload scan
2. Get verdict: "Consult a Doctor" or "Low Risk — Monitor"
3. Read patient-friendly AI explanation
4. Get downloadable brief to share with doctor

---

## 🔌 API Endpoints

The Flask API (`api.py`) serves the ML model to the React frontend:

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System health + available models |
| `/api/auth/signup` | POST | Create account (practitioner or patient) |
| `/api/auth/signin` | POST | Login with email + password |
| `/api/predict` | POST | Upload image → prediction + XAI heatmaps (base64) |
| `/api/explain` | POST | Generate AI explanation via Ollama (role-aware) |

---

## 🏗️ Architecture

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

## 🔍 Explainability Methods

| Method | Granularity | Speed | Description |
|---|---|---|---|
| **Grad-CAM** | Coarse ROI | ~1s | Gradient-weighted activation map from layer4 |
| **Integrated Gradients** | Pixel-level PLI | ~5-15s | Captum IG with NoiseTunnel (SmoothGrad²) |
| **Ollama LLM** | Textual | ~5-20s | AI-generated clinical explanation (role-aware) |

---

## 📊 Evaluation

```bash
python -m src.evaluate
```

Outputs: AUC-ROC, F1, Precision, Recall, Dice coefficient, IoU, confusion matrix, and curve plots saved to `models/eval_plots/`.

---

## 📓 Research Notebook

```bash
jupyter notebook Montgomery.ipynb
```

Covers: EDA, DataLoader construction, model architecture overview, training, training curves, test evaluation, Grad-CAM & IG visualisations, and the dual-granularity comparison figure.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch, SE-ResNet-18 |
| Explainability | Captum (IG), Custom Grad-CAM |
| Frontend | React 19, Vite 5, Framer Motion |
| Styling | Vanilla CSS (glassmorphism design system) |
| Backend API | Flask + Flask-CORS |
| AI Explanations | Ollama (llama3 / mistral) |
| Legacy Frontend | Streamlit (`app.py`) |

---

## 📈 Results

| Task | Dataset | Baseline (paper) | SE-ResNet-18 (ours) |
|---|---|---|---|
| Pneumonia detection | PneumoniaMNIST | 92.56% | After training |
| Breast malignancy | BreastMNIST | 88.46% | After training |

Run `python -m src.train` and check `models/test_metrics_*.json` for actual numbers.

---

## 🎯 Legacy Streamlit App

The original Streamlit interface is still available:

```bash
streamlit run app.py
# → http://localhost:8501
```

---

## 📄 Citation

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

---

<p align="center">
  <em>Built with ❤️ at NSUT, Delhi</em>
</p>
