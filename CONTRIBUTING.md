# Contributing Guide

Thank you for your interest in contributing!

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/hns.git
   cd hns
   ```

2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feat/your-feature-name
   ```

3. **Set up the environment**:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

## Project Layout

```
src/
  dataset.py       — MedMNIST v2 loading & augmentation
  model.py         — SE-ResNet-18 architecture
  train.py         — Training loop
  evaluate.py      — Metrics & plots
  explainability.py— Grad-CAM + Integrated Gradients
  generate_report.py — PDF architecture report
app.py             — Streamlit frontend
models/            — Trained weights (not committed, see README)
```

## Making Changes

- **Model changes** → edit `src/model.py` + re-run `src/train.py`
- **New dataset support** → add an entry to `TASKS` / `TASK_INFO` in `src/dataset.py`
- **Frontend changes** → edit `app.py`
- **New XAI method** → add a class in `src/explainability.py` and wire it into `explain()`

## Code Style

- Follow PEP 8 (max 100 chars per line)
- Type-annotate all public functions
- Keep module-level docstrings up to date

## Pull Request Checklist

- [ ] Code runs without errors (`python -m src.train --task pneumonia --epochs 1`)
- [ ] New features have been tested
- [ ] `requirements.txt` updated if new packages added
- [ ] PR description explains *what* and *why*

## Reporting Issues

Open a GitHub Issue with:
- Steps to reproduce
- Expected vs actual behaviour
- Python version + OS
- Relevant error traceback
