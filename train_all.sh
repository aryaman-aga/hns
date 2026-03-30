#!/bin/bash
# train_all.sh — trains all three tasks sequentially then launches the app
# Usage: bash train_all.sh

set -e
cd "$(dirname "$0")"
PYTHON="./venv/bin/python"

echo "=========================================="
echo " Training all models — SE-ResNet-18"
echo "=========================================="

echo ""
echo "[1/3] PneumoniaMNIST (binary — ~10 min)"
$PYTHON -m src.train --task pneumonia --epochs 50 --batch_size 64 \
  2>&1 | tee models/train_pneumonia.log
echo "✓ Pneumonia done"

echo ""
echo "[2/3] BreastMNIST (binary — ~3 min)"
$PYTHON -m src.train --task breast --epochs 50 --batch_size 32 \
  2>&1 | tee models/train_breast.log
echo "✓ Breast done"

echo ""
echo "[3/3] ChestMNIST 14-disease (multi-label — ~20 min)"
$PYTHON -m src.train --task chest14 --epochs 30 --batch_size 64 \
  2>&1 | tee models/train_chest14.log
echo "✓ Chest14 done"

echo ""
echo "=========================================="
echo " All models trained! Launching Streamlit…"
echo "=========================================="
./venv/bin/streamlit run app.py
