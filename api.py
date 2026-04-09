"""
Flask REST API for the Dual-Granularity Medical Imaging system.
Wraps the existing PyTorch model + Ollama for the React frontend.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import json
import base64
import io
import os
import hashlib
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Setup paths ───────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
from src.dataset import TASK_INFO, IMG_SIZE
from src.model import build_model
from src.explainability import explain, preprocess_image, DEVICE

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

MODEL_DIR = ROOT / "models"
USERS_FILE = ROOT / "users.json"

TASK_LABELS = {
    "pneumonia": "Chest X-Ray — Pneumonia Detection",
    "breast":    "Breast Ultrasound — Malignancy Detection",
}

# ── Model cache ───────────────────────────────────────────────
_model_cache = {}

def get_model(task):
    if task not in _model_cache:
        ckpt_path = MODEL_DIR / f"best_{task}.pth"
        if not ckpt_path.exists():
            return None
        n_classes = TASK_INFO[task]["n_classes"]
        model = build_model(n_classes, DEVICE, use_metadata=False)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _model_cache[task] = model
    return _model_cache[task]


# ── Helpers ───────────────────────────────────────────────────
def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def numpy_to_base64(arr):
    """Convert numpy array to base64 PNG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ══════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["name", "email", "password", "role"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Missing field: {field}"}), 400

    users = load_users()

    # Check duplicate
    if any(u["email"] == data["email"] for u in users):
        return jsonify({"error": "An account with this email already exists"}), 409

    user = {
        "id": len(users) + 1,
        "name": data["name"],
        "email": data["email"],
        "password": hash_password(data["password"]),
        "role": data["role"],
        "specialization": data.get("specialization", ""),
        "licenseId": data.get("licenseId", ""),
        "created": datetime.now().isoformat(),
    }
    users.append(user)
    save_users(users)

    # Return user without password
    safe_user = {k: v for k, v in user.items() if k != "password"}
    return jsonify({"user": safe_user}), 201


@app.route("/api/auth/signin", methods=["POST"])
def signin():
    data = request.json
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password required"}), 400

    users = load_users()
    hashed = hash_password(data["password"])

    for user in users:
        if user["email"] == data["email"] and user["password"] == hashed:
            safe_user = {k: v for k, v in user.items() if k != "password"}
            return jsonify({"user": safe_user}), 200

    return jsonify({"error": "Invalid email or password"}), 401


# ══════════════════════════════════════════════════════════════
#  PREDICTION ENDPOINT
# ══════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    task = request.form.get("task", "pneumonia")
    method = request.form.get("method", "gradcam")

    model = get_model(task)
    if model is None:
        return jsonify({"error": f"No trained model found for task '{task}'"}), 404

    # Read image
    file = request.files["image"]
    pil_image = Image.open(file.stream)

    # Run inference + XAI
    heatmaps = {}

    if method == "both":
        result_gc = explain(pil_image, task, model, method="gradcam")
        result_ig = explain(pil_image, task, model, method="ig")
        result = result_gc  # Use gradcam result as base

        heatmaps["original"] = numpy_to_base64(result["orig_rgb"])
        heatmaps["gradcam"] = numpy_to_base64(result_gc["overlay"])
        heatmaps["ig"] = numpy_to_base64(result_ig["overlay"])
    else:
        result = explain(pil_image, task, model, method=method)
        heatmaps["original"] = numpy_to_base64(result["orig_rgb"])
        heatmaps[method] = numpy_to_base64(result["overlay"])

    response = {
        "label": result["label"],
        "confidence": float(result["confidence"]),
        "class_probs": result["class_probs"],
        "heatmaps": heatmaps,
        "multi_label": result.get("multi_label", False),
    }

    if result.get("active_diseases"):
        response["active_diseases"] = result["active_diseases"]

    return jsonify(response)


# ══════════════════════════════════════════════════════════════
#  AI EXPLANATION ENDPOINT (Ollama)
# ══════════════════════════════════════════════════════════════

@app.route("/api/explain", methods=["POST"])
def generate_explanation():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    task = data.get("task", "pneumonia")
    label = data.get("label", "Unknown")
    confidence = data.get("confidence", 0.0)
    class_probs = data.get("class_probs", {})
    role = data.get("role", "practitioner")
    ollama_model = data.get("model", "llama3")

    task_label = TASK_LABELS.get(task, task)

    if role == "practitioner":
        prompt = (
            f"You are an expert AI medical assistant providing clinical analysis support. "
            f"A {task_label} scan has been analyzed by our SE-ResNet-18 deep learning model with "
            f"Squeeze-Excitation attention blocks.\n\n"
            f"PREDICTION: '{label}' with {confidence*100:.1f}% confidence.\n"
            f"CLASS PROBABILITIES: {json.dumps(class_probs)}\n\n"
            f"The model uses dual-granularity explainability:\n"
            f"- Grad-CAM: Highlights the most relevant regions (coarse, region-level)\n"
            f"- Integrated Gradients: Traces prediction to individual pixels (fine, pixel-level)\n\n"
            f"Please provide a detailed clinical analysis:\n"
            f"1. What the prediction means and its clinical significance\n"
            f"2. What typical features the model likely identified in the scan "
            f"(e.g., opacities/consolidation for pneumonia, irregular masses for malignancies)\n"
            f"3. What the highlighted heatmap areas indicate about the problematic regions\n"
            f"4. Recommended next steps for clinical evaluation\n"
            f"5. Important caveat that this is AI-assisted and requires physician verification\n\n"
            f"Be thorough, professional, and clinically precise."
        )
    else:
        prompt = (
            f"You are a caring AI health assistant explaining medical scan results to a patient "
            f"in simple, non-technical language.\n\n"
            f"A {task_label} scan was analyzed by our AI system.\n"
            f"RESULT: '{label}' with {confidence*100:.1f}% confidence.\n\n"
            f"Please provide:\n"
            f"1. A simple, reassuring explanation of what this result means\n"
            f"2. Whether the patient should see a doctor based on these results\n"
            f"3. A brief, easy-to-understand summary they can show their doctor\n"
            f"4. General health tips related to this condition\n"
            f"5. A strong reminder that AI results are not a diagnosis and they should "
            f"always consult a healthcare professional\n\n"
            f"Use warm, supportive language. Avoid medical jargon. Keep it concise."
        )

    try:
        payload = json.dumps({
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=120) as response:
            res_body = response.read().decode("utf-8")
            ollama_res = json.loads(res_body)
            explanation = ollama_res.get("response", "No response received from AI.")

        return jsonify({"explanation": explanation})

    except Exception as e:
        return jsonify({
            "explanation": (
                f"⚠️ AI explanation unavailable. Please ensure Ollama is running:\n\n"
                f"1. Open a terminal\n"
                f"2. Run: ollama run llama3\n"
                f"3. Wait for it to download and start\n"
                f"4. Try again\n\n"
                f"Error: {str(e)}"
            )
        })


# ══════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "models_available": {
            task: (MODEL_DIR / f"best_{task}.pth").exists()
            for task in TASK_INFO
        },
    })


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  MedXAI Flask API")
    print(f"  Device: {DEVICE}")
    print(f"  Models directory: {MODEL_DIR}")
    for task in ["pneumonia", "breast"]:
        exists = (MODEL_DIR / f"best_{task}.pth").exists()
        print(f"  {task}: {'✓ loaded' if exists else '✗ not found'}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
