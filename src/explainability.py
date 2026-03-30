"""
Dual-Granularity Explainability for SE-ResNet-18
─────────────────────────────────────────────────
1. Grad-CAM (Macro / Region-level ROI)
   Hooks on model.layer4 — the last feature-map block (512-ch, 7×7).
   Returns a coloured heatmap overlay on the original image.

2. Integrated Gradients  (Micro / Pixel-level PLI)
   Uses Captum IntegratedGradients + NoiseTunnel (SmoothGrad²).
   Traces the prediction back to every input pixel.

Both return:
  overlay  — np.ndarray (H, W, 3) uint8  blended with heatmap
  heatmap  — np.ndarray (H, W)    float  normalised attribution [0,1]
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from captum.attr import IntegratedGradients, NoiseTunnel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model   import SEResNet18, build_model
from src.dataset import IMG_SIZE

DEVICE = torch.device(
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else 'cpu'
)


# ── Image helpers ────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Returns
    -------
    tensor   : (1, 1, 224, 224) float32
    orig_rgb : (224, 224, 3)    uint8 — for overlay rendering
    """
    gray    = pil_image.convert('L').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    orig_np = np.array(gray, dtype=np.uint8)               # (224, 224)

    img_f  = orig_np.astype(np.float32) / 255.0
    normed = (img_f - 0.5) / 0.5
    tensor = torch.from_numpy(normed).unsqueeze(0).unsqueeze(0).float()  # (1,1,224,224)

    # For display we show a 3-channel grayscale version
    orig_rgb = np.stack([orig_np, orig_np, orig_np], axis=-1)
    return tensor, orig_rgb


def _overlay(heatmap: np.ndarray, orig_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h8   = (heatmap * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig_rgb, 1.0 - alpha, cmap, alpha, 0).astype(np.uint8)


def _norm(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr - arr.min()
    return arr / (arr.max() + eps)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks on model.layer4 (output shape: B×512×7×7 for 224×224 input).
    """

    def __init__(self, model: SEResNet18):
        self.model      = model
        self._feats     = None
        self._grads     = None
        self._hook_f    = model.gradcam_layer.register_forward_hook(self._fwd)
        self._hook_b    = model.gradcam_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):  self._feats = out.detach()
    def _bwd(self, _, __, g):    self._grads = g[0].detach()

    def remove_hooks(self):
        self._hook_f.remove()
        self._hook_b.remove()

    def __call__(
        self,
        tensor:    torch.Tensor,
        class_idx: int,
        orig_rgb:  np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad()
        self.model.eval()
        tensor = tensor.to(DEVICE).requires_grad_(True)

        logits = self.model(tensor, None)
        logits[0, class_idx].backward()

        weights  = self._grads.mean(dim=(2, 3), keepdim=True)   # (1,512,1,1)
        cam      = (weights * self._feats).sum(dim=1).squeeze()  # (7,7)
        cam      = F.relu(cam)
        cam      = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode='bilinear', align_corners=False,
        ).squeeze().cpu().numpy()

        heatmap = _norm(cam)
        return heatmap, _overlay(heatmap, orig_rgb)


# ── Integrated Gradients ─────────────────────────────────────────────────────

class IntGrad:
    """
    Integrated Gradients + SmoothGrad² (NoiseTunnel).
    Provides pixel-level attribution without requiring internal hooks.
    """

    def __init__(self, model: SEResNet18):
        self.model = model
        self.ig    = IntegratedGradients(self._forward_fn)
        self.nt    = NoiseTunnel(self.ig)

    def _forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, None)

    def __call__(
        self,
        tensor:    torch.Tensor,
        class_idx: int,
        orig_rgb:  np.ndarray,
        n_steps:   int = 50,
        nt_samples: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        tensor    = tensor.to(DEVICE).requires_grad_(True)
        baseline  = torch.zeros_like(tensor)

        attribs = self.nt.attribute(
            tensor,
            nt_type='smoothgrad_sq',
            nt_samples=nt_samples,
            stdevs=0.1,
            target=class_idx,
            baselines=baseline,
            n_steps=n_steps,
        )
        attr = attribs.squeeze().cpu().numpy()      # (224, 224) — 1-channel
        heatmap = _norm(np.abs(attr))
        return heatmap, _overlay(heatmap, orig_rgb)


# ── Unified explain() ────────────────────────────────────────────────────────

def load_model(task: str, model_dir: Path) -> SEResNet18:
    task_info = TASK_INFO[task]
    model     = build_model(task_info['n_classes'], DEVICE, use_metadata=False)
    ckpt_path = Path(model_dir) / f'best_{task}.pth'
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def explain(
    pil_image:  Image.Image,
    task:       str,
    model:      SEResNet18,
    method:     str = 'gradcam',
) -> dict:
    """
    Parameters
    ----------
    pil_image  : PIL image (any mode — converted to grayscale internally)
    task       : 'pneumonia' | 'breast' | 'chest14'
    model      : loaded SEResNet18
    method     : 'gradcam' | 'ig'

    Returns
    -------
    dict with keys:
        label, confidence, class_probs, orig_rgb, heatmap, overlay,
        multi_label (bool), active_diseases (list[str]) for chest14
    """
    from src.dataset import TASK_INFO
    task_info   = TASK_INFO[task]
    multi_label = task_info.get('multi_label', False)
    class_names = task_info['classes']

    tensor, orig_rgb = preprocess_image(pil_image)
    tensor_dev = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor_dev, None)
        if multi_label:
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_class = int(probs.argmax())          # highest-prob class for XAI
        else:
            probs      = torch.softmax(logits, 1).squeeze().cpu().numpy()
            pred_class = int(probs.argmax())

    if method == 'gradcam':
        gc      = GradCAM(model)
        heatmap, overlay = gc(tensor, pred_class, orig_rgb)
        gc.remove_hooks()
    else:
        ig      = IntGrad(model)
        heatmap, overlay = ig(tensor, pred_class, orig_rgb)

    result = {
        'label':        class_names[pred_class],
        'confidence':   float(probs[pred_class]),
        'class_probs':  {name: float(p) for name, p in zip(class_names, probs)},
        'orig_rgb':     orig_rgb,
        'heatmap':      heatmap,
        'overlay':      overlay,
        'multi_label':  multi_label,
    }

    if multi_label:
        threshold = 0.5
        result['active_diseases'] = [
            name for name, p in zip(class_names, probs) if p >= threshold
        ]

    return result
