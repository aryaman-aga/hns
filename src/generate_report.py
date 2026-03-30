"""
Generates a PDF technical report of the SE-ResNet-18 architecture.
Requires: reportlab  (pip install reportlab)

Usage:
    python -m src.generate_report
Output:
    architecture_report.pdf  (in project root)
"""

import json
from pathlib import Path
import sys
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Architecture diagram via matplotlib ──────────────────────────────────────

def _draw_architecture() -> bytes:
    """Returns a PNG bytes blob of the architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    fig.patch.set_facecolor('#f9f9f9')

    def box(x, y, w, h, label, color='#4A90D9', fontsize=9):
        r = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.03", linewidth=1.2,
            edgecolor='#2c2c2c', facecolor=color, zorder=3,
        )
        ax.add_patch(r)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    layers = [
        (0.5, 0.92, 0.30, 0.05, 'Input\n1×224×224 grayscale',     '#6c757d'),
        (0.5, 0.82, 0.30, 0.05, 'Conv1  7×7, s=2, 64 ch\n(weight-avg from ImageNet)', '#5c85d6'),
        (0.5, 0.72, 0.32, 0.05, 'Layer1 — BasicBlock×2 + SE\n64 ch  →  56×56',        '#4A90D9'),
        (0.5, 0.62, 0.32, 0.05, 'Layer2 — BasicBlock×2 + SE\n128 ch →  28×28',       '#3a7bc8'),
        (0.5, 0.52, 0.32, 0.05, 'Layer3 — BasicBlock×2 + SE\n256 ch →  14×14',       '#2a6ab7'),
        (0.5, 0.42, 0.32, 0.05, 'Layer4 — BasicBlock×2 + SE\n512 ch →   7×7   ← Grad-CAM', '#1a59a6'),
        (0.5, 0.32, 0.28, 0.05, 'AdaptiveAvgPool2d → (512)',       '#6c757d'),
        (0.5, 0.22, 0.28, 0.05, 'Dropout(0.3) → FC(128) → ReLU',   '#5c6e7a'),
        (0.5, 0.12, 0.28, 0.05, 'Dropout(0.15) → FC(n_classes)',   '#495057'),
        (0.5, 0.03, 0.22, 0.04, 'Softmax → Prediction',            '#28a745'),
    ]

    for x, y, w, h, label, color in layers:
        box(x, y, w, h, label, color)

    ys = [l[1] for l in layers]
    for i in range(len(ys) - 1):
        arrow(0.5, ys[i] - layers[i][3]/2 - 0.005,
              0.5, ys[i+1] + layers[i+1][3]/2 + 0.005)

    # SE detail annotation
    ax.annotate(
        'SE Block:\nGlobalAvgPool → FC(r=16) → ReLU\n→ FC → Sigmoid → Scale',
        xy=(0.5, 0.42), xytext=(0.82, 0.55),
        fontsize=8, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', fc='#fff9c4', ec='#999', lw=1),
        arrowprops=dict(arrowstyle='->', color='#888', lw=1),
    )

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('SE-ResNet-18 Architecture', fontsize=14, fontweight='bold', pad=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


# ── PDF generation ────────────────────────────────────────────────────────────

def generate(output_path: Path = ROOT / 'architecture_report.pdf'):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
            Table, TableStyle, HRFlowable,
        )
        from reportlab.lib import colors
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab")
        return

    doc    = SimpleDocTemplate(str(output_path), pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    h1 = ParagraphStyle('h1', parent=styles['Heading1'], fontSize=16, spaceAfter=6)
    h2 = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=13, spaceAfter=4)
    body = styles['BodyText']
    body.leading = 16

    # ── Title ──
    story.append(Paragraph(
        'Enhancing Clinical Trust: SE-ResNet-18 with Dual-Granularity Explainability', h1
    ))
    story.append(Paragraph('Aryaman Agarwal, Kanik Chawla, Sparsh Kalia', styles['Italic']))
    story.append(Paragraph('Netaji Subhas University of Technology', styles['Italic']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.3*cm))

    # ── Architecture diagram ──
    story.append(Paragraph('1. Model Architecture', h2))
    png_bytes = _draw_architecture()
    img_obj   = RLImage(io.BytesIO(png_bytes), width=14*cm, height=9*cm)
    story.append(img_obj)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        'The SE-ResNet-18 adapts the standard ResNet-18 for single-channel (grayscale) medical images. Key modifications: (1) the first convolutional layer is re-initialised with weights averaged over the 3 RGB channels of the ImageNet-pretrained model, preserving pretrained knowledge; (2) a Squeeze-and-Excitation (SE) block is appended after every residual block, providing channel-wise feature recalibration at each scale; (3) a two-layer dropout-regularised classifier head replaces the original FC layer.', body
    ))
    story.append(Spacer(1, 0.4*cm))

    # ── SE block explanation ──
    story.append(Paragraph('2. Squeeze-and-Excitation (SE) Block', h2))
    story.append(Paragraph(
        'The SE block (Hu et al., 2018) recalibrates channel responses using global context. Given a feature map X ∈ ℝ^(C×H×W):', body
    ))
    se_data = [
        ['Step', 'Operation', 'Output shape'],
        ['Squeeze',   'GlobalAvgPool2d(X)',                   '(C,)'],
        ['Excitation','FC(C→C/r) → ReLU → FC(C/r→C) → σ', '(C,)'],
        ['Scale',     'X ⊙ s.reshape(C,1,1)',                '(C,H,W)'],
    ]
    se_table = Table(se_data, colWidths=[3*cm, 8*cm, 3*cm])
    se_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.steelblue),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
    ]))
    story.append(se_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Training improvements ──
    story.append(Paragraph('3. Training Improvements Over Baseline', h2))
    improvements = [
        ['Improvement',          'Detail',                               'Benefit'],
        ['Cosine Annealing LR',  'T_max=50, η_min=1e-6',                'Smooth decay, avoids plateau'],
        ['Mixup Augmentation',   'α=0.2 (Beta distribution)',           'Better decision boundary'],
        ['Label Smoothing',      'ε=0.1',                               'Prevents overconfidence'],
        ['SE Blocks',            'Reduction ratio r=16',                 'Channel recalibration'],
        ['AdamW + WD',           'lr=3e-4, wd=1e-4',                    'Decoupled weight decay'],
        ['Class Weighting',      'WeightedRandomSampler',               'Handles class imbalance'],
        ['Gradient Clipping',    'max_norm=1.0',                        'Training stability'],
    ]
    t = Table(improvements, colWidths=[4*cm, 5*cm, 5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.steelblue),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ── Explainability ──
    story.append(Paragraph('4. Dual-Granularity Explainability', h2))
    xai_data = [
        ['Method',              'Granularity',    'Target',          'Speed'],
        ['Grad-CAM',            'Macro (ROI)',    'model.layer4',    '~0.5s'],
        ['Integrated Gradients','Micro (pixel)',  'input pixels',   '~10s'],
    ]
    t2 = Table(xai_data, colWidths=[4.5*cm, 3.5*cm, 4*cm, 2.5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkorange),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        'Grad-CAM computes gradient-weighted spatial activations from Layer4, producing coarse region maps (7×7 upsampled to 224×224). Integrated Gradients accumulates gradients along a linear path from a zero baseline to the input (50 steps, 10 NoiseTunnel samples with SmoothGrad²), producing high-resolution pixel-level attributions.', body
    ))
    story.append(Spacer(1, 0.4*cm))

    # ── Results ──
    story.append(Paragraph('5. Results vs Baseline', h2))
    results_data = [
        ['Dataset',        'Metric',    'Baseline (paper)', 'SE-ResNet-18 (ours)'],
        ['PneumoniaMNIST', 'Accuracy',  '92.56%',           '(see models/test_metrics_pneumonia.json)'],
        ['PneumoniaMNIST', 'AUC-ROC',   '—',                '(see models/test_metrics_pneumonia.json)'],
        ['BreastMNIST',    'Accuracy',  '88.46%',           '(see models/test_metrics_breast.json)'],
        ['BreastMNIST',    'AUC-ROC',   '—',                '(see models/test_metrics_breast.json)'],
    ]
    t3 = Table(results_data, colWidths=[3.5*cm, 2.5*cm, 3.5*cm, 5*cm])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
    ]))
    story.append(t3)

    doc.build(story)
    print(f'PDF written to: {output_path}')


if __name__ == '__main__':
    generate()
