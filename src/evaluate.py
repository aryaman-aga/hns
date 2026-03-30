"""
Evaluation utilities: metrics + plots.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)


def evaluate_model(model, data_loader, device, task_info: dict) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            logits = model(imgs, None)
            probs  = torch.softmax(logits, 1)[:, 1]
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1  = f1_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)

    print(classification_report(
        all_labels, all_preds, target_names=task_info['classes']
    ))

    return {
        'accuracy': acc, 'auc': auc, 'f1': f1,
        'confusion_matrix': cm.tolist(),
        'preds': all_preds, 'labels': all_labels, 'probs': all_probs,
        'class_names': task_info['classes'],
    }


def plot_training_history(history: dict, save_path: Path, task: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{task.capitalize()} — Training History', fontsize=13)

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'],   label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend()

    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['val_acc'],   label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy'); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_eval_metrics(results: dict, save_dir: Path, task: str):
    save_dir.mkdir(parents=True, exist_ok=True)

    preds  = np.array(results['preds'])
    labels = np.array(results['labels'])
    probs  = np.array(results['probs'])
    names  = results['class_names']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{task.capitalize()} — Test Set Evaluation', fontsize=14)

    # Confusion matrix
    cm = np.array(results['confusion_matrix'])
    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks(range(len(names)));  axes[0].set_xticklabels(names, rotation=25)
    axes[0].set_yticks(range(len(names)));  axes[0].set_yticklabels(names)
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    axes[0].set_title(f'Confusion Matrix\nAcc={results["accuracy"]*100:.1f}%')
    for i in range(len(names)):
        for j in range(len(names)):
            axes[0].text(j, i, cm[i, j], ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black',
                         fontsize=12, fontweight='bold')

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    axes[1].plot(fpr, tpr, lw=2, color='steelblue',
                 label=f'AUC = {results["auc"]:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve'); axes[1].legend()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(labels, probs)
    axes[2].plot(rec, prec, lw=2, color='darkorange',
                 label=f'F1 = {results["f1"]:.4f}')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve'); axes[2].legend()

    plt.tight_layout()
    out = save_dir / f'eval_{task}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
