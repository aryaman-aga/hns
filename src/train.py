"""
Training loop for SE-ResNet-18 on MedMNIST.

Improvements over the baseline paper:
  • Cosine Annealing LR schedule (smooth decay, no plateau hunting)
  • Mixup augmentation (alpha=0.2) for better decision boundary generalisation
  • Label-smoothed cross-entropy loss (smoothing=0.1)
  • Gradient clipping (max_norm=1.0) for training stability
  • Class-balanced WeightedRandomSampler (in dataset.py)
  • Early stopping (patience=10)

Usage:
  python -m src.train --task pneumonia --epochs 50 --batch_size 64
  python -m src.train --task breast    --epochs 50 --batch_size 32
"""

import argparse
import json
import sys
import ssl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import get_dataloaders, TASK_INFO
from src.model   import build_model, LabelSmoothingCELoss


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(args):
    device   = get_device()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.root) / 'data'

    print(f"Task   : {args.task}")
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        task=args.task,
        batch_size=args.batch_size,
        data_dir=str(data_dir),
        num_workers=args.num_workers,
    )
    task_info   = TASK_INFO[args.task]
    multi_label = task_info.get('multi_label', False)
    n_classes   = task_info['n_classes']

    print(f"Train  : {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")
    if multi_label:
        print(f"Mode   : multi-label  ({n_classes} classes)")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(n_classes, device, use_metadata=False)

    # ── Loss ──────────────────────────────────────────────────────────────
    if multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = LabelSmoothingCELoss(
            n_classes=n_classes,
            smoothing=0.1,
            weight=class_weights.to(device),
        )

    # ── Optimiser + Cosine Annealing ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_metric  = 0.0          # val_acc for single-label, mAUC for multi-label
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for imgs, labels in tqdm(
            train_loader,
            desc=f'Epoch {epoch:3d}/{args.epochs} [train]',
            leave=False,
        ):
            imgs, labels = imgs.to(device), labels.to(device)

            if args.mixup and not multi_label:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels)
                logits = model(imgs, None)
                loss   = mixup_loss(criterion, logits, y_a, y_b, lam)
                correct += (
                    lam * (logits.argmax(1) == y_a).float() +
                    (1 - lam) * (logits.argmax(1) == y_b).float()
                ).sum().item()
            else:
                logits = model(imgs, None)
                loss   = criterion(logits, labels)
                if multi_label:
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == labels).all(dim=1).sum().item()
                else:
                    correct += (logits.argmax(1) == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        scheduler.step()
        train_loss = total_loss / n
        train_acc  = correct / n

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_n = 0.0, 0, 0
        all_probs_val, all_labels_val = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs, None)
                v_loss += criterion(logits, labels).item() * imgs.size(0)

                if multi_label:
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    v_correct += (preds == labels).all(dim=1).sum().item()
                    all_probs_val.append(torch.sigmoid(logits).cpu())
                    all_labels_val.append(labels.cpu())
                else:
                    v_correct += (logits.argmax(1) == labels).sum().item()
                v_n += imgs.size(0)

        val_loss = v_loss / v_n
        val_acc  = v_correct / v_n

        # For multi-label: compute mAUC as the early-stopping metric
        if multi_label and len(all_probs_val) > 0:
            from sklearn.metrics import roc_auc_score
            import numpy as np
            p = torch.cat(all_probs_val).numpy()
            l = torch.cat(all_labels_val).numpy()
            try:
                aucs = [roc_auc_score(l[:, c], p[:, c])
                        for c in range(n_classes)
                        if l[:, c].sum() > 0]
                monitor_metric = float(np.mean(aucs))
            except Exception:
                monitor_metric = val_acc
        else:
            monitor_metric = val_acc

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(monitor_metric)

        metric_name = 'mAUC' if multi_label else 'acc'
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} {metric_name}={monitor_metric:.4f} | "
            f"lr={lr_now:.2e}"
        )

        if monitor_metric > best_metric:
            best_metric = monitor_metric
            torch.save(
                {'epoch': epoch, 'state_dict': model.state_dict(),
                 'val_metric': monitor_metric, 'args': vars(args)},
                save_dir / f'best_{args.task}.pth',
            )
            patience_cnt = 0
            print(f"  ✓ Saved best model ({metric_name}={monitor_metric:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    with open(save_dir / f'history_{args.task}.json', 'w') as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ──────────────────────────────────────────────
    print('\nLoading best checkpoint for test evaluation...')
    ckpt = torch.load(save_dir / f'best_{args.task}.pth', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs, None)
            if multi_label:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(logits, 1)
                preds = logits.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
    import numpy as np

    if multi_label:
        per_class_auc = {}
        valid_classes = []
        for i, cls in enumerate(task_info['classes']):
            if all_labels[:, i].sum() > 0:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                per_class_auc[cls] = auc
                valid_classes.append(cls)
        mauc = float(np.mean(list(per_class_auc.values())))
        subset_acc = float((all_preds == all_labels).all(axis=1).mean())
        print(f'\n{"="*60}')
        print(f'Task           : {args.task}')
        print(f'Subset Accuracy: {subset_acc*100:.2f}%')
        print(f'Mean AUC-ROC   : {mauc:.4f}')
        print('\nPer-class AUC-ROC:')
        for cls, auc in per_class_auc.items():
            print(f'  {cls:<22}: {auc:.4f}')
        metrics = {
            'subset_accuracy': subset_acc,
            'mean_auc': mauc,
            'per_class_auc': per_class_auc,
        }
    else:
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        f1  = f1_score(all_labels, all_preds)
        print(f'\n{"="*50}')
        print(f'Task     : {args.task.capitalize()}')
        print(f'Accuracy : {acc*100:.2f}%  (paper: {"92.56" if args.task=="pneumonia" else "88.46"}%)')
        print(f'AUC-ROC  : {auc:.4f}')
        print(f'F1       : {f1:.4f}')
        print(classification_report(
            all_labels, all_preds,
            target_names=task_info['classes']
        ))
        metrics = {'accuracy': acc, 'auc': auc, 'f1': f1,
                   'best_val_metric': best_metric}

    with open(save_dir / f'test_metrics_{args.task}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train SE-ResNet-18 on MedMNIST'
    )
    parser.add_argument('--task',         default='pneumonia',
                        choices=['pneumonia', 'breast', 'chest14'])
    parser.add_argument('--root',         default=str(ROOT))
    parser.add_argument('--save_dir',     default=str(ROOT / 'models'))
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience',     type=int,   default=10)
    parser.add_argument('--num_workers',  type=int,   default=0)
    parser.add_argument('--mixup',        action='store_true', default=True)
    args = parser.parse_args()
    train(args)
