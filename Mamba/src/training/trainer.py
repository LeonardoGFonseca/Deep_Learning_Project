import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from dataset import make_loso_loaders

def load_strains(index_path="data/data_index/data_ids.txt"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Strain index not found at {index_path}. "
            "Run Phase 1 (download_data.py) first, or update the path."
        )
    with open(index_path) as f:
        strains = [line.strip() for line in f if line.strip()]
    if not strains:
        raise ValueError(f"Strain index at {index_path} is empty.")
    return strains

def train_epoch(model, loader, criterion, optimizer, device, task='binary'):
    model.train()
    total_loss = 0.0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(signals)
        if task == 'binary':
            logits = logits.squeeze(1)
            loss = criterion(logits, labels)
        else:
            loss = criterion(logits, labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, task='binary'):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for signals, labels in loader:
        signals = signals.to(device)
        logits = model(signals)
        
        if task == 'binary':
            logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            all_probs.extend(probs.cpu().numpy())
        else:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    if task == 'binary':
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'auprc': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist(),
        }
    else:
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'auprc': 0.0,
            'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist(),
        }


def run_loso_fold(model_fn, h5_path, test_strain, config, device, fold_dir):
    balanced = config.get('balanced', True)
    task = config.get('task', 'binary')
    
    train_loader, test_loader = make_loso_loaders(
        h5_path, test_strain, batch_size=config['batch_size'], balanced=balanced, task=task
    )

    val_size = int(0.15 * len(train_loader.dataset))
    train_size = len(train_loader.dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=config['batch_size'], shuffle=False
    )

    model = model_fn().to(device)
    
    if task == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    else:
        pos_weight = config.get('pos_weight', None)
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()
            
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'])

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(config['max_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, task=task)
        val_loss, _ = validate(model, val_loader, criterion, device, task=task)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"  Epoch {epoch+1}/{config['max_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}", end='\r', flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= config['early_stop_patience']:
                print(f"  [Early stopping at epoch {epoch+1}]")
                break

    print()

    model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device, threshold=config.get('threshold', 0.5), task=task)
    plot_loss_curve(train_losses, val_losses, os.path.join(fold_dir, 'loss_curve.png'))
    plot_confusion_matrix(test_metrics['confusion_matrix'], test_strain, os.path.join(fold_dir, 'confusion_matrix.png'), task=task)

    return test_metrics


@torch.no_grad()
def validate(model, loader, criterion, device, task='binary'):
    model.eval()
    total_loss = 0.0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        if task == 'binary':
            logits = logits.squeeze(1)
            loss = criterion(logits, labels)
        else:
            loss = criterion(logits, labels.long())
        total_loss += loss.item()
    return total_loss / len(loader), None


def run_loso(model_fn, h5_path, model_type, config, device, output_dir, strain_start=None, strain_end=None):
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    strains = load_strains()
    if strain_start is not None or strain_end is not None:
        start = strain_start if strain_start is not None else 0
        end = strain_end if strain_end is not None else len(strains)
        strains = strains[start:end]
        print(f"[SLICE] Processing strains from index {start} to {end}: {strains}", flush=True)

    all_metrics = []
    for strain in strains:
        fold_dir = os.path.join(results_dir, f'{model_type}_{strain}')
        os.makedirs(fold_dir, exist_ok=True)
        print(f"Fold: {strain}", flush=True)
        metrics = run_loso_fold(model_fn, h5_path, strain, config, device, fold_dir)
        metrics['strain'] = strain
        metrics['model_type'] = model_type
        all_metrics.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"AUPRC: {metrics['auprc']:.4f}", flush=True)

    csv_path = os.path.join(output_dir, 'ablation_results.csv')
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model_type', 'strain', 'accuracy', 'f1', 'precision', 'recall', 'auprc'])
        if f.tell() == 0:
            writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: v for k, v in m.items() if k != 'confusion_matrix'})

    mean_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in ['accuracy', 'f1', 'precision', 'recall', 'auprc']}
    print(f"\n{model_type} mean: {mean_metrics}")
    return all_metrics


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, strain, save_path, task='binary'):
    if task == 'binary':
        labels = ['Neg', 'Pos']
    else:
        labels = ['APH(6)-I', 'blaSHV', 'oqxA']
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix — {strain}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
