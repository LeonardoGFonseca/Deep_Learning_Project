import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))
from cnns_backbones import CNNStem
from classifiers import GapBaseline, MambaClassifier
from dataset import NanoSquiggleDataset

MODEL_REGISTRY = {
    'gap': GapBaseline,
    'mamba': MambaClassifier,
}

GENE_TO_CLASS = {
    "ENA|HEE1644226|HEE1644226.1": 1,  # APH(6)-I -> mapped to 1 in full 4-class
    "ENA|MH733892|MH733892.1": 2,      # blaSHV -> mapped to 2 in full 4-class
    "ENA|MZ092836|MZ092836.1": 3,      # oqxA -> mapped to 3 in full 4-class
}

def load_model(model_type, state_dict_path, num_classes, device):
    backbone = CNNStem(dropout=0.2)
    model = MODEL_REGISTRY[model_type](backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def run_sequential_inference(binary_model, multiclass_model, test_loader, device, threshold=0.5):
    all_signals = []
    all_binary_labels = []    # 0 = Noise, 1 = Gene
    all_binary_preds = []     # 0 = Noise, 1 = Gene
    all_true_classes = []     # 0 = Noise, 1 = APH, 2 = blaSHV, 3 = oqxA
    all_sequential_preds = [] # 0 = Noise, 1 = APH, 2 = blaSHV, 3 = oqxA

    # For tracing multiclass predictions on true genes
    multiclass_true_subset = []
    multiclass_pred_subset = []

    print("[*] Running sequential inference on test dataset...", flush=True)

    for signals, binary_labels, gene_targets in test_loader:
        signals = signals.to(device)
        
        # 1. Binary classification
        binary_logits = binary_model(signals).squeeze(1)
        binary_probs = torch.sigmoid(binary_logits)
        binary_preds = (binary_probs > threshold).long()

        # Iterate over batch items
        for i in range(signals.size(0)):
            b_label = int(binary_labels[i].item())
            gene_name = gene_targets[i]
            
            # Map true class to 4-class system
            if b_label == 0:
                true_class = 0  # Noise
            else:
                true_class = GENE_TO_CLASS.get(gene_name, 0)
            
            all_binary_labels.append(b_label)
            all_true_classes.append(true_class)

            b_pred = int(binary_preds[i].item())
            all_binary_preds.append(b_pred)
            if b_pred == 0:
                # Predicted as Noise
                all_sequential_preds.append(0)
            else:
                # Predicted as Gene -> feed to multiclass model
                single_signal = signals[i].unsqueeze(0)  # (1, 1, SeqLen)
                multi_logits = multiclass_model(single_signal)
                # Multiclass model classes: 0 = APH(6)-I, 1 = blaSHV, 2 = oqxA
                m_pred = int(torch.argmax(multi_logits, dim=1).item())
                # Map back to 4-class: 0 = Noise, 1 = APH, 2 = blaSHV, 3 = oqxA
                all_sequential_preds.append(m_pred + 1)
                
                # If it's a true gene, record for the multiclass-only subset metrics
                if true_class > 0:
                    multiclass_true_subset.append(true_class - 1)
                    multiclass_pred_subset.append(m_pred)

    return (np.array(all_binary_labels), np.array(all_binary_preds), 
            np.array(all_true_classes), np.array(all_sequential_preds),
            np.array(multiclass_true_subset), np.array(multiclass_pred_subset))

def plot_confusion_matrices(binary_cm, sequential_cm, save_dir, strain):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Binary Confusion Matrix
    plt.figure(figsize=(6, 5))
    disp_b = ConfusionMatrixDisplay(confusion_matrix=binary_cm, display_labels=['Noise', 'Gene'])
    disp_b.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Stage 1 Binary CM — {strain}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sequential_binary_cm_{strain}.png'))
    plt.close()

    # 2. Sequential 4-Class Confusion Matrix
    plt.figure(figsize=(8, 7))
    labels = ['Noise', 'APH(6)-I', 'blaSHV', 'oqxA']
    disp_s = ConfusionMatrixDisplay(confusion_matrix=sequential_cm, display_labels=labels)
    disp_s.plot(cmap=plt.cm.Greens, values_format='d')
    plt.title(f'Sequential 4-Class CM — {strain}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sequential_4class_cm_{strain}.png'))
    plt.close()

class SequentialDataset(NanoSquiggleDataset):
    """Custom dataset helper to return gene target names in getitem for sequential mapping."""
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        signal = torch.from_numpy(self.x[real_idx])
        label = torch.tensor(self.y[real_idx], dtype=torch.float32)
        gene_name = self.genes[real_idx]
        return signal, label, gene_name

def main():
    parser = argparse.ArgumentParser(description="Run Sequential Binary+Multiclass Inference")
    parser.add_argument('--h5_path', required=True, help="Path to amr_features.h5")
    parser.add_argument('--binary_model_type', required=True, choices=['gap', 'mamba'])
    parser.add_argument('--binary_model_path', required=True, help="Path to binary model weights (.pt)")
    parser.add_argument('--multiclass_model_type', required=True, choices=['gap', 'mamba'])
    parser.add_argument('--multiclass_model_path', required=True, help="Path to multiclass model weights (.pt)")
    parser.add_argument('--test_strain', required=True, help="Strain to run inference on (held out in LOSO)")
    parser.add_argument('--threshold', type=float, default=0.5, help="Binary threshold")
    parser.add_argument('--output_dir', default="sequential_output", help="Directory to save plots and report")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")
    print(f"[*] Evaluating strain: {args.test_strain}")

    # Load models
    binary_model = load_model(args.binary_model_type, args.binary_model_path, num_classes=1, device=device)
    multiclass_model = load_model(args.multiclass_model_type, args.multiclass_model_path, num_classes=3, device=device)
    print("[+] Loaded binary and multiclass models successfully.")

    # Load dataset for test strain
    test_ds = SequentialDataset(args.h5_path, include_only=args.test_strain, task='binary')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"[+] Loaded {len(test_ds)} test reads for strain {args.test_strain}")

    # Run sequential predictions
    b_labels, b_preds, true_4class, pred_4class, m_true, m_pred = run_sequential_inference(
        binary_model, multiclass_model, test_loader, device, threshold=args.threshold
    )

    # Compute metrics
    # Stage 1: Binary
    binary_acc = accuracy_score(b_labels, b_preds)
    binary_cm = confusion_matrix(b_labels, b_preds, labels=[0, 1])
    
    # Stage 2: Sequential 4-Class
    seq_acc = accuracy_score(true_4class, pred_4class)
    seq_cm = confusion_matrix(true_4class, pred_4class, labels=[0, 1, 2, 3])

    # Report outputs
    print("\n" + "="*50)
    print(f" SEQUENTIAL EVALUATION REPORT — {args.test_strain}")
    print("="*50)
    
    print(f"\n[1] STAGE 1: BINARY DETECTOR (Noise vs Gene)")
    print(f"    Accuracy: {binary_acc:.4f}")
    print("    Classification Report:")
    print(classification_report(b_labels, b_preds, target_names=['Noise', 'Gene'], zero_division=0))
    print("    Confusion Matrix:")
    print(binary_cm)

    print(f"\n[2] STAGE 2: MULTICLASS CLASSIFICATION (On predicted/true subsets)")
    if len(m_true) > 0:
        multi_acc = accuracy_score(m_true, m_pred)
        print(f"    Recall/Accuracy on true gene subset: {multi_acc:.4f}")
        print("    Classification Report (Subset):")
        print(classification_report(m_true, m_pred, target_names=['APH(6)-I', 'blaSHV', 'oqxA'], zero_division=0))
    else:
        print("    No true gene samples predicted/available in this fold subset.")

    print(f"\n[3] END-TO-END 4-CLASS PERFORMANCE")
    print(f"    Global Accuracy: {seq_acc:.4f}")
    print("    Classification Report:")
    print(classification_report(true_4class, pred_4class, target_names=['Noise', 'APH(6)-I', 'blaSHV', 'oqxA'], zero_division=0))
    print("    Confusion Matrix:")
    print(seq_cm)
    
    # Save plots
    plot_confusion_matrices(binary_cm, seq_cm, args.output_dir, args.test_strain)
    print(f"\n[+] Confusion matrices saved to: {args.output_dir}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
