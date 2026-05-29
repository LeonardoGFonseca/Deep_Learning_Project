import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from cnns_backbones import CNNStem
from classifiers import GapBaseline, MambaClassifier, FlattenBaseline
from trainer import run_loso

MODEL_REGISTRY = {
    'gap': GapBaseline,
    'mamba': MambaClassifier,
    'flatten': FlattenBaseline,
}


def create_model_fn(model_type, dropout=0.2, num_classes=1):
    backbone = CNNStem(dropout=dropout)
    model_cls = MODEL_REGISTRY[model_type]

    def model_fn():
        return model_cls(backbone, num_classes=num_classes)

    return model_fn


def main():
    parser = argparse.ArgumentParser(description='NanoSquiggle-AMR Training Sweep')
    parser.add_argument('--h5_path', required=True, help='Path to amr_features.h5')
    parser.add_argument('--model_type', required=True, choices=['gap', 'mamba', 'flatten'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.2, help='CNNStem dropout')
    parser.add_argument('--output_dir', type=str, default='training_output',
                        help='Output directory for results')
    parser.add_argument('--legacy', action='store_true',
                        help='Acknowledge flatten model is legacy (non-primary ablation)')
    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Custom positive class weight for loss')
    parser.add_argument('--no_balanced', action='store_true',
                        help='Disable balanced undersampling (train on full dataset)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification decision threshold (default 0.5)')
    parser.add_argument('--strain_start', type=int, default=None,
                        help='Start index of strains to process')
    parser.add_argument('--strain_end', type=int, default=None,
                        help='End index of strains to process')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multiclass'],
                        help='Task type: binary or multiclass')
    args = parser.parse_args()

    if args.model_type == 'flatten' and not args.legacy:
        print("Warning: Flatten model has 17x more parameters than Mamba.")
        print("Results are not comparable. Use --legacy to acknowledge and run.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {args.model_type}, Task: {args.task}, Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    config = {
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'early_stop_patience': args.patience,
        'pos_weight': args.pos_weight,
        'balanced': not args.no_balanced,
        'threshold': args.threshold,
        'task': args.task,
    }

    num_classes = 3 if args.task == 'multiclass' else 1
    model_fn = create_model_fn(args.model_type, dropout=args.dropout, num_classes=num_classes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_loso(
        model_fn, args.h5_path, args.model_type, config, device, str(output_dir),
        strain_start=args.strain_start, strain_end=args.strain_end
    )


if __name__ == '__main__':
    main()
