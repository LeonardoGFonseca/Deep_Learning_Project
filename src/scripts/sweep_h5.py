import torch
import h5py
from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys
import argparse
import csv
from datetime import datetime

# Adicionar src ao path para conseguir importar os modelos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_2 import BinaryDetectorCNN, GeneClassifierCNN
from src.models.train_evaluate import train_model, evaluate_metrics

class NanoSquiggleDatasetH5(Dataset):
    """Dataset especializado para ler o ficheiro amr_features.h5 ultra-rápido."""
    def __init__(self, h5_path, task='binary'):
        self.h5_file = h5py.File(h5_path, 'r')
        self.task = task
        self.x = self.h5_file['X']
        self.y_bin = self.h5_file['Y']
        self.genes = self.h5_file['gene_target']
        
        # Mapeamento dinâmico para as 3 classes do teu segundo modelo
        self.gene_to_class = {
            "ENA|HEE1644226|HEE1644226.1": 0, 
            "ENA|MH733892|MH733892.1": 1,     
            "ENA|MZ092836|MZ092836.1": 2      
        }
        
        # Se for para o modelo multiclasse, vamos ignorar todo o ruído/background!
        if self.task == 'multiclass':
            self.valid_indices = []
            genes_array = self.genes[:]
            for i, g in enumerate(genes_array):
                gene_str = g.decode('utf-8')
                if gene_str in self.gene_to_class:
                    self.valid_indices.append(i)
        else:
            # Para o binário, usamos todos os dados (ruído e genes)
            self.valid_indices = list(range(self.x.shape[0]))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        # Carrega os pontos do sinal
        signal = torch.tensor(self.x[real_idx], dtype=torch.float32)
        
        if self.task == 'binary':
            label = torch.tensor(self.y_bin[real_idx], dtype=torch.long)
        elif self.task == 'multiclass':
            gene_name = self.genes[real_idx].decode('utf-8')
            class_idx = self.gene_to_class[gene_name]
            label = torch.tensor(class_idx, dtype=torch.long)
            
        return signal, label

def get_dataloaders(h5_path, task, batch_size=32):
    dataset = NanoSquiggleDatasetH5(h5_path, task=task)
    total = len(dataset)
    if total == 0:
        return None, None, None
        
    tr_size = int(0.80 * total)
    vl_size = int(0.15 * total)
    ts_size = total - tr_size - vl_size

    train_ds, val_ds, test_ds = random_split(dataset, [tr_size, vl_size, ts_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def save_results_csv(filename, record):
    """Grava os resultados num ficheiro CSV de forma incremental."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        # Colunas do CSV
        fieldnames = ['timestamp', 'phase', 'seq_len', 'batch_size', 'epochs', 'lr', 'kernels', 'accuracy', 'macro_f1', 'mean_auprc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(record)

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Sweep - Treinar Modelos em Cascata.')
    parser.add_argument('--h5_path', required=True, help='Caminho para o ficheiro amr_features.h5')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas.')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch.')
    parser.add_argument('--seq_len', type=int, default=30000, help='Tamanho da janela no modelo.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--kernels', nargs='+', type=int, default=[15, 7, 5, 5], help='Lista de tamanhos de kernel. Ex: 15 7 5 5')
    parser.add_argument('--csv_out', type=str, default='hyperparams_results.csv', help='Ficheiro CSV para gravar resultados.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Dispositivo: {device}")
    print(f"🔧 Configuração: SeqLen={args.seq_len}, Kernels={args.kernels}, Batch={args.batch_size}, LR={args.lr}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- FASE 1: Treino do Modelo Binário ---
    print("\n" + "="*50)
    print("🚀 FASE 1: DETETOR BINÁRIO")
    print("="*50)
    
    train_loader_bin, val_loader_bin, test_loader_bin = get_dataloaders(args.h5_path, task='binary', batch_size=args.batch_size)
    
    if train_loader_bin:
        binary_model = BinaryDetectorCNN(sequence_length=args.seq_len, kernel_sizes=args.kernels).to(device)
        train_model(binary_model, train_loader_bin, val_loader_bin, epochs=args.epochs, lr=args.lr, device=device)
        
        print("\n📊 Avaliação FASE 1 (Binário):")
        metrics_bin = evaluate_metrics(binary_model, test_loader_bin, device=device, class_names=["Ruído", "Gene"], show_plots=False)
        
        # Guardar resultados da Fase 1 no CSV
        record_bin = {
            'timestamp': timestamp,
            'phase': 'binary',
            'seq_len': args.seq_len,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'kernels': str(args.kernels),
            'accuracy': metrics_bin.get('accuracy', 0),
            'macro_f1': metrics_bin.get('macro_f1', 0),
            'mean_auprc': metrics_bin.get('mean_auprc', 0)
        }
        save_results_csv(args.csv_out, record_bin)
        print(f"✅ Resultados Fase 1 gravados em {args.csv_out}")

    # --- FASE 2: Treino do Modelo Multiclasse ---
    print("\n" + "="*50)
    print("🚀 FASE 2: CLASSIFICADOR DE GENES")
    print("="*50)
    
    train_loader_mul, val_loader_mul, test_loader_mul = get_dataloaders(args.h5_path, task='multiclass', batch_size=args.batch_size)
    
    if train_loader_mul:
        multi_model = GeneClassifierCNN(sequence_length=args.seq_len, kernel_sizes=args.kernels).to(device)
        train_model(multi_model, train_loader_mul, val_loader_mul, epochs=args.epochs, lr=args.lr, device=device)
        
        print("\n📊 Avaliação FASE 2 (Multiclasse):")
        metrics_mul = evaluate_metrics(multi_model, test_loader_mul, device=device, class_names=["Gene A", "Gene B", "Gene C"], show_plots=False)
        
        # Guardar resultados da Fase 2 no CSV
        record_mul = {
            'timestamp': timestamp,
            'phase': 'multiclass',
            'seq_len': args.seq_len,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'kernels': str(args.kernels),
            'accuracy': metrics_mul.get('accuracy', 0),
            'macro_f1': metrics_mul.get('macro_f1', 0),
            'mean_auprc': metrics_mul.get('mean_auprc', 0)
        }
        save_results_csv(args.csv_out, record_mul)
        print(f"✅ Resultados Fase 2 gravados em {args.csv_out}")

    print("\n🎉 Sweep Pipeline concluído para esta configuração!")

if __name__ == "__main__":
    main()
