import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay, 
                             average_precision_score)

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """
    Treina o modelo multiclasse (4 classes) e regista a evolução da Loss.
    """
    model.to(device)
    
    # CrossEntropyLoss é a standard para Multiclasse. 
    # No PyTorch, ela JÁ APLICA o Softmax internamente aos logits brutos!
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- MODO DE TREINO ---
        model.train()
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Outputs será um tensor de tamanho [Batch_size, 4]
            outputs = model(inputs) 
            
            # Labels têm de ser do tipo inteiro/long (0, 1, 2 ou 3)
            loss = criterion(outputs, labels.long())
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- MODO DE VALIDAÇÃO ---
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

def plot_loss_curve(train_losses, val_losses, save_path=None):
    """
    Gera um gráfico com a avaliação da Loss ao longo do treino.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Evolução da Loss ao longo do Treino')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def evaluate_metrics(model, test_loader, device='cpu', class_names=None, show_plots=True, save_prefix=None):
    """
    Calcula métricas robustas para datasets desbalanceados:
    - Accuracy e Macro F1-Score
    - Classification Report (Precision, Recall e F1 por classe)
    - AUPRC (Area Under the Precision-Recall Curve)
    - Confusion Matrix Plot
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    num_classes = outputs.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\\n" + "=" * 50)
    print("📊 AVALIAÇÃO FINAL NO TEST SET")
    print("=" * 50)
    print(f"Accuracy Global: {acc * 100:.2f}%")
    print(f"Macro F1-Score:  {macro_f1 * 100:.2f}%\\n")
    
    print("📋 Relatório por Classe (Classification Report):")
    # classification_report dá-nos o precision, recall e f1-score per-class
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Calcular AUPRC (Average Precision)
    # Se for binário ou multiclasse, o sklearn lida de forma diferente
    mean_auprc = 0.0
    try:
        if num_classes == 2:
            # Para binário, all_probs[:, 1] são as probabilidades da classe positiva
            auprc = average_precision_score(all_labels, all_probs[:, 1])
            mean_auprc = auprc
            print(f"📈 AUPRC (Area Under PR Curve): {auprc:.4f}")
        else:
            # Para multiclasse (one-vs-rest), usando os arrays todos
            # O y_true precisa de estar em formato one-hot para a função funcionar
            # Uma forma simples de contornar é calcular per-class
            auprc_list = []
            for i in range(num_classes):
                # Criar labels binárias para esta classe (1 se for a classe, 0 se não)
                y_true_binary = (all_labels == i).astype(int)
                y_scores = all_probs[:, i]
                # Se a classe não existir no y_true, salta para evitar erro
                if sum(y_true_binary) > 0:
                    ap = average_precision_score(y_true_binary, y_scores)
                    auprc_list.append(ap)
                    print(f"📈 AUPRC ({class_names[i]}): {ap:.4f}")
            
            if auprc_list:
                mean_auprc = float(np.mean(auprc_list))
                print(f"📈 Mean AUPRC: {mean_auprc:.4f}")
    except Exception as e:
        print(f"⚠️ Não foi possível calcular o AUPRC: {e}")

    # Plot Confusion Matrix
    if show_plots or save_prefix:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_cm.png")
        if show_plots:
            plt.show()
        plt.close()
    
    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'mean_auprc': mean_auprc
    }