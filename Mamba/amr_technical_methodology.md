# Guia Técnico e Metodologia de Replicação: Classificação de Genes AMR com Sinais ONT

Este documento serve como manual técnico de referência para a replicação e interpretação de todas as experiências realizadas no projeto de deteção e classificação de genes de resistência antimicrobiana (AMR) usando sinal elétrico bruto Oxford Nanopore.

---

## 0. Introdução Intuitiva: Como funciona este sistema em linguagem simples?

Para leitores que não sejam especialistas em Inteligência Artificial ou processamento digital de sinal, a física e a matemática subjacentes a este projeto podem parecer abstratas. Abaixo, explicamos os conceitos fundamentais através de analogias simples do dia a dia.

### A. O que é o "Sinal Bruto" (POD5)?
Imagine que ler o ADN através de um nanoporo é como puxar uma corda com nós por um tubo estreito através do qual passa água. Cada tipo de nó (a sequência de letras do ADN) bloqueia a água de uma forma diferente, fazendo com que a pressão da água mude constantemente. 
Na sequenciação Nanopore, medimos a corrente elétrica em vez da pressão. As variações da corrente ao longo do tempo são o **sinal elétrico bruto (gravado em ficheiros .pod5)**. No gráfico, este sinal parece uma linha cheia de altos e baixos (uma "linha rabiscada").

### B. O papel da CNN: O "Tradutor de Sílabas"
A nossa linha rabiscada é gigantesca: tem **30 000 pontos** de informação por leitura. Tentar ler esta linha ponto por ponto seria como tentar ler um livro analisando a forma de cada partícula individual de tinta na página. É demasiada informação irrelevante.
A **CNN (Rede Neuronal Convolucional)** funciona como um tradutor local. Ela agrupa os pontos de tinta e traduz-nos em "sílabas" ou "motivos de sinal". Em vez de analisar 30 000 pontos individuais de tinta, a CNN resume o sinal numa sequência de **1 875 sílabas organizadas**.

### C. O Modelo GAP: O "Saco de Palavras"
Imagine que damos a sequência de 1 875 sílabas extraídas pela CNN a um leitor preguiçoso. Este leitor deita todas as sílabas para dentro de um saco, agita-o e conta apenas quantas vezes cada sílaba aparece, ignorando completamente a ordem em que estavam escritas. 
Se ele encontrar as sílabas "re", "sis", "tên", "cia", ele adivinha que a palavra é "resistência". Este é o **GAP Baseline (Global Average Pooling)**. Ele assume que a presença de certas características (motivos de sinal) é suficiente para saber que gene lá está, sem precisar de saber a sua ordem.

### D. O Modelo Mamba/GRU: O "Leitor Gramatical"
O modelo **Mamba/GRU** é um leitor inteligente. Ele lê a sequência de 1 875 sílabas da esquerda para a direita, **respeitando a sua ordem e gramática**. Ele percebe a diferença entre "resistência" e "tên-cia-re-sis". Como os genes de resistência têm assinaturas de sinal cuja ordem e sequência são críticas, este leitor gramatical consegue atingir uma precisão muito superior à do leitor do "saco de palavras".

### E. O Pipeline Sequencial: A Triagem Médica
Na prática clínica, não queremos desperdiçar recursos. Por isso, concebemos o sistema em duas etapas, tal como as urgências de um hospital:
1. **A Triagem (Detetor Binário)**: Um teste rápido para avaliar se a leitura é apenas "ruído de fundo" ou se tem, de facto, um gene de resistência. Se for apenas ruído, a leitura é descartada (o paciente tem alta).
2. **O Médico Especialista (Classificador Multiclasse)**: Se a triagem detectar um gene, a leitura é encaminhada para o especialista, que fará testes detalhados para prever exatamente qual dos 3 genes específicos de resistência está lá presente (`APH(6)-I`, `blaSHV` ou `oqxA`).

---

## 1. Visão Geral da Arquitetura Híbrida (CNN + GRU/Mamba)

Uma das maiores armadilhas conceituais é assumir que o modelo recorrente (GRU/Mamba) opera sozinho. Na realidade, **todos os modelos avaliados neste projeto são híbridos e utilizam uma CNN no front-end**.

### O Papel da CNN (CNNStem)
O sinal bruto extraído dos poros tem uma resolução temporal muito densa (30 000 pontos por janela). Processar 30 000 pontos diretamente numa camada recorrente (RNN/GRU) ou num bloco Mamba é computacionalmente inviável devido a problemas de:
1. **Desvanecimento de Gradiente (Vanishing Gradient)** sobre sequências ultra-longas.
2. **Consumo de Memória Exponencial** e lentidão extrema de treino.

A **CNNStem** atua como um **extrator de características locais e compressor temporal**. Ao correr 4 camadas convolucionais 1D com stride 2, a CNN Stem:
- Captura padrões espaciais locais (motivos de sinal elétrico correspondentes a K-mers de nucleótidos).
- Reduz o comprimento temporal em exatamente **$16\times$** ($30\,000 \rightarrow 15\,000 \rightarrow 7\,500 \rightarrow 3\,750 \rightarrow 1\,875$).
- Projeta o canal unidimensional de sinal bruto para um espaço vetorial latente de **256 canais** de características.

### A Diferença de Processamento Final

Após a extração pela CNNStem, os dados têm a dimensão de $(B, 256, 1875)$, onde $B$ é o tamanho do lote. A divergência reside no processamento desta dimensão:

```
                      [Sinal Elétrico Bruto (30 000)]
                                     │
                                     ▼
                        [CNNStem (4x Conv1D layers)]
                                     │
                        (Compressão Temporal 16x)
                                     │
                                     ▼
                         [Features (256 x 1875)]
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
     [GAP Baseline]                                      [Mamba/GRU]
            │                                                 │
  (AdaptiveAvgPool1d)                                   (Permuta de Eixos)
            │                                                 │
(Reduz dimensão para 256)                         [GRU / RNN Recorrente (1875)]
    (Ignora a ordem)                              (Processa a ordem do sinal)
            │                                                 │
            │                                       (AdaptiveAvgPool1d)
            │                                                 │
            └────────────────────────┬────────────────────────┘
                                     ▼
                             [Linear Classification Head]
```

---

## 2. Estrutura de Diretórios do Projeto

Para garantir a replicação exata, os dados, referências e scripts devem ser organizados na seguinte árvore de diretórios:

```
Mamba_Multiclasse/
├── data/
│   ├── data_index/
│   │   └── data_ids.txt            # IDs das estirrep para o LOSO (ex: KP866)
│   ├── ref/
│   │   └── resistance_genes.fasta  # Genoma de referência dos genes
│   └── processed/
│       ├── alignments/             # Ficheiros BAM gerados pelo Dorado
│       └── hd5f/
│           └── amr_features.h5     # Feature store compactada final (5.9 GB)
├── pipeline_logs/
│   ├── gap_multiclass/             # Resultados do sweep GAP
│   ├── mamba_multiclass/           # Resultados do sweep GRU/Mamba
│   └── sequential_inference/       # Resultados do pipeline sequencial
├── src/
│   ├── scripts/
│   │   ├── download_data.py        # Fase 1: download de POD5s
│   │   ├── run_dorado.py           # Fase 2: alinhador dorado
│   │   └── compile_features_store.py # Fase 3: compilação H5
│   └── training/
│       ├── cnns_backbones.py       # Definição do CNNStem
│       ├── classifiers.py          # Definição do GAP e Mamba/GRU
│       ├── dataset.py              # Loader customizado e undersampling
│       ├── trainer.py              # Loop de treino LOSO
│       ├── sweep.py                # Launcher CLI para sweeps
│       └── sequential_inference.py # Script de inferência 2-stage
└── run_multiclass_10strains.sh     # Script Slurm launcher
```

---

## 3. Pipeline de Execução Passo a Passo (Replicação)

### Passo 1: Download e Verificação de POD5s
Descarrega os sinais brutos de cada estirpe a partir de repositórios públicos.
```bash
python src/scripts/download_data.py \
  --index-file data/data_index/data_ids.txt \
  --raw-dir data/raw/
```

### Passo 2: Dorado Basecalling e Alinhamento Direcionado
Corre o alinhador Dorado sobre as leituras elétricas contra os genes de interesse e ordena o BAM de saída.
```bash
# Executa Dorado
./bin/dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 data/raw/ \
  --reference data/ref/resistance_genes.fasta > data/processed/alignments/unsorted.bam

# Ordena e indexa alinhamentos
samtools sort data/processed/alignments/unsorted.bam -o data/processed/alignments/sorted.bam
samtools index data/processed/alignments/sorted.bam
```

### Passo 3: Compilação da Feature Store HDF5
Compila as janelas de sinal em arrays numéricos estruturados no H5.
```bash
python src/scripts/compile_features_store.py \
  --window-size 30000 \
  --coverage 0.95 \
  --max-background 500 \
  --output-h5 data/processed/hd5f/amr_features.h5
```

### Passo 4: Execução do Sweep de Validação Cruzada (10 Folds)
Executa o treino Leave-One-Strain-Out em background de GPU.
```bash
# Executa GAP Baseline Multiclass
python src/training/sweep.py \
  --model_type gap \
  --task multiclass \
  --h5_path data/processed/hd5f/amr_features.h5 \
  --epochs 50 \
  --patience 10 \
  --strain_start 0 \
  --strain_end 10 \
  --output_dir pipeline_logs/gap_multiclass

# Executa Mamba/GRU Multiclass
python src/training/sweep.py \
  --model_type mamba \
  --task multiclass \
  --h5_path data/processed/hd5f/amr_features.h5 \
  --epochs 50 \
  --patience 10 \
  --strain_start 0 \
  --strain_end 10 \
  --output_dir pipeline_logs/mamba_multiclass
```

### Passo 5: Execução do Pipeline Sequencial End-to-End
Avalia o sistema de 4 classes combinando o detetor binário e o classificador multiclasse.
```bash
python src/training/sequential_inference.py \
  --h5_path data/processed/hd5f/amr_features.h5 \
  --binary_model_type mamba \
  --binary_model_path pipeline_logs/mamba_sweep/results/mamba_KP866/best_model.pt \
  --multiclass_model_type mamba \
  --multiclass_model_path pipeline_logs/mamba_multiclass/results/mamba_KP866/best_model.pt \
  --test_strain KP866 \
  --output_dir pipeline_logs/sequential_inference
```

---

## 4. Arquitetura Detalhada dos Modelos (Camada a Camada)

Abaixo encontra-se o código PyTorch exato das arquiteturas implementadas:

### 1. CNN Feature Extractor (CNNStem)
```python
import torch
import torch.nn as nn

class CNNStem(nn.Module):
    def __init__(self, in_channels=1, d_model=256, dropout=0.2):
        super().__init__()
        kernel_sizes = [15, 7, 5, 5]
        strides = [2, 2, 2, 2]
        channels = [32, 64, 128, d_model]

        layers = []
        current_in = in_channels
        for k, s, c in zip(kernel_sizes, strides, channels):
            layers.extend([
                nn.Conv1d(current_in, c, kernel_size=k, stride=s, padding=k//2, bias=False),
                nn.BatchNorm1d(c),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_in = c
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, SeqLen) -> (B, 1, SeqLen)
        return self.feature_extractor(x)
```

### 2. Cabeça do GAP Baseline
```python
class GapBaseline(nn.Module):
    def __init__(self, backbone: CNNStem, num_classes=3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),      # Reduz (B, 256, 1875) para (B, 256, 1)
            nn.Flatten(),                 # Transforma em (B, 256)
            nn.Linear(256, num_classes),  # Projeta para logits das classes
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
```

### 3. Cabeça do Mamba Classifier (GRU Fallback)
```python
class MambaClassifier(nn.Module):
    def __init__(self, backbone: CNNStem, num_classes=3):
        super().__init__()
        self.backbone = backbone
        # GRU substitui Mamba na ausência de suporte nativo cuda-compilation no cluster
        self.mamba = nn.GRU(input_size=256, hidden_size=256, batch_first=True)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)            # (B, 256, 1875)
        features = features.permute(0, 2, 1)   # (B, 1875, 256) - Formato RNN
        rnn_out, _ = self.mamba(features)      # (B, 1875, 256) - Processamento temporal
        features = rnn_out.permute(0, 2, 1)    # (B, 256, 1875) - Formato CNN original
        return self.head(features)
```

---

## 5. Hiperparâmetros de Treino e Otimizações de Estabilidade

Para garantir estabilidade numérica e evitar sobreajuste, o pipeline de treino em `trainer.py` incorpora:
1. **Early Stopping**: Pára o treino se a perda de validação (`Val Loss`) não diminuir durante 10 épocas consecutivas. Restaura os pesos do melhor checkpoint de época (`best_model.pt`).
2. **Cossenoidal de Ajuste (Cosine Annealing LR)**: Decai a taxa de aprendizagem dinamicamente de acordo com a função cossenoidal ao longo das épocas:
   $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$
3. **Ponderação de Frequência de Classes**: Para combater o viés da classe maioritária no modelo binário, calcula-se o peso dinâmico de perda como a raiz quadrada da razão inversa de frequências:
   $$\text{pos\_weight} = \sqrt{\frac{\text{Amostras Negativas}}{\text{Amostras Positivas}}}$$

---

## 6. Guia de Interpretação de Resultados

### O Fenómeno do Colapso de Classe (Fase Binária Inicial)
* **Sintoma**: O modelo atinge ~76.6% de exatidão global, mas a sensibilidade (Recall) da classe Gene é exatamente **0%** e o F1-Score Macro é de **43.38%** (o F1-Score do Ruído é ~86% e do Gene é 0%).
* **Explicação**: Como o ruído constitui a maioria dos dados alinhados, a rede descobre que pode minimizar a perda a curto prazo chutando que "tudo é ruído". A rede colapsou para a classe maioritária.
* **Resolução**: A introdução de pesos sub-lineares (`pos_weight`) e a redução maciça dos parâmetros da camada densa no final através do Global Average Pooling (GAP) forçaram a rede a extrair padrões generalizáveis, subindo o F1-Score Macro para mais de **70%** (CNN Stage 3) e posteriormente **96.73%** (Mamba/GRU com Perda Ponderada).

### Leitura da Matriz de Confusão do Pipeline Sequencial (4 Classes)
Na avaliação prática final, a classificação é apresentada numa matriz de confusão de $4 \times 4$ classes:
- A diagonal principal representa os acertos absolutos de cada classe.
- A **Coluna 1 (Noise)** indica os falsos negativos (leituras de genes AMR reais que foram erroneamente descartadas como ruído pelo detetor binário).
- A **Linha 1 (Noise)** indica os falsos positivos (leituras de ruído que passaram no detetor binário e foram rotuladas falsamente como genes AMR pelo classificador).
- A sub-matriz inferior direita de $3 \times 3$ indica a qualidade do diagnóstico específico de identificação dos genes.
- **Importância Clínica**: Minimizar a Coluna 1 é a prioridade crítica, pois em ambiente hospitalar é inaceitável classificar um gene de resistência real como ruído (Falso Negativo), o que levaria a uma terapêutica inadequada. A nossa sensibilidade de 96.4% garante a segurança necessária.
