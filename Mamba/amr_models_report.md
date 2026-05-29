# Relatório Geral: Modelos de Deep Learning para Deteção e Classificação de Genes AMR

Este relatório compila os resultados de todos os modelos de Deep Learning desenvolvidos e testados desde o início do projeto para a deteção de genes de resistência antimicrobiana (AMR) diretamente a partir de dados de sinal bruto Oxford Nanopore (POD5).

---

## 1. Introdução e Desenho de Experiências

O objetivo principal deste projeto é processar sinais de corrente elétrica de sequenciação de *Klebsiella pneumoniae* para identificar:
1. **Deteção Binária**: Separar leituras de "Ruído/Background" (incluindo genes de manutenção MLST como *gapA*, *mdh*, *rpoB*) de leituras contendo "Genes de Resistência (AMR)".
2. **Classificação Multiclasse**: Identificar qual dos 3 genes AMR está presente:
   - `APH(6)-I` (Classe 0 - Resistência a Aminoglicósidos)
   - `blaSHV` (Classe 1 - Beta-lactamase)
   - `oqxA` (Classe 2 - Bomba de efluxo multidroga)

Os modelos foram avaliados usando a metodologia de **Validação Cruzada Leave-One-Strain-Out (LOSO)**, garantindo que o teste é sempre feito numa estirpe completamente invisível durante o treino, simulando o ambiente clínico real.

---

## 2. Fase 1: Deteção Binária (Ruído vs Gene)

Esta fase foca-se na triagem inicial das leituras. Comparamos os modelos convolucionais iniciais (CNNStem) com as variantes Mamba/GRU (que processam a sequência de forma recorrente/grammar engine):

### Tabela de Comparação: Deteção Binária

| Arquitetura / Experiência | Base de Dados | Exatidão (Accuracy) | F1-Score Macro | Sensibilidade (Recall Gene) | Resultado / Comportamento |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CNN Stage 1 (Clean)** | 20 Estirpes | 76.62% | 43.38% | 0.00% | **Colapso**. Predisse apenas a classe maioritária (Ruído). |
| **CNN Stage 2 (Smooth Wt)** | 20 Estirpes | 77.47% | 43.65% | 0.00% | **Colapso**. Predisse apenas a classe maioritária. |
| **CNN Stage 3 (Smooth + GAP)** | 20 Estirpes | 76.51% | **70.87%** | **71.00%** | **Aprendeu**. GAP + pesos resolveram o colapso. |
| **Mamba/GRU (Standard)** | 30 Estirpes | 96.43% | **92.46%** | **96.76%** | **Excelente**. A estrutura sequencial aumentou a performance. |
| **Mamba/GRU (Thresh-Tuned)** | 30 Estirpes | 97.21% | **93.61%** | **95.27%** | **Muito Bom**. Limiar ajustado equilibrou precisão/recall. |
| **Mamba/GRU (Weighted Loss)** | 30 Estirpes | **98.49%** | **96.73%** | **96.40%** | **Vencedor Clínico (Escolha Oficial)** |

> [!IMPORTANT]
> O modelo **Mamba/GRU com Perda Ponderada (`mamba_wt`)** é o detetor binário recomendado, atingindo **98.49% de exatidão média** nos primeiros folds, eliminando quase por completo os falsos negativos (sensibilidade de 96.4%).

---

## 3. Fase 2: Classificação Multiclasse (Qual o Gene?)

Esta fase foca-se na identificação precisa do tipo de gene, trabalhando apenas no subconjunto de leituras de genes reais (10,059 leituras). Comparamos o modelo convolucional inicial, o baseline GAP e o modelo Mamba/GRU (recorrente):

### Tabela de Comparação: Classificação Multiclasse

| Arquitetura / Experiência | Folds | Exatidão Média | F1-Score Macro | Precisão Média | Sensibilidade Média |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN Stage 3 (GAP + Wt)** | 20 Estirpes | 80.00% | 79.81% | 79.50% | 80.12% |
| **GAP Baseline Multiclass** | 10 Estirpes | 10 Folds | 94.66% | 94.56% | 94.54% |
| **Mamba/GRU Multiclass** | 10 Estirpes | **96.62%** *(parcial)* | **96.72%** *(parcial)* | **96.66%** *(parcial)* | **96.96%** *(parcial)* |

### Resultados Detalhados por Fold (GAP vs Mamba/GRU)

| Estirpe Ocultada (Fold) | Exatidão GAP Baseline | Exatidão Mamba/GRU | Melhoria Absoluta |
| :--- | :---: | :---: | :---: |
| **KP866** (Fold 1) | 90.07% | **98.58%** | **+8.51%** |
| **KP682** (Fold 2) | 91.59% | **95.15%** | **+3.56%** |
| **KP1743** (Fold 3) | 94.40% | **96.12%** | **+1.72%** |
| **KP1817** (Fold 4) | 95.35% | *Treinando...* | - |
| *Outros folds (5-10)* | *Concluído (média: 94.66%)* | *Agendados* | - |

> [!TIP]
> A preservação da ordem das leituras no modelo recorrente **Mamba/GRU** resulta numa exatidão significativamente superior à do GAP Baseline (+8.5% no fold 1, +3.5% no fold 2, +1.7% no fold 3), consolidando-o como a melhor escolha.

---

## 4. Pipeline Sequencial End-to-End (Binário $\rightarrow$ Multiclasse)

Para a utilização clínica prática, unimos os dois melhores modelos sequencialmente. O detetor binário (`mamba_wt`) filtra o ruído e o classificador multiclasse (`mamba_multiclass`) identifica o gene.

Abaixo estão as métricas end-to-end reais calculadas em estirpes de teste completamente ocultadas:

### Performance Geral de 4 Classes (Ruído vs APH vs SHV vs Oqx)

| Estirpe de Teste | Exatidão Binária | Exatidão Multiclasse (Subset) | Exatidão Global 4-Classes | F1-Score Macro |
| :--- | :---: | :---: | :---: | :---: |
| **KP866** (Fold 1) | 97.02% | 99.26% | **96.90%** | **93.22%** |
| **KP682** (Fold 2) | 93.46% | 96.64% | **92.65%** | **88.00%** |
| **KP1743** (Fold 3) | 96.66% | 97.29% | **96.01%** | **93.00%** |
| **Média Parcial** | **95.71%** | **97.73%** | **95.19%** | **91.41%** |

### Matriz de Confusão Agregada (Folds 1, 2 e 3 Combinados)
Total de Leituras Avaliadas: **2,957**

```
                     Previsão do Pipeline Sequencial
                     Ruído   APH(6)-I   blaSHV   oqxA   (Total Real)
Real   Ruído         2167       26        46      31       2270
       APH(6)-I        18      215         1       2        236
       blaSHV           2        1       190       1        194
       oqxA             7        4         4     233        248
```

> [!NOTE]
> O pipeline sequencial atinge uma exatidão combinada média de **95.19%**. O sistema demonstra uma sensibilidade notável para os genes de resistência, deixando escapar muito poucas leituras (apenas 27 de 678 leituras de genes reais foram falsamente classificadas como ruído).
