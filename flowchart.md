# Fluxograma de Dados (Streaming Engine)

Este fluxograma ilustra o percurso lógico do script `master_server_pipeline.py`.
O foco principal é o ciclo repetitivo e isolado (streaming) para garantir uma pegada de memória e armazenamento muito reduzida no servidor.

```mermaid
graph TD
    A[Início: Ler data_ids.txt] --> B{Há mais estirpes?}
    
    B -- Sim --> C[Download tar.gz da Estirpe N]
    
    subgraph Gestão de RAM/Disco
        C --> D[Extrair para temp_extract/]
        D --> E[Filtrar e Mover Apenas pod5_pass/*.pod5]
        E --> F[Basecalling com Dorado]
        F --> G[Gerar ficheiro .bam]
        G --> H[Extração Tensorial com FeatureStoreAppender]
        H --> I[(amr_features_master.h5)]
    end
    
    I --> J[Purga: Apagar tar.gz, pod5, bam, bai]
    J --> B
    
    B -- Não --> K[Fase Final: Acionar sweep_h5.py]
    
    subgraph Treino e Análise (Sweep)
        K --> L[Carregar amr_features_master.h5]
        L --> M[Treinar CNN_1 Binária]
        M --> N[Guardar Pesos, Loss.png, MatrizConfusão.png]
        N --> O[Filtrar HDF5 para conter só Genes POSITIVOS]
        O --> P[Treinar CNN_2 Multiclasse]
        P --> Q[Guardar Pesos, Loss.png, MatrizConfusão.png]
        Q --> R[Fazer Append das Métricas em hyperparams_results.csv]
    end
    
    R --> S[Fim do Processo]
    
    style I fill:#f9f,stroke:#333,stroke-width:4px
    style J fill:#fbb,stroke:#333,stroke-width:2px
    style R fill:#bfb,stroke:#333,stroke-width:2px
```
