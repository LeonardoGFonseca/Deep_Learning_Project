# NanoSquiggle AMR CNN - Server Deployment

Este repositório foi criado especificamente para execução em servidores remotos com limites de armazenamento, focado em Grid Search e Hyperparameter Sweeping para modelos CNN aplicados a sequências genómicas em formato POD5.

## Como Executar no Servidor

1. Fazer clone do repositório:
   `git clone <LINK_DO_TEU_GITHUB>`
   `cd Final_Project`

2. Preparar as dependências (Linux Ubuntu/Debian):
   `sudo apt-get update && sudo apt-get install -y samtools`
   `pip install -r requirements.txt`

3. (IMPORTANTE) Instalar o Dorado:
   Cria a pasta `bin/dorado/bin/` e `bin/dorado/lib/` na raiz deste projeto, e coloca lá o executável do Dorado. (Podes também adaptar o script para descarregar o Dorado automaticamente).

4. Preencher os Dados de Entrada:
   Garante que tens o ficheiro `data_index/data_ids.txt` com as estirpes a analisar.
   Garante que o ficheiro de referência FASTA está em `data/ref/resistance_genes.fasta`.

5. Iniciar a Pipeline Automática:
   O comando abaixo fará tudo: downloads sequenciais, extração de pod5_pass, basecalling, construção do HDF5 otimizado, eliminação de dados brutos para poupar espaço e, por fim, treino das redes neuronais com as diferentes hiperparametrizações.
   
   `python master_server_pipeline.py`

## Para alterar os hiperparâmetros de teste:
Se quiseres testar outras combinações de janela (seq_len), kernels, etc, edita o final do ficheiro `master_server_pipeline.py` na chamada `subprocess.run(["python", str(sweep_script), ...])` e adiciona as tuas `--seq_len 15000` ou `--kernels 7 5 3 3`.

Os resultados serão guardados em:
- `hyperparams_results.csv` (As métricas todas)
- Pasta `graficos/` (Matrizes de Confusão e Curvas de Loss em PNG)
