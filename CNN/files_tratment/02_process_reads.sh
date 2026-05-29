#!/bin/bash
BASE_DIR="$HOME/Klebsiella_POD5"
DORADO="$BASE_DIR/dorado"
DATA_DIR="$BASE_DIR/pod5_data/KP1779"
FASTA_DIR="$DATA_DIR/fasta_files"

# Caminho absoluto para a pasta dos POD5 no teu Google Drive
# (Se a estirpe for a KP1055 em vez da KP1779, altera o nome da pasta abaixo)
DRIVE_DATA_DIR="/Users/leonardofonseca/Library/CloudStorage/GoogleDrive-leonardofonseca30leo@gmail.com/O meu disco/Klebsiella_POD5/pod5_data/KP1779/pod5_pass"

mkdir -p "$FASTA_DIR"

echo "🧬 Iniciando Basecalling (Dorado)..."
# Loop pelos POD5 diretamente da Drive
for pod5 in "$DRIVE_DATA_DIR"/*.pod5; do
    if [ -f "$pod5" ]; then
        name=$(basename "$pod5" .pod5)
        echo "🚀 Processando a partir do Drive: $name"
        
        # Basecalling -> FASTQ -> Conversão para FASTA (compatível com macOS)
        "$DORADO" basecaller hac "$pod5" --emit-fastq | \
        awk 'NR % 4 == 1 {sub(/^@/,">"); print} NR % 4 == 2 {print}' > "$FASTA_DIR/${name}.fasta"
    else
        echo "⚠️ Ficheiro não encontrado. Verifica se a pasta no Drive está correta."
    fi
done

echo "✅ Reads convertidas para FASTA em $FASTA_DIR"
