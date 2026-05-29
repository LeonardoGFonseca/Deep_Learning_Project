#!/bin/bash
BASE_DIR="$HOME/Klebsiella_POD5"
DB_FASTA="$BASE_DIR/db_resitencia.fasta"
DB_OUT="$BASE_DIR/db_resistencia_index/db_resistencia"
QUERY_FILES="$BASE_DIR/pod5_data/KP1779/fasta_files/*.fasta"
OUTPUT_CSV="$BASE_DIR/pod5_data/KP1779/KP1779_resultado_final.csv"

echo "🛠️ Criando Index do BLAST..."
mkdir -p "$(dirname "$DB_OUT")"
makeblastdb -in "$DB_FASTA" -dbtype nucl -out "$DB_OUT"

echo "🔍 Correndo BLAST..."
# Combinamos as reads para um BLAST único (mais eficiente)
cat $QUERY_FILES > combined_temp.fasta

blastn -query combined_temp.fasta \
       -db "$DB_OUT" \
       -out "$OUTPUT_CSV" \
       -outfmt "10 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" \
       -evalue 0.001

rm combined_temp.fasta
echo "✅ Análise concluída! Resultado em: $OUTPUT_CSV"
