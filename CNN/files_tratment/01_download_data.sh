#!/bin/bash
# Configurações de Caminhos
BASE_DIR="$HOME/Klebsiella_POD5"
DATA_DIR="$BASE_DIR/pod5_data/KP1779"
URL_KP1779="https://data.narodni-repozitar.cz/general/datasets/dj8ys-a4r49/files/KP1779_pod5.tar.gz"

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# 1. Download do Dorado (se não existir)
if [ ! -d "dorado-1.4.0-linux-x64" ]; then
    echo "⬇️ Descarregando Dorado..."
    curl -L "https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.4.0-linux-x64.tar.gz" -o dorado.tar.gz
    tar -xzf dorado.tar.gz && rm dorado.tar.gz
fi

# 2. Download do NanoBLAST (Repositório)
if [ ! -d "NanoBLAST" ]; then
    echo "⬇️ Clonando NanoBLAST..."
    git clone https://github.com/MarketaJakubickova/NanoBLAST.git
fi

# 3. Download da Estirpe KP1779
mkdir -p "$BASE_DIR/pod5_data"
echo "⬇️ Descarregando KP1779 (1.5GB)..."
wget -q --show-progress -O "$BASE_DIR/pod5_data/KP1779_pod5.tar.gz" "$URL_KP1779"

# 4. Extração e Limpeza imediata
echo "📦 Extraindo e limpando..."
mkdir -p "$DATA_DIR"
tar -xzf "$BASE_DIR/pod5_data/KP1779_pod5.tar.gz" -C "$DATA_DIR"
rm "$BASE_DIR/pod5_data/KP1779_pod5.tar.gz"
rm -rf "$DATA_DIR/pod5_fail"

# 5. Renomeação Padronizada (KP1779_01, 02...)
echo "🏷️ Renomeando ficheiros POD5..."
count=1
find "$DATA_DIR/pod5_pass" -name "*.pod5" | sort | while read f; do
    new_name=$(printf "KP1779_%02d.pod5" $count)
    mv "$f" "$(dirname "$f")/$new_name"
    ((count++))
done

echo "✅ Setup concluído com sucesso."
