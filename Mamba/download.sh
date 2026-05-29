#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[*] Setting up dependencies..."
bash src/utils/setup_dependencies.sh

echo "[*] Setting up Python virtual environment..."
if [ ! -d "deep_env" ]; then
    python3 -m venv deep_env
fi
source deep_env/bin/activate
pip install -q -r requirements.txt

echo "[*] Downloading and verifying raw data..."
python src/scripts/download_data.py

echo "[+] Done. Run 'python src/scripts/run_dorado.py' to start basecalling and alignment."
