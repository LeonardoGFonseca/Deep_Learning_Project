#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INDEX="$ROOT_DIR/data/data_index/data_ids.txt"
LOG_DIR="$ROOT_DIR/pipeline_logs"
FAILED=""
COMPLETED=0
TOTAL=0

mkdir -p "$LOG_DIR"

echo "============================================"
echo "  NanoSquiggle-AMR Pipeline Orchestrator"
echo "============================================"

# --- Setup ---
echo "[*] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq samtools curl wget

echo "[*] Setting up Python virtual environment..."
if [ ! -d "deep_env" ]; then
    python3 -m venv deep_env
fi
source deep_env/bin/activate
pip install -q -r requirements.txt

echo "[*] Installing Dorado..."
bash src/utils/setup_dependencies.sh

# --- Read strain list (strip CR from Windows line endings) ---
mapfile -t strains < <(tr -d '\r' < "$INDEX")
TOTAL=${#strains[@]}
echo "[*] Pipeline target: $TOTAL strains"

# --- Process each strain ---
for strain in "${strains[@]}"; do
    echo ""
    echo "===== [$((COMPLETED + 1))/$TOTAL] Processing $strain ====="

    STRAIN_LOG="$LOG_DIR/${strain}.log"
    START_TS=$(date +%s)

    # Phase 1: Download
    echo "[*] Phase 1: Downloading $strain..."
    python src/scripts/download_data.py --strain "$strain" 2>&1 | tee -a "$STRAIN_LOG"
    if [ $? -ne 0 ]; then
        echo "[-] $strain: Phase 1 failed. Skipping."
        FAILED="$FAILED $strain"
        continue
    fi

    # Phase 2: Dorado basecalling
    echo "[*] Phase 2: Basecalling $strain..."
    python src/scripts/run_dorado.py --strain "$strain" 2>&1 | tee -a "$STRAIN_LOG"
    if [ $? -ne 0 ]; then
        echo "[-] $strain: Phase 2 failed. Skipping."
        FAILED="$FAILED $strain"
        # Cleanup POD5 anyway to save disk
        rm -rf "$ROOT_DIR/data/raw/$strain"
        continue
    fi

    # Phase 3: Compile features
    echo "[*] Phase 3: Compiling features for $strain..."
    python src/scripts/compile_features_store.py --strain "$strain" --append 2>&1 | tee -a "$STRAIN_LOG"
    if [ $? -ne 0 ]; then
        echo "[-] $strain: Phase 3 failed."
        FAILED="$FAILED $strain"
    fi

    # --- Cleanup intermediates to save disk ---
    echo "[*] Cleaning up intermediates for $strain..."
    rm -rf "$ROOT_DIR/data/raw/$strain"
    rm -f "$ROOT_DIR/data/processed/alignments/$strain.bam"
    rm -f "$ROOT_DIR/data/processed/alignments/$strain.bam.bai"

    ELAPSED=$(( $(date +%s) - START_TS ))
    echo "[+] $strain complete in ${ELAPSED}s"
    COMPLETED=$((COMPLETED + 1))
done

# --- Summary ---
echo ""
echo "============================================"
echo "  Pipeline Complete"
echo "============================================"
echo "  Successful: $COMPLETED / $TOTAL"
if [ -n "$FAILED" ]; then
    echo "  Failed:    $FAILED"
    echo "  Logs:      $LOG_DIR/"
fi
echo "  Feature store: $ROOT_DIR/data/processed/hd5f/amr_features.h5"
echo "============================================"
