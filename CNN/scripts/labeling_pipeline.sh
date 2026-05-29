#!/bin/bash

# Orchestation Script for AMR Annotation.
# V1.

# Exit if any command Exit with non-zero status.

set -e

# Env Variables.

POD5_DIR="../data/raw"
REF_GEN_PATH="../data/reference/"
BASE_CALL_MODEL="sup" # = Super accurate model for Dorado basecalling
OUTPU_DIR="../data/interim"
THREADS=16

# Create the output dir in case does not exist

mkdir -p $OUTPU_DIR

echo "[1 Phase : Initializing Dorado Basecalling]"
echo "Loading data ..."
