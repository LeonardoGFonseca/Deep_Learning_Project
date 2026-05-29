#!/bin/bash

# In case of a failure of any command.

set -e

# Setting up variables.
# --- 1. Project Structure ---
echo "📁 Creating directory structure..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

ROOT_SRC_DIR="$(dirname "$SCRIPT_DIR")"

ROOT_DIR="$(dirname "$ROOT_SRC_DIR")"

# Move to the root dir so all the relative routes start from there.

cd "$ROOT_DIR" || { echo "Error: No se pudo acceder a $ROOT_DIR"; exit 1; }

echo " Initializing project Structure in $ROOT_DIR"

mkdir -p bin data/raw data/processed configs docs notebooks scripts tests data/ref

# Installation of Dorado.

DORADO_VERSION="0.5.0"

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64) DORADO_PLATFORM="linux-x64" ;;
      aarch64) DORADO_PLATFORM="linux-arm64" ;;
      *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$ARCH" in
      x86_64) DORADO_PLATFORM="osx-x64" ;;
      arm64)  DORADO_PLATFORM="osx-arm64" ;;
      *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS"; exit 1
    ;;
esac

DORADO_TAR="dorado-${DORADO_VERSION}-${DORADO_PLATFORM}.tar.gz"
DORADO_DIR="dorado-${DORADO_VERSION}-${DORADO_PLATFORM}"

if [ ! -f "bin/dorado" ]; then
  echo "Downloading Dorado ${DORADO_VERSION} for ${DORADO_PLATFORM}..."
  curl -L "https://cdn.oxfordnanoportal.com/software/analysis/${DORADO_TAR}" -o dorado.tar.gz
  tar -xzf dorado.tar.gz
  echo "Moving Dorado"

  # move dorado to /bin
  mv "${DORADO_DIR}/bin/dorado" bin/
  mv "${DORADO_DIR}/lib" bin/

  # Cleaning the tar and extracted folder.
  rm -rf "${DORADO_DIR}" dorado.tar.gz
  echo "Dorado ${DORADO_VERSION} installed in bin/"
fi

