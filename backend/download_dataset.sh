#!/bin/bash

set -e

echo "========================================"
echo "English Dialects Dataset Download"
echo "========================================"

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

mkdir -p data

echo ""
echo "Downloading English Dialects dataset (TAR.GZ format)..."
gdown 1N9OOK7s6c7NoUKbIMn5eAcDGX-SeZkeS -O english_dialects.tar.gz

echo "Extracting dataset..."
tar -xzvf english_dialects.tar.gz -C data/

rm english_dialects.tar.gz

if [ -d "data/english_dialects" ]; then
    echo ""
    echo "✅ Dataset extraction complete!"
    echo "Location: ./data/english_dialects/"
else
    echo "❌ Extraction failed"
    exit 1
fi
