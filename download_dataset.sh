#!/bin/bash

set -e

echo "========================================"
echo "English Dialects Dataset Download"
echo "========================================"

# Check for required packages
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Install required packages for preprocessing
pip install datasets scikit-learn

# Create data directory
mkdir -p data

# Download English Dialects dataset
echo ""
echo "Downloading English Dialects dataset from Google Drive..."
gdown YOUR_FILE_ID_HERE -O english_dialects_full.zip

echo "Extracting dataset..."
unzip -q english_dialects_full.zip -d data/temp/
rm english_dialects_full.zip

# Split dataset into train/validation/test
echo ""
echo "Splitting dataset into train/validation/test..."
python preprocessing.py \
    --save_dir ./data/english_dialects \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# Clean up temp directory
rm -rf data/temp/

# Verify splits
if [ -d "data/english_dialects/train" ] && \
   [ -d "data/english_dialects/validation" ] && \
   [ -d "data/english_dialects/test" ]; then
    echo ""
    echo "========================================"
    echo "✅ Dataset preparation complete!"
    echo "========================================"
    echo "Train: ./data/english_dialects/train"
    echo "Validation: ./data/english_dialects/validation"
    echo "Test: ./data/english_dialects/test"
    echo ""
    echo "Next steps:"
    echo "  1. python train.py"
    echo "  2. python evaluate.py --split test"
    echo "========================================"
else
    echo "❌ Error: Split creation failed"
    exit 1
fi
