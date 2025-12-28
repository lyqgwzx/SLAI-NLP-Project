#!/bin/bash
# Neural Machine Translation - Experiment Runner
# Executes all experiments and generates analysis report

set -e

export DATA_DIR="${DATA_DIR:-./data}"
export SAVE_DIR="${SAVE_DIR:-./checkpoints}"
export EPOCHS="${EPOCHS:-20}"
export DEVICE="${DEVICE:-cuda}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

echo "[1/5] Installing dependencies..."
pip install torch jieba numpy matplotlib seaborn pandas tqdm -q
pip install transformers sentencepiece sacrebleu -q

echo "[2/5] Verifying GPU configuration..."
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  [{i}] {torch.cuda.get_device_name(i)}')
"

echo "[3/5] Running experiments..."
python scripts/run_full_experiments.py --experiments all

echo "[4/5] Generating report..."
python scripts/generate_report.py --checkpoint_dir $SAVE_DIR --output_dir results

echo "[5/5] Summary"
if [ -f "results/report.md" ]; then
    echo "Output: checkpoints/, results/report.md, results/figures/"
else
    echo "Warning: Report generation failed"
fi
