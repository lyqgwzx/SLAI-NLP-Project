#!/usr/bin/env python3
"""
NMT Experiment Runner

Systematic evaluation of neural machine translation architectures:
- RNN: GRU/LSTM variants, attention mechanisms, teacher forcing
- Transformer: positional encoding, normalization, hyperparameters
- T5: fine-tuning baseline
"""
import os
import sys
import subprocess
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.environ.get('DATA_DIR', './data')
SAVE_DIR = os.environ.get('SAVE_DIR', './checkpoints')
EPOCHS = int(os.environ.get('EPOCHS', '20'))
DEVICE = os.environ.get('DEVICE', 'cuda')


def run_experiment(name: str, args: list):
    """Execute a single experiment."""
    cmd = [
        'python', 'train.py',
        '--data_dir', DATA_DIR,
        '--save_dir', SAVE_DIR,
        '--exp_name', name,
        '--device', DEVICE,
        '--epochs', str(EPOCHS)
    ] + args
    
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + '\n')
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_rnn_experiments():
    """
    RNN-based seq2seq experiments.
    
    Variables:
    - RNN cell: GRU vs LSTM
    - Attention: dot-product, multiplicative, additive
    - Teacher forcing ratio: 1.0, 0.5, 0.0
    """
    experiments = []
    
    # Attention mechanism comparison (GRU baseline)
    for attn in ['dot', 'multiplicative', 'additive']:
        experiments.append((
            f'rnn_gru_{attn}',
            ['--model_type', 'rnn', '--rnn_type', 'gru', '--attention_type', attn,
             '--batch_size', '128', '--embed_dim', '256', '--hidden_dim', '512']
        ))
    
    # RNN cell comparison (additive attention baseline)
    experiments.append((
        'rnn_lstm_additive',
        ['--model_type', 'rnn', '--rnn_type', 'lstm', '--attention_type', 'additive',
         '--batch_size', '128', '--embed_dim', '256', '--hidden_dim', '512']
    ))
    
    # Teacher forcing ratio comparison
    for tf_ratio in [1.0, 0.5, 0.0]:
        experiments.append((
            f'rnn_gru_tf{int(tf_ratio*100)}',
            ['--model_type', 'rnn', '--rnn_type', 'gru', '--attention_type', 'additive',
             '--teacher_forcing', str(tf_ratio), '--batch_size', '128']
        ))
    
    print(f"\nRNN Experiments: {len(experiments)}")
    
    results = {}
    for name, args in experiments:
        success = run_experiment(name, args)
        results[name] = 'success' if success else 'failed'
    
    return results


def run_transformer_experiments():
    """
    Transformer experiments.
    
    Variables:
    - Positional encoding: sinusoidal, learned, RoPE
    - Normalization: LayerNorm, RMSNorm
    - Model scale: small, base, large
    - Learning rate: 1e-4, 5e-4, 1e-3
    - Batch size: 32, 64, 128
    """
    experiments = []
    
    # Positional encoding comparison
    for pos_enc in ['sinusoidal', 'learned', 'rope']:
        experiments.append((
            f'transformer_pos_{pos_enc}',
            ['--model_type', 'transformer', '--pos_encoding', pos_enc,
             '--norm_type', 'layernorm', '--embed_dim', '256', '--num_layers', '4',
             '--num_heads', '8', '--batch_size', '128']
        ))
    
    # Normalization comparison
    for norm in ['layernorm', 'rmsnorm']:
        experiments.append((
            f'transformer_norm_{norm}',
            ['--model_type', 'transformer', '--pos_encoding', 'sinusoidal',
             '--norm_type', norm, '--embed_dim', '256', '--num_layers', '4',
             '--num_heads', '8', '--batch_size', '128']
        ))
    
    # Model scale comparison
    scales = {
        'small': {'embed_dim': '128', 'num_layers': '2', 'num_heads': '4', 'd_ff': '512'},
        'base':  {'embed_dim': '256', 'num_layers': '4', 'num_heads': '8', 'd_ff': '1024'},
        'large': {'embed_dim': '512', 'num_layers': '6', 'num_heads': '8', 'd_ff': '2048'}
    }
    for scale_name, config in scales.items():
        experiments.append((
            f'transformer_scale_{scale_name}',
            ['--model_type', 'transformer', '--pos_encoding', 'sinusoidal',
             '--embed_dim', config['embed_dim'], '--num_layers', config['num_layers'],
             '--num_heads', config['num_heads'], '--d_ff', config['d_ff'],
             '--batch_size', '128' if scale_name != 'large' else '64']
        ))
    
    # Learning rate comparison
    for lr in ['0.0001', '0.0005', '0.001']:
        experiments.append((
            f'transformer_lr_{lr}',
            ['--model_type', 'transformer', '--lr', lr,
             '--embed_dim', '256', '--num_layers', '4', '--batch_size', '128']
        ))
    
    # Batch size comparison
    for bs in ['32', '64', '128']:
        experiments.append((
            f'transformer_bs_{bs}',
            ['--model_type', 'transformer', '--batch_size', bs,
             '--embed_dim', '256', '--num_layers', '4']
        ))
    
    print(f"\nTransformer Experiments: {len(experiments)}")
    
    results = {}
    for name, args in experiments:
        success = run_experiment(name, args)
        results[name] = 'success' if success else 'failed'
    
    return results


def run_t5_experiment():
    """T5 fine-tuning experiment."""
    print("\nT5 Fine-tuning Experiment")
    
    cmd = [
        'python', 'scripts/train_t5.py',
        '--data_dir', DATA_DIR,
        '--save_dir', SAVE_DIR,
        '--epochs', '5',
        '--batch_size', '16',
        '--device', DEVICE
    ]
    
    result = subprocess.run(cmd)
    return {'t5_finetune': 'success' if result.returncode == 0 else 'failed'}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NMT Experiment Runner')
    parser.add_argument('--experiments', type=str, nargs='+',
                        default=['rnn', 'transformer'],
                        choices=['rnn', 'transformer', 't5', 'all'])
    parser.add_argument('--quick', action='store_true',
                        help='Run subset of experiments')
    args = parser.parse_args()
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    all_results = {}
    
    if args.quick:
        print("Running reduced experiment set")
        global EPOCHS
        EPOCHS = 10
    
    if 'all' in args.experiments or 'rnn' in args.experiments:
        rnn_results = run_rnn_experiments()
        all_results.update(rnn_results)
    
    if 'all' in args.experiments or 'transformer' in args.experiments:
        trans_results = run_transformer_experiments()
        all_results.update(trans_results)
    
    if 'all' in args.experiments or 't5' in args.experiments:
        t5_results = run_t5_experiment()
        all_results.update(t5_results)
    
    # Save experiment status
    status_file = os.path.join(SAVE_DIR, 'experiment_status.json')
    with open(status_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    
    # Summary
    success = sum(1 for v in all_results.values() if v == 'success')
    total = len(all_results)
    print(f"\nCompleted: {success}/{total}")
    
    for name, status in all_results.items():
        mark = '[OK]' if status == 'success' else '[FAIL]'
        print(f"  {mark} {name}")


if __name__ == '__main__':
    main()
