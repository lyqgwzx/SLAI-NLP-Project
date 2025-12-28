#!/usr/bin/env python3
"""
Main training script for NMT models
Supports: RNN (GRU/LSTM), Transformer, T5 fine-tuning
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import create_dataloaders, load_data, build_vocabularies
from utils.training_utils import (
    Trainer, LabelSmoothingLoss, 
    get_linear_warmup_scheduler, get_transformer_scheduler,
    count_parameters
)
from models.rnn_nmt import create_rnn_model
from models.transformer_nmt import create_transformer_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train NMT Model')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train/valid/test JSONL files')
    parser.add_argument('--train_file', type=str, default='train_10k.jsonl',
                        help='Training data filename')
    parser.add_argument('--valid_file', type=str, default='valid.jsonl',
                        help='Validation data filename')
    parser.add_argument('--test_file', type=str, default='test.jsonl',
                        help='Test data filename')
    
    # Model
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['rnn', 'transformer', 't5'],
                        help='Model type')
    parser.add_argument('--rnn_type', type=str, default='gru',
                        choices=['gru', 'lstm'],
                        help='RNN cell type')
    parser.add_argument('--attention_type', type=str, default='dot',
                        choices=['dot', 'multiplicative', 'additive'],
                        help='Attention type for RNN model')
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'learned', 'rope'],
                        help='Position encoding for Transformer')
    parser.add_argument('--norm_type', type=str, default='layernorm',
                        choices=['layernorm', 'rmsnorm'],
                        help='Normalization type for Transformer')
    
    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads (Transformer)')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward dimension (Transformer)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='Warmup steps')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--teacher_forcing', type=float, default=1.0,
                        help='Teacher forcing ratio (RNN only)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing')
    
    # Other
    parser.add_argument('--max_len', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='Minimum word frequency for vocabulary')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/mps/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get appropriate device"""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{args.model_type}_{timestamp}"
    
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataloaders
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(
        train_path=os.path.join(args.data_dir, args.train_file),
        valid_path=os.path.join(args.data_dir, args.valid_file),
        test_path=os.path.join(args.data_dir, args.test_file),
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=args.min_freq
    )
    
    # Save vocabularies
    src_vocab.save(os.path.join(save_dir, 'src_vocab.json'))
    tgt_vocab.save(os.path.join(save_dir, 'tgt_vocab.json'))
    
    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    
    if args.model_type == 'rnn':
        model_config = {
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'rnn_type': args.rnn_type,
            'attention_type': args.attention_type,
            'dropout': args.dropout,
            'padding_idx': src_vocab.pad_idx,
            'sos_idx': tgt_vocab.sos_idx,
            'eos_idx': tgt_vocab.eos_idx
        }
        model = create_rnn_model(len(src_vocab), len(tgt_vocab), model_config)
        
    elif args.model_type == 'transformer':
        d_model = args.embed_dim
        model_config = {
            'd_model': d_model,
            'num_heads': args.num_heads,
            'num_encoder_layers': args.num_layers,
            'num_decoder_layers': args.num_layers,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'max_len': args.max_len,
            'padding_idx': src_vocab.pad_idx,
            'sos_idx': tgt_vocab.sos_idx,
            'eos_idx': tgt_vocab.eos_idx,
            'pos_encoding': args.pos_encoding,
            'norm_type': args.norm_type
        }
        model = create_transformer_model(len(src_vocab), len(tgt_vocab), model_config)
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    num_params = count_parameters(model)
    print(f"Model: {args.model_type}")
    print(f"Total parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    total_steps = len(train_loader) * args.epochs
    if args.model_type == 'transformer':
        scheduler = get_transformer_scheduler(optimizer, d_model, args.warmup_steps)
    else:
        scheduler = get_linear_warmup_scheduler(optimizer, args.warmup_steps, total_steps)
    
    # Loss function
    criterion = LabelSmoothingLoss(
        len(tgt_vocab),
        padding_idx=tgt_vocab.pad_idx,
        smoothing=args.label_smoothing
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        clip_grad=args.clip_grad
    )
    
    # Train
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    
    history = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        tgt_vocab=tgt_vocab,
        num_epochs=args.epochs,
        model_type=args.model_type,
        teacher_forcing_ratio=args.teacher_forcing,
        save_dir=save_dir,
        save_best=True
    )
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model = model
    
    test_loss, test_bleu = trainer.evaluate(
        test_loader, tgt_vocab, args.model_type, decode_method='greedy'
    )
    print(f"Test Loss (greedy): {test_loss:.4f}")
    print(f"Test BLEU (greedy): {test_bleu:.2f}")
    
    test_loss_beam, test_bleu_beam = trainer.evaluate(
        test_loader, tgt_vocab, args.model_type, decode_method='beam'
    )
    print(f"Test Loss (beam): {test_loss_beam:.4f}")
    print(f"Test BLEU (beam): {test_bleu_beam:.2f}")
    
    # Save results
    results = {
        'test_loss_greedy': test_loss,
        'test_bleu_greedy': test_bleu,
        'test_loss_beam': test_loss_beam,
        'test_bleu_beam': test_bleu_beam,
        'best_valid_bleu': checkpoint['bleu'],
        'num_parameters': num_params,
        'history': history
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
