#!/usr/bin/env python3
"""
Inference script for NMT models
One-click inference as required by the assignment

Usage:
    # 交互式翻译
    python inference.py --checkpoint_dir checkpoints/rnn_lstm_additive
    
    # 翻译单句
    python inference.py --checkpoint_dir checkpoints/rnn_lstm_additive --input "今天天气很好"
    
    # 评估测试集
    python inference.py --checkpoint_dir checkpoints/rnn_lstm_additive --test_file ./data/test.jsonl
"""
import os
import sys
import json
import argparse
import torch
from typing import List

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.data_utils import (
    Vocabulary, load_data, tokenize_chinese, tokenize_english
)
from utils.training_utils import compute_bleu
from models.rnn_nmt import create_rnn_model
from models.transformer_nmt import create_transformer_model


def ensure_vocab_exists(checkpoint_dir: str):
    """
    确保vocab文件存在，如果不存在则从训练数据生成
    """
    src_vocab_path = os.path.join(checkpoint_dir, 'src_vocab.json')
    tgt_vocab_path = os.path.join(checkpoint_dir, 'tgt_vocab.json')
    
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        return True
    
    print("⚠ Vocab files not found, generating from training data...")
    
    # 查找数据目录
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    train_file = os.path.join(data_dir, 'train_10k.jsonl')
    
    if not os.path.exists(train_file):
        print(f"❌ Training data not found: {train_file}")
        return False
    
    # 动态导入生成函数（避免循环依赖）
    import re
    from collections import Counter
    import jieba
    
    class VocabBuilder:
        PAD_TOKEN = '<pad>'
        UNK_TOKEN = '<unk>'
        SOS_TOKEN = '<sos>'
        EOS_TOKEN = '<eos>'
        
        def __init__(self, min_freq=2):
            self.min_freq = min_freq
            self.word2idx = {}
            self.word_freq = Counter()
            special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
            for idx, token in enumerate(special_tokens):
                self.word2idx[token] = idx
        
        def build_vocab(self, sentences):
            for sentence in sentences:
                self.word_freq.update(sentence)
            idx = len(self.word2idx)
            for word, freq in self.word_freq.items():
                if freq >= self.min_freq and word not in self.word2idx:
                    self.word2idx[word] = idx
                    idx += 1
        
        def save(self, path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'word2idx': self.word2idx, 'min_freq': self.min_freq}, f, ensure_ascii=False, indent=2)
    
    def tokenize_zh(text):
        text = re.sub(r'\s+', ' ', text).strip()
        return [t for t in jieba.cut(text) if t.strip()]
    
    def tokenize_en(text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.,!?;:\'\"\(\)\[\]])', r' \1 ', text)
        return text.lower().split()
    
    # 加载数据并构建词表
    print(f"  Loading: {train_file}")
    train_data = load_data(train_file)
    
    src_vocab = VocabBuilder(min_freq=2)
    tgt_vocab = VocabBuilder(min_freq=2)
    
    src_sentences = [tokenize_zh(item['zh']) for item in train_data]
    tgt_sentences = [tokenize_en(item['en']) for item in train_data]
    
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    src_vocab.save(src_vocab_path)
    tgt_vocab.save(tgt_vocab_path)
    
    print(f"  ✓ Generated src_vocab.json ({len(src_vocab.word2idx)} tokens)")
    print(f"  ✓ Generated tgt_vocab.json ({len(tgt_vocab.word2idx)} tokens)")
    
    return True


def check_model_file(checkpoint_dir: str):
    """
    检查模型文件是否存在，如果不存在给出下载提示
    """
    model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    if os.path.exists(model_path):
        return True
    
    exp_name = os.path.basename(checkpoint_dir)
    print(f"""
❌ Model file not found: {model_path}

Please download the model weights from the training server:

    scp user@server:/path/to/checkpoints/{exp_name}/best_model.pt \\
        {checkpoint_dir}/

Or use rsync to sync all checkpoints:

    rsync -avz --include='*/' --include='best_model.pt' --exclude='*' \\
        user@server:/path/to/checkpoints/ ./checkpoints/
""")
    return False


def load_model(checkpoint_dir: str, device: torch.device):
    """
    Load model from checkpoint directory
    """
    # 自动检查和生成vocab
    if not ensure_vocab_exists(checkpoint_dir):
        raise FileNotFoundError("Cannot generate vocab files. Please check data directory.")
    
    # 检查模型文件
    if not check_model_file(checkpoint_dir):
        raise FileNotFoundError(f"Model file not found: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load vocabularies
    src_vocab = Vocabulary.load(os.path.join(checkpoint_dir, 'src_vocab.json'))
    tgt_vocab = Vocabulary.load(os.path.join(checkpoint_dir, 'tgt_vocab.json'))
    
    # Create model
    model_type = config['model_type']
    
    if model_type == 'rnn':
        model_config = {
            'embed_dim': config['embed_dim'],
            'hidden_dim': config['hidden_dim'],
            'num_layers': config['num_layers'],
            'rnn_type': config['rnn_type'],
            'attention_type': config['attention_type'],
            'dropout': config['dropout'],
            'padding_idx': src_vocab.pad_idx,
            'sos_idx': tgt_vocab.sos_idx,
            'eos_idx': tgt_vocab.eos_idx
        }
        model = create_rnn_model(len(src_vocab), len(tgt_vocab), model_config)
        
    elif model_type == 'transformer':
        model_config = {
            'd_model': config['embed_dim'],
            'num_heads': config['num_heads'],
            'num_encoder_layers': config['num_layers'],
            'num_decoder_layers': config['num_layers'],
            'd_ff': config['d_ff'],
            'dropout': config['dropout'],
            'max_len': config['max_len'],
            'padding_idx': src_vocab.pad_idx,
            'sos_idx': tgt_vocab.sos_idx,
            'eos_idx': tgt_vocab.eos_idx,
            'pos_encoding': config['pos_encoding'],
            'norm_type': config['norm_type']
        }
        model = create_transformer_model(len(src_vocab), len(tgt_vocab), model_config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, 'best_model.pt'),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, src_vocab, tgt_vocab, config


def translate_sentence(
    sentence: str,
    model,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    config: dict,
    device: torch.device,
    decode_method: str = 'beam',
    beam_width: int = 5,
    max_len: int = 100
) -> str:
    """
    Translate a single Chinese sentence to English
    """
    model.eval()
    
    # Tokenize
    tokens = tokenize_chinese(sentence)
    tokens = tokens[:config['max_len'] - 2]
    
    # Encode
    indices = src_vocab.encode(tokens, add_sos=True, add_eos=True)
    src = torch.tensor([indices], dtype=torch.long, device=device)
    
    with torch.no_grad():
        if config['model_type'] == 'rnn':
            src_lens = torch.tensor([len(indices)], device=device)
            
            if decode_method == 'greedy':
                preds, _ = model.greedy_decode(src, src_lens, max_len=max_len)
                pred_indices = preds[0].tolist()
            else:
                pred_seqs = model.beam_search_decode(src, src_lens, beam_width=beam_width, max_len=max_len)
                pred_indices = pred_seqs[0]
        else:
            if decode_method == 'greedy':
                preds = model.greedy_decode(src, max_len=max_len)
                pred_indices = preds[0].tolist()
            else:
                pred_seqs = model.beam_search_decode(src, beam_width=beam_width, max_len=max_len)
                pred_indices = pred_seqs[0]
    
    # Decode
    pred_tokens = tgt_vocab.decode(pred_indices)
    translation = ' '.join(pred_tokens)
    
    return translation


def translate_batch(
    sentences: List[str],
    model,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    config: dict,
    device: torch.device,
    decode_method: str = 'beam',
    beam_width: int = 5
) -> List[str]:
    """
    Translate a batch of sentences
    """
    translations = []
    for sentence in sentences:
        translation = translate_sentence(
            sentence, model, src_vocab, tgt_vocab, config, device,
            decode_method, beam_width
        )
        translations.append(translation)
    return translations


def evaluate_test_set(
    test_path: str,
    model,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    config: dict,
    device: torch.device,
    decode_method: str = 'beam'
) -> dict:
    """
    Evaluate model on test set and compute BLEU score
    """
    # Load test data
    test_data = load_data(test_path)
    
    references = []
    hypotheses = []
    
    print(f"Translating {len(test_data)} sentences...")
    
    for i, item in enumerate(test_data):
        # Get reference
        ref_tokens = tgt_vocab.decode(tgt_vocab.encode(tokenize_english(item['en']), add_sos=True, add_eos=True))
        references.append(ref_tokens)
        
        # Translate
        translation = translate_sentence(
            item['zh'], model, src_vocab, tgt_vocab, config, device, decode_method
        )
        hyp_tokens = translation.split()
        hypotheses.append(hyp_tokens)
        
        if (i + 1) % 50 == 0:
            print(f"  Translated {i+1}/{len(test_data)} sentences")
    
    # Compute BLEU
    bleu = compute_bleu(references, hypotheses)
    
    return {
        'bleu': bleu,
        'num_samples': len(test_data),
        'decode_method': decode_method
    }


def find_best_checkpoint(checkpoints_dir: str):
    """
    自动查找可用的最佳checkpoint
    """
    if not os.path.exists(checkpoints_dir):
        return None
    
    best_exp = None
    best_bleu = -1
    
    for exp_name in os.listdir(checkpoints_dir):
        exp_dir = os.path.join(checkpoints_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        
        # 检查必要文件
        model_path = os.path.join(exp_dir, 'best_model.pt')
        config_path = os.path.join(exp_dir, 'config.json')
        results_path = os.path.join(exp_dir, 'results.json')
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            continue
        
        # 读取BLEU分数
        bleu = 0
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    results = json.load(f)
                    bleu = results.get('test_bleu_beam', 0)
            except:
                pass
        
        if bleu > best_bleu:
            best_bleu = bleu
            best_exp = exp_dir
    
    return best_exp


def list_available_checkpoints(checkpoints_dir: str):
    """
    列出所有可用的checkpoint
    """
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return
    
    print("\n" + "="*60)
    print("Available Checkpoints")
    print("="*60)
    
    available = []
    missing_model = []
    
    for exp_name in sorted(os.listdir(checkpoints_dir)):
        exp_dir = os.path.join(checkpoints_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        
        config_path = os.path.join(exp_dir, 'config.json')
        if not os.path.exists(config_path):
            continue
        
        model_path = os.path.join(exp_dir, 'best_model.pt')
        results_path = os.path.join(exp_dir, 'results.json')
        
        bleu = "N/A"
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    results = json.load(f)
                    bleu = f"{results.get('test_bleu_beam', 0):.2f}"
            except:
                pass
        
        if os.path.exists(model_path):
            available.append((exp_name, bleu))
            print(f"  ✓ {exp_name:35} BLEU={bleu}")
        else:
            missing_model.append(exp_name)
    
    if missing_model:
        print(f"\n  Missing model files ({len(missing_model)}):")
        for exp in missing_model[:5]:
            print(f"    ✗ {exp}")
        if len(missing_model) > 5:
            print(f"    ... and {len(missing_model)-5} more")
    
    print(f"\nReady: {len(available)}/{len(available)+len(missing_model)}")
    
    if available:
        print(f"\nUsage example:")
        print(f"  python inference.py --checkpoint_dir checkpoints/{available[0][0]}")


def main():
    parser = argparse.ArgumentParser(description='NMT Inference')
    
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing model checkpoint (auto-detect if not specified)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Test file path for evaluation')
    parser.add_argument('--input', type=str, default=None,
                        help='Input Chinese text to translate')
    parser.add_argument('--decode_method', type=str, default='beam',
                        choices=['greedy', 'beam'],
                        help='Decoding method')
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Beam width for beam search')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/mps/cpu/auto)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for translations')
    parser.add_argument('--list', action='store_true',
                        help='List available checkpoints')
    
    args = parser.parse_args()
    
    # 默认checkpoints目录
    default_checkpoints_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    
    # 列出可用模型
    if args.list:
        list_available_checkpoints(default_checkpoints_dir)
        return
    
    # 自动选择checkpoint
    if args.checkpoint_dir is None:
        print("No checkpoint specified, searching for best available model...")
        args.checkpoint_dir = find_best_checkpoint(default_checkpoints_dir)
        
        if args.checkpoint_dir is None:
            print("\n❌ No available checkpoints found!")
            print("\nPlease either:")
            print("  1. Specify a checkpoint: --checkpoint_dir checkpoints/rnn_lstm_additive")
            print("  2. Download model weights from training server")
            print("\nTo see what's available:")
            print("  python inference.py --list")
            return
        
        print(f"  → Auto-selected: {os.path.basename(args.checkpoint_dir)}")
    
    # Get device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint_dir}")
    try:
        model, src_vocab, tgt_vocab, config = load_model(args.checkpoint_dir, device)
    except FileNotFoundError as e:
        print(f"\n{e}")
        return
    
    print(f"Model type: {config['model_type']}")
    
    # Interactive or batch mode
    if args.input:
        # Translate single input
        translation = translate_sentence(
            args.input, model, src_vocab, tgt_vocab, config, device,
            args.decode_method, args.beam_width
        )
        print(f"\nInput: {args.input}")
        print(f"Translation: {translation}")
        
    elif args.test_file:
        # Evaluate on test set
        print(f"\nEvaluating on: {args.test_file}")
        results = evaluate_test_set(
            args.test_file, model, src_vocab, tgt_vocab, config, device,
            args.decode_method
        )
        print(f"\nResults:")
        print(f"  BLEU Score: {results['bleu']:.2f}")
        print(f"  Samples: {results['num_samples']}")
        print(f"  Decode Method: {results['decode_method']}")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Results saved to: {args.output}")
        
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("Interactive Translation Mode")
        print("Enter Chinese text to translate (type 'quit' to exit)")
        print("="*50)
        
        # 先翻译几个示例
        examples = ["今天天气很好", "机器翻译是自然语言处理的重要任务"]
        print("\nExamples:")
        for zh in examples:
            en = translate_sentence(zh, model, src_vocab, tgt_vocab, config, device, args.decode_method, args.beam_width)
            print(f"  中: {zh}")
            print(f"  英: {en}\n")
        
        print("-" * 50)
        
        while True:
            try:
                text = input("\nChinese: ").strip()
                if text.lower() == 'quit':
                    break
                if not text:
                    continue
                    
                translation = translate_sentence(
                    text, model, src_vocab, tgt_vocab, config, device,
                    args.decode_method, args.beam_width
                )
                print(f"English: {translation}")
                
            except KeyboardInterrupt:
                break
        
        print("\nSession ended.")


if __name__ == '__main__':
    main()
