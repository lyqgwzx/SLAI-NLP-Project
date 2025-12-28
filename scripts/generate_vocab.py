#!/usr/bin/env python3
"""
本地Vocab生成脚本

由于所有实验使用相同的训练数据，vocab是一样的。
这个脚本可以在本地从数据文件重新生成vocab，
这样只需要从云端下载best_model.pt文件。

Usage:
    python generate_vocab.py --data_dir ./data --output_dir ./checkpoints
"""
import os
import sys
import json
import argparse
import re
from collections import Counter

try:
    import jieba
except ImportError:
    print("请先安装jieba: pip install jieba")
    sys.exit(1)


class Vocabulary:
    """轻量级词表类（不依赖torch）"""
    
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def build_vocab(self, sentences):
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'min_freq': self.min_freq
            }, f, ensure_ascii=False, indent=2)


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_chinese(text):
    text = clean_text(text)
    tokens = list(jieba.cut(text))
    return [t for t in tokens if t.strip()]


def tokenize_english(text):
    text = clean_text(text)
    text = re.sub(r'([.,!?;:\'\"\(\)\[\]])', r' \1 ', text)
    return text.lower().split()


def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def build_vocabularies(train_data, src_lang='zh', tgt_lang='en', min_freq=2):
    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)
    
    src_sentences = []
    tgt_sentences = []
    
    for item in train_data:
        if src_lang == 'zh':
            src_sentences.append(tokenize_chinese(item[src_lang]))
        else:
            src_sentences.append(tokenize_english(item[src_lang]))
        
        if tgt_lang == 'zh':
            tgt_sentences.append(tokenize_chinese(item[tgt_lang]))
        else:
            tgt_sentences.append(tokenize_english(item[tgt_lang]))
    
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)
    
    return src_vocab, tgt_vocab


def generate_vocab(data_dir, output_dir, train_file='train_10k.jsonl', min_freq=2):
    """生成vocab文件并保存到所有checkpoint目录"""
    
    # 加载训练数据
    train_path = os.path.join(data_dir, train_file)
    print(f"Loading training data from: {train_path}")
    train_data = load_data(train_path)
    print(f"Loaded {len(train_data)} samples")
    
    # 构建词表
    print("\nBuilding vocabularies...")
    src_vocab, tgt_vocab = build_vocabularies(train_data, 'zh', 'en', min_freq)
    
    # 保存到所有checkpoint目录
    print(f"\nSaving vocabularies to checkpoint directories in: {output_dir}")
    
    saved_count = 0
    for exp_name in os.listdir(output_dir):
        exp_dir = os.path.join(output_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        
        # 检查是否有config.json（确认是实验目录）
        config_path = os.path.join(exp_dir, 'config.json')
        if not os.path.exists(config_path):
            continue
        
        # 保存vocab
        src_vocab_path = os.path.join(exp_dir, 'src_vocab.json')
        tgt_vocab_path = os.path.join(exp_dir, 'tgt_vocab.json')
        
        src_vocab.save(src_vocab_path)
        tgt_vocab.save(tgt_vocab_path)
        
        print(f"  ✓ {exp_name}/")
        saved_count += 1
    
    print(f"\n✅ Generated vocab for {saved_count} experiments")
    print(f"   Source vocab size: {len(src_vocab)}")
    print(f"   Target vocab size: {len(tgt_vocab)}")
    
    return src_vocab, tgt_vocab


def main():
    parser = argparse.ArgumentParser(description='Generate vocabulary files locally')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory containing checkpoint folders')
    parser.add_argument('--train_file', type=str, default='train_10k.jsonl',
                        help='Training data filename')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='Minimum word frequency for vocabulary')
    
    args = parser.parse_args()
    
    generate_vocab(args.data_dir, args.output_dir, args.train_file, args.min_freq)


if __name__ == '__main__':
    main()
