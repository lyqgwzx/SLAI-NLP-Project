"""
Data utilities for Chinese-English Machine Translation
"""
import json
import re
import jieba
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Tuple, Dict, Optional
import numpy as np


class Vocabulary:
    """Vocabulary class for mapping tokens to indices"""
    
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens"""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    @property
    def pad_idx(self):
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_idx(self):
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def sos_idx(self):
        return self.word2idx[self.SOS_TOKEN]
    
    @property
    def eos_idx(self):
        return self.word2idx[self.EOS_TOKEN]
    
    def build_vocab(self, sentences: List[List[str]]):
        """Build vocabulary from tokenized sentences"""
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def encode(self, tokens: List[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert tokens to indices"""
        indices = []
        if add_sos:
            indices.append(self.sos_idx)
        indices.extend([self.word2idx.get(token, self.unk_idx) for token in tokens])
        if add_eos:
            indices.append(self.eos_idx)
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """Convert indices to tokens"""
        special_indices = {self.pad_idx, self.sos_idx, self.eos_idx}
        tokens = []
        for idx in indices:
            if remove_special and idx in special_indices:
                continue
            if idx == self.eos_idx:
                break
            tokens.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return tokens
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'min_freq': self.min_freq
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls(min_freq=data['min_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(v): k for k, v in data['word2idx'].items()}
        return vocab


def clean_text(text: str, lang: str = 'en') -> str:
    """Clean text by removing illegal characters"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def tokenize_chinese(text: str) -> List[str]:
    """Tokenize Chinese text using jieba"""
    text = clean_text(text, 'zh')
    tokens = list(jieba.cut(text))
    # Remove empty tokens
    tokens = [t for t in tokens if t.strip()]
    return tokens


def tokenize_english(text: str) -> List[str]:
    """Tokenize English text (simple whitespace + punctuation split)"""
    text = clean_text(text, 'en')
    # Add space around punctuation
    text = re.sub(r'([.,!?;:\'\"\(\)\[\]])', r' \1 ', text)
    tokens = text.lower().split()
    return tokens


def load_data(filepath: str) -> List[Dict]:
    """Load JSONL data file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class TranslationDataset(Dataset):
    """Dataset for machine translation"""
    
    def __init__(
        self,
        data: List[Dict],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        src_lang: str = 'zh',
        tgt_lang: str = 'en',
        max_len: int = 128
    ):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        
        # Tokenize data
        self.tokenized_data = self._tokenize_data()
    
    def _tokenize_data(self) -> List[Tuple[List[int], List[int]]]:
        """Tokenize and encode all data"""
        tokenized = []
        for item in self.data:
            src_text = item[self.src_lang]
            tgt_text = item[self.tgt_lang]
            
            # Tokenize
            if self.src_lang == 'zh':
                src_tokens = tokenize_chinese(src_text)
            else:
                src_tokens = tokenize_english(src_text)
            
            if self.tgt_lang == 'zh':
                tgt_tokens = tokenize_chinese(tgt_text)
            else:
                tgt_tokens = tokenize_english(tgt_text)
            
            # Truncate
            src_tokens = src_tokens[:self.max_len - 2]
            tgt_tokens = tgt_tokens[:self.max_len - 2]
            
            # Encode
            src_indices = self.src_vocab.encode(src_tokens, add_sos=True, add_eos=True)
            tgt_indices = self.tgt_vocab.encode(tgt_tokens, add_sos=True, add_eos=True)
            
            tokenized.append((src_indices, tgt_indices))
        
        return tokenized
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        src_indices, tgt_indices = self.tokenized_data[idx]
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_len': len(src_indices),
            'tgt_len': len(tgt_indices)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    # Sort by source length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['src_len'], reverse=True)
    
    src_lens = [item['src_len'] for item in batch]
    tgt_lens = [item['tgt_len'] for item in batch]
    
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    
    batch_size = len(batch)
    
    # Pad sequences
    src_padded = torch.zeros(batch_size, max_src_len, dtype=torch.long)
    tgt_padded = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    
    for i, item in enumerate(batch):
        src_padded[i, :item['src_len']] = item['src']
        tgt_padded[i, :item['tgt_len']] = item['tgt']
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lens': torch.tensor(src_lens, dtype=torch.long),
        'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long)
    }


def build_vocabularies(
    train_data: List[Dict],
    src_lang: str = 'zh',
    tgt_lang: str = 'en',
    min_freq: int = 2
) -> Tuple[Vocabulary, Vocabulary]:
    """Build source and target vocabularies from training data"""
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
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    return src_vocab, tgt_vocab


def create_dataloaders(
    train_path: str,
    valid_path: str,
    test_path: str,
    src_lang: str = 'zh',
    tgt_lang: str = 'en',
    batch_size: int = 32,
    max_len: int = 128,
    min_freq: int = 2,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """Create train, validation, and test dataloaders"""
    
    # Load data
    print("Loading data...")
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)
    test_data = load_data(test_path)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    src_vocab, tgt_vocab = build_vocabularies(train_data, src_lang, tgt_lang, min_freq)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_len)
    valid_dataset = TranslationDataset(valid_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_len)
    test_dataset = TranslationDataset(test_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    # Test data loading
    data_dir = "/home/claude/dataset/AP0004_Midterm&Final_translation_dataset_zh_en"
    
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(
        train_path=f"{data_dir}/train_10k.jsonl",
        valid_path=f"{data_dir}/valid.jsonl",
        test_path=f"{data_dir}/test.jsonl",
        batch_size=32
    )
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  src: {batch['src'].shape}")
    print(f"  tgt: {batch['tgt'].shape}")
