#!/usr/bin/env python3
"""
T5 微调训练脚本
使用 HuggingFace Transformers 对 T5 进行微调用于中英翻译
"""
import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Install transformers: pip install transformers sentencepiece")

from utils.data_utils import load_data
from utils.training_utils import compute_bleu


class T5TranslationDataset(torch.utils.data.Dataset):
    """T5翻译数据集"""
    
    def __init__(self, data, tokenizer, max_length=128, task_prefix="translate Chinese to English: "):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = task_prefix
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = self.task_prefix + item['zh']
        tgt_text = item['en']
        
        src_encoding = self.tokenizer(
            src_text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        tgt_encoding = self.tokenizer(
            tgt_text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        labels = tgt_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': src_encoding['input_ids'].squeeze(),
            'attention_mask': src_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, tokenizer, device, task_prefix):
    """评估模型"""
    model.eval()
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 解码参考
            labels = batch['labels']
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_preds.extend([p.split() for p in preds])
            all_refs.extend([r.split() for r in refs])
    
    bleu = compute_bleu(all_refs, all_preds)
    return bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='t5-small')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if not HAS_TRANSFORMERS:
        print("transformers 未安装，跳过T5实验")
        return
    
    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型和tokenizer
    print(f"Loading {args.model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)
    
    # 加载数据
    print("Loading data...")
    train_data = load_data(os.path.join(args.data_dir, 'train_10k.jsonl'))
    valid_data = load_data(os.path.join(args.data_dir, 'valid.jsonl'))
    test_data = load_data(os.path.join(args.data_dir, 'test.jsonl'))
    
    task_prefix = "translate Chinese to English: "
    
    train_dataset = T5TranslationDataset(train_data, tokenizer, task_prefix=task_prefix)
    valid_dataset = T5TranslationDataset(valid_data, tokenizer, task_prefix=task_prefix)
    test_dataset = T5TranslationDataset(test_data, tokenizer, task_prefix=task_prefix)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    
    # 保存目录
    save_dir = os.path.join(args.save_dir, 't5_finetune')
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练
    best_bleu = 0
    history = {'train_loss': [], 'valid_bleu': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        valid_bleu = evaluate(model, valid_loader, tokenizer, device, task_prefix)
        
        history['train_loss'].append(train_loss)
        history['valid_bleu'].append(valid_bleu)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid BLEU: {valid_bleu:.2f}")
        
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  -> Saved best model (BLEU: {best_bleu:.2f})")
    
    # 测试集评估
    print("\nEvaluating on test set...")
    test_bleu = evaluate(model, test_loader, tokenizer, device, task_prefix)
    print(f"Test BLEU: {test_bleu:.2f}")
    
    # 保存结果
    results = {
        'model_name': args.model_name,
        'test_bleu': test_bleu,
        'best_valid_bleu': best_bleu,
        'history': history
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
