"""
Evaluation metrics and training utilities for NMT
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Optional
import time
import json
import os


def compute_bleu(
    references: List[List[str]],
    hypotheses: List[List[str]],
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> float:
    """
    Compute BLEU score
    
    Args:
        references: List of reference token lists
        hypotheses: List of hypothesis token lists
        max_n: Maximum n-gram order
        weights: Weights for each n-gram (default: uniform)
    
    Returns:
        BLEU score (0-100)
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    # Collect n-gram statistics
    total_matches = [0] * max_n
    total_counts = [0] * max_n
    total_ref_len = 0
    total_hyp_len = 0
    
    for ref, hyp in zip(references, hypotheses):
        total_ref_len += len(ref)
        total_hyp_len += len(hyp)
        
        for n in range(1, max_n + 1):
            # Get n-grams
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1))
            
            # Count matches (clipped)
            matches = 0
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            total_matches[n-1] += matches
            total_counts[n-1] += max(len(hyp) - n + 1, 0)
    
    # Compute precision for each n-gram
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0)
        else:
            precisions.append(total_matches[n] / total_counts[n])
    
    # Compute log precision (with smoothing)
    log_precision = 0
    for n, (p, w) in enumerate(zip(precisions, weights)):
        if p > 0:
            log_precision += w * math.log(p)
        else:
            # Smoothing: if any precision is 0, return 0
            return 0.0
    
    # Brevity penalty
    if total_hyp_len == 0:
        return 0.0
    
    bp = 1.0
    if total_hyp_len < total_ref_len:
        bp = math.exp(1 - total_ref_len / total_hyp_len)
    
    bleu = bp * math.exp(log_precision)
    
    return bleu * 100


def compute_bleu_from_strings(
    references: List[str],
    hypotheses: List[str],
    tokenize_fn=None
) -> float:
    """
    Compute BLEU score from string inputs
    """
    if tokenize_fn is None:
        tokenize_fn = lambda x: x.lower().split()
    
    ref_tokens = [tokenize_fn(r) for r in references]
    hyp_tokens = [tokenize_fn(h) for h in hypotheses]
    
    return compute_bleu(ref_tokens, hyp_tokens)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for NMT
    """
    
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        output: torch.Tensor,  # (batch * tgt_len, vocab_size)
        target: torch.Tensor   # (batch * tgt_len,)
    ) -> torch.Tensor:
        # Create smooth labels
        smooth_target = torch.full_like(output, self.smoothing / (self.vocab_size - 2))
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_target[:, self.padding_idx] = 0
        
        # Mask padding
        mask = (target != self.padding_idx)
        smooth_target = smooth_target * mask.unsqueeze(1)
        
        # Compute loss
        log_probs = torch.log_softmax(output, dim=-1)
        loss = -torch.sum(smooth_target * log_probs) / mask.sum()
        
        return loss


def get_linear_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int
) -> LambdaLR:
    """
    Linear warmup then linear decay scheduler
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


def get_transformer_scheduler(
    optimizer: optim.Optimizer,
    d_model: int,
    warmup_steps: int = 4000
) -> LambdaLR:
    """
    Transformer learning rate scheduler (Noam scheduler)
    """
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Trainer class for NMT models
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[LambdaLR] = None,
        device: torch.device = torch.device('cpu'),
        clip_grad: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.clip_grad = clip_grad
        
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'valid_bleu': [],
            'learning_rate': []
        }
    
    def train_epoch(
        self,
        train_loader,
        teacher_forcing_ratio: float = 1.0,
        model_type: str = 'rnn'
    ) -> float:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if model_type == 'rnn':
                src_lens = batch['src_lens'].to(self.device)
                output = self.model(src, src_lens, tgt, teacher_forcing_ratio)
            else:  # transformer
                output = self.model(src, tgt[:, :-1])
            
            # Compute loss
            if model_type == 'rnn':
                output = output[:, 1:].contiguous().view(-1, output.size(-1))
                target = tgt[:, 1:].contiguous().view(-1)
            else:
                output = output.contiguous().view(-1, output.size(-1))
                target = tgt[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        valid_loader,
        tgt_vocab,
        model_type: str = 'rnn',
        decode_method: str = 'greedy'
    ) -> Tuple[float, float]:
        """
        Evaluate model on validation set
        Returns: (loss, bleu_score)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_references = []
        all_hypotheses = []
        
        for batch in valid_loader:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Compute loss
            if model_type == 'rnn':
                src_lens = batch['src_lens'].to(self.device)
                output = self.model(src, src_lens, tgt, teacher_forcing_ratio=0)
                output_flat = output[:, 1:].contiguous().view(-1, output.size(-1))
            else:
                output = self.model(src, tgt[:, :-1])
                output_flat = output.contiguous().view(-1, output.size(-1))
            
            target = tgt[:, 1:].contiguous().view(-1)
            loss = self.criterion(output_flat, target)
            total_loss += loss.item()
            num_batches += 1
            
            # Generate translations
            if model_type == 'rnn':
                if decode_method == 'greedy':
                    preds, _ = self.model.greedy_decode(src, src_lens)
                    pred_tokens = [tgt_vocab.decode(p.tolist()) for p in preds]
                else:  # beam search
                    pred_seqs = self.model.beam_search_decode(src, src_lens, beam_width=5)
                    pred_tokens = [tgt_vocab.decode(seq) for seq in pred_seqs]
            else:
                if decode_method == 'greedy':
                    preds = self.model.greedy_decode(src)
                    pred_tokens = [tgt_vocab.decode(p.tolist()) for p in preds]
                else:
                    pred_seqs = self.model.beam_search_decode(src, beam_width=5)
                    pred_tokens = [tgt_vocab.decode(seq) for seq in pred_seqs]
            
            # Reference tokens
            ref_tokens = [tgt_vocab.decode(t.tolist()) for t in tgt]
            
            all_hypotheses.extend(pred_tokens)
            all_references.extend(ref_tokens)
        
        # Compute BLEU
        bleu = compute_bleu(all_references, all_hypotheses)
        
        return total_loss / num_batches, bleu
    
    def train(
        self,
        train_loader,
        valid_loader,
        tgt_vocab,
        num_epochs: int,
        model_type: str = 'rnn',
        teacher_forcing_ratio: float = 1.0,
        save_dir: str = 'checkpoints',
        save_best: bool = True
    ) -> Dict:
        """
        Full training loop
        """
        os.makedirs(save_dir, exist_ok=True)
        best_bleu = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(
                train_loader, teacher_forcing_ratio, model_type
            )
            
            # Evaluate
            valid_loss, valid_bleu = self.evaluate(
                valid_loader, tgt_vocab, model_type
            )
            
            # Record history
            lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_bleu'].append(valid_bleu)
            self.history['learning_rate'].append(lr)
            
            elapsed = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Valid Loss: {valid_loss:.4f}")
            print(f"  Valid BLEU: {valid_bleu:.2f}")
            print(f"  LR: {lr:.6f}")
            
            # Save best model
            if save_best and valid_bleu > best_bleu:
                best_bleu = valid_bleu
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'bleu': valid_bleu,
                    'history': self.history
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  -> Saved best model (BLEU: {best_bleu:.2f})")
        
        # Save final model
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, os.path.join(save_dir, 'final_model.pt'))
        
        return self.history


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test BLEU computation
    references = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['there', 'is', 'a', 'cat', 'on', 'the', 'mat']
    ]
    hypotheses = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'cat', 'sat', 'on', 'the', 'mat']
    ]
    
    bleu = compute_bleu(references, hypotheses)
    print(f"BLEU Score: {bleu:.2f}")
