"""
Transformer-based Neural Machine Translation Model
Supports: Absolute/Relative Position Embedding, LayerNorm/RMSNorm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (Absolute)"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned Positional Embedding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Relative Position Encoding"""
    
    def __init__(self, dim: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cache', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cache', emb.sin().unsqueeze(0).unsqueeze(0))
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors"""
        seq_len = q.size(2)
        
        if seq_len > self.cos_cache.size(2):
            self._build_cache(seq_len)
        
        cos = self.cos_cache[:, :, :seq_len]
        sin = self.sin_cache[:, :, :seq_len]
        
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional RoPE"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, tgt_len, d_model)
            key: (batch, src_len, d_model)
            value: (batch, src_len, d_model)
            mask: (batch, 1, tgt_len, src_len) or similar
        Returns:
            output: (batch, tgt_len, d_model)
        """
        batch_size = query.size(0)
        
        # Project
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            Q, K = self.rope(Q, K)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_type: str = 'layernorm',
        use_rope: bool = False
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rope)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:  # rmsnorm
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_type: str = 'layernorm',
        use_rope: bool = False
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rope)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rope=False)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:  # rmsnorm
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention
        cross_out = self.cross_attn(x, encoder_output, encoder_output, memory_mask)
        x = self.norm2(x + self.dropout(cross_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x


class TransformerNMT(nn.Module):
    """
    Transformer-based Neural Machine Translation Model
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0,
        sos_idx: int = 2,
        eos_idx: int = 3,
        pos_encoding: str = 'sinusoidal',  # 'sinusoidal', 'learned', 'rope'
        norm_type: str = 'layernorm'  # 'layernorm', 'rmsnorm'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pos_encoding_type = pos_encoding
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=padding_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=padding_idx)
        
        # Position encoding
        use_rope = pos_encoding == 'rope'
        if pos_encoding == 'sinusoidal':
            self.src_pos_encoder = PositionalEncoding(d_model, max_len, dropout)
            self.tgt_pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == 'learned':
            self.src_pos_encoder = LearnedPositionalEmbedding(d_model, max_len, dropout)
            self.tgt_pos_encoder = LearnedPositionalEmbedding(d_model, max_len, dropout)
        else:  # rope - position handled in attention
            self.src_pos_encoder = nn.Dropout(dropout)
            self.tgt_pos_encoder = nn.Dropout(dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, norm_type, use_rope)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, norm_type, use_rope)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder self-attention"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask: 1 for valid tokens, 0 for padding"""
        return (x != self.padding_idx).unsqueeze(1).unsqueeze(2)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode source sequence"""
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_pos_encoder(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence"""
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pos_encoder(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        
        return self.output_proj(x)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
        Returns:
            output: (batch, tgt_len, vocab_size)
        """
        # Create masks
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt)
        
        # Causal mask for decoder
        tgt_len = tgt.size(1)
        causal_mask = self.generate_square_subsequent_mask(tgt_len, tgt.device)
        
        # Combine padding and causal mask
        tgt_mask = tgt_mask & (causal_mask == 0).unsqueeze(0)
        
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        return output
    
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int = 100
    ) -> torch.Tensor:
        """Greedy decoding"""
        batch_size = src.size(0)
        device = src.device
        
        # Encode
        src_mask = self.create_padding_mask(src)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize decoder input with <sos>
        ys = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Decode
            tgt_mask = self.create_padding_mask(ys)
            causal_mask = self.generate_square_subsequent_mask(ys.size(1), device)
            tgt_mask = tgt_mask & (causal_mask == 0).unsqueeze(0)
            
            output = self.decode(ys, encoder_output, tgt_mask, src_mask)
            
            # Get next token
            next_token = output[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            
            # Check if all sequences have <eos>
            if (next_token == self.eos_idx).all():
                break
        
        return ys
    
    def beam_search_decode(
        self,
        src: torch.Tensor,
        beam_width: int = 5,
        max_len: int = 100
    ) -> list:
        """Beam search decoding"""
        batch_size = src.size(0)
        device = src.device
        results = []
        
        for i in range(batch_size):
            src_i = src[i:i+1]
            src_mask = self.create_padding_mask(src_i)
            encoder_output = self.encode(src_i, src_mask)
            
            # Initialize beams: (score, sequence)
            beams = [(0.0, [self.sos_idx])]
            
            for _ in range(max_len):
                new_beams = []
                
                for score, seq in beams:
                    if seq[-1] == self.eos_idx:
                        new_beams.append((score, seq))
                        continue
                    
                    # Prepare decoder input
                    ys = torch.tensor([seq], dtype=torch.long, device=device)
                    tgt_mask = self.create_padding_mask(ys)
                    causal_mask = self.generate_square_subsequent_mask(ys.size(1), device)
                    tgt_mask = tgt_mask & (causal_mask == 0).unsqueeze(0)
                    
                    # Decode
                    output = self.decode(ys, encoder_output, tgt_mask, src_mask)
                    log_probs = F.log_softmax(output[0, -1], dim=-1)
                    
                    # Get top-k
                    topk_probs, topk_indices = log_probs.topk(beam_width)
                    
                    for j in range(beam_width):
                        new_score = score + topk_probs[j].item()
                        new_seq = seq + [topk_indices[j].item()]
                        new_beams.append((new_score, new_seq))
                
                # Keep top beams
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
                
                if all(b[1][-1] == self.eos_idx for b in beams):
                    break
            
            # Get best sequence (excluding <sos> and <eos>)
            best_seq = beams[0][1][1:]
            if best_seq and best_seq[-1] == self.eos_idx:
                best_seq = best_seq[:-1]
            results.append(best_seq)
        
        return results


def create_transformer_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    config: dict = None
) -> TransformerNMT:
    """Factory function to create Transformer model"""
    default_config = {
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_len': 512,
        'padding_idx': 0,
        'sos_idx': 2,
        'eos_idx': 3,
        'pos_encoding': 'sinusoidal',
        'norm_type': 'layernorm'
    }
    
    if config:
        default_config.update(config)
    
    return TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **default_config
    )


if __name__ == "__main__":
    # Test model
    model = create_transformer_model(
        src_vocab_size=10000,
        tgt_vocab_size=8000,
        config={
            'd_model': 256,
            'num_heads': 4,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'pos_encoding': 'rope',
            'norm_type': 'rmsnorm'
        }
    )
    
    batch_size = 4
    src_len = 20
    tgt_len = 15
    
    src = torch.randint(4, 10000, (batch_size, src_len))
    tgt = torch.randint(4, 8000, (batch_size, tgt_len))
    
    # Forward pass
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    # Greedy decode
    preds = model.greedy_decode(src, max_len=20)
    print(f"Predictions shape: {preds.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
