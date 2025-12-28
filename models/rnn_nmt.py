"""
RNN-based Neural Machine Translation Model
Supports: GRU/LSTM, Multiple Attention Types, Teacher Forcing, Beam Search
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional, List
import math


class Attention(nn.Module):
    """
    Attention mechanism supporting different alignment functions:
    - dot: Dot-product attention
    - multiplicative: Multiplicative (Luong) attention  
    - additive: Additive (Bahdanau) attention
    """
    
    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_type: str = 'dot'
    ):
        super().__init__()
        self.attention_type = attention_type
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        if attention_type == 'dot':
            # Dot product attention: requires same dimensions
            assert encoder_hidden_dim == decoder_hidden_dim, \
                "Dot attention requires same encoder/decoder hidden dims"
        
        elif attention_type == 'multiplicative':
            # Multiplicative attention: score = h_t^T * W * h_s
            self.W = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        
        elif attention_type == 'additive':
            # Additive attention: score = v^T * tanh(W_1*h_s + W_2*h_t)
            self.W1 = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
            self.W2 = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
            self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,  # (batch, decoder_hidden_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, encoder_hidden_dim)
        mask: Optional[torch.Tensor] = None  # (batch, src_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context: (batch, encoder_hidden_dim)
            attention_weights: (batch, src_len)
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        if self.attention_type == 'dot':
            # (batch, src_len)
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        elif self.attention_type == 'multiplicative':
            # (batch, src_len, decoder_hidden_dim)
            transformed = self.W(encoder_outputs)
            # (batch, src_len)
            scores = torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        elif self.attention_type == 'additive':
            # (batch, src_len, decoder_hidden_dim)
            encoder_proj = self.W1(encoder_outputs)
            # (batch, 1, decoder_hidden_dim)
            decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
            # (batch, src_len, decoder_hidden_dim)
            combined = torch.tanh(encoder_proj + decoder_proj)
            # (batch, src_len)
            scores = self.v(combined).squeeze(2)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class Encoder(nn.Module):
    """Bidirectional RNN Encoder"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = 'gru',
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        rnn_class = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = rnn_class(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
    
    def forward(
        self,
        src: torch.Tensor,  # (batch, src_len)
        src_lens: torch.Tensor  # (batch,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            outputs: (batch, src_len, hidden_dim)
            hidden: (num_layers, batch, hidden_dim) or tuple for LSTM
        """
        # Embedding
        embedded = self.dropout(self.embedding(src))
        
        # Pack sequence
        packed = pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # RNN forward
        packed_outputs, hidden = self.rnn(packed)
        
        # Unpack
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        return outputs, hidden


class Decoder(nn.Module):
    """RNN Decoder with Attention"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = 'gru',
        attention_type: str = 'dot',
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        # Attention
        self.attention = Attention(encoder_hidden_dim, hidden_dim, attention_type)
        
        # RNN input: embedding + context
        rnn_class = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = rnn_class(
            embed_dim + encoder_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim + encoder_hidden_dim + embed_dim, vocab_size)
    
    def forward(
        self,
        input_token: torch.Tensor,  # (batch,)
        hidden: torch.Tensor,  # (num_layers, batch, hidden_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, encoder_hidden_dim)
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step decoding
        Returns:
            output: (batch, vocab_size) - logits
            hidden: updated hidden state
            attention_weights: (batch, src_len)
        """
        # Embedding: (batch, embed_dim)
        embedded = self.dropout(self.embedding(input_token))
        
        # Get decoder hidden for attention
        if self.rnn_type == 'lstm':
            decoder_hidden_for_attn = hidden[0][-1]  # Last layer hidden state
        else:
            decoder_hidden_for_attn = hidden[-1]  # Last layer hidden state
        
        # Attention
        context, attention_weights = self.attention(
            decoder_hidden_for_attn, encoder_outputs, mask
        )
        
        # RNN input: concat embedding and context
        rnn_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        
        # RNN forward
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        
        # Output projection
        output = self.fc_out(torch.cat([output, context, embedded], dim=1))
        
        return output, hidden, attention_weights


class Seq2SeqRNN(nn.Module):
    """
    Sequence-to-Sequence model with RNN encoder-decoder and attention
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        rnn_type: str = 'gru',
        attention_type: str = 'dot',
        dropout: float = 0.3,
        padding_idx: int = 0,
        sos_idx: int = 2,
        eos_idx: int = 3
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.padding_idx = padding_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.rnn_type = rnn_type
        
        self.encoder = Encoder(
            src_vocab_size, embed_dim, hidden_dim, num_layers,
            rnn_type, dropout, padding_idx
        )
        self.decoder = Decoder(
            tgt_vocab_size, embed_dim, hidden_dim, hidden_dim,
            num_layers, rnn_type, attention_type, dropout, padding_idx
        )
    
    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for source sequence"""
        return (src != self.padding_idx).float()
    
    def forward(
        self,
        src: torch.Tensor,  # (batch, src_len)
        src_lens: torch.Tensor,  # (batch,)
        tgt: torch.Tensor,  # (batch, tgt_len)
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing
        Returns:
            outputs: (batch, tgt_len, vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        device = src.device
        
        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)
        
        # Prepare decoder
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=device)
        input_token = tgt[:, 0]  # <sos> token
        
        # Decode step by step
        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            
            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = tgt[:, t]
            else:
                input_token = output.argmax(dim=1)
        
        return outputs
    
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        max_len: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding
        Returns:
            predictions: (batch, max_len)
            attention_weights: (batch, max_len, src_len)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)
        
        # Initialize
        predictions = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        attention_weights = torch.zeros(batch_size, max_len, src.size(1), device=device)
        input_token = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=device)
        
        # Decode
        for t in range(max_len):
            output, hidden, attn = self.decoder(input_token, hidden, encoder_outputs, mask)
            predictions[:, t] = output.argmax(dim=1)
            attention_weights[:, t, :attn.size(1)] = attn
            input_token = predictions[:, t]
        
        return predictions, attention_weights
    
    def beam_search_decode(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        beam_width: int = 5,
        max_len: int = 100
    ) -> List[List[int]]:
        """
        Beam search decoding
        Returns:
            List of best sequences for each batch item
        """
        batch_size = src.size(0)
        device = src.device
        results = []
        
        # Process each sample individually for beam search
        for i in range(batch_size):
            src_i = src[i:i+1]
            src_lens_i = src_lens[i:i+1]
            
            # Encode
            encoder_outputs, hidden = self.encoder(src_i, src_lens_i)
            mask = torch.ones(1, encoder_outputs.size(1), device=device)
            
            # Initialize beams: (score, sequence, hidden)
            beams = [(0.0, [self.sos_idx], hidden)]
            
            for _ in range(max_len):
                new_beams = []
                
                for score, seq, h in beams:
                    if seq[-1] == self.eos_idx:
                        new_beams.append((score, seq, h))
                        continue
                    
                    # Decode one step
                    input_token = torch.tensor([seq[-1]], device=device)
                    output, new_h, _ = self.decoder(input_token, h, encoder_outputs, mask)
                    log_probs = F.log_softmax(output, dim=1)
                    
                    # Get top-k candidates
                    topk_probs, topk_indices = log_probs.topk(beam_width)
                    
                    for j in range(beam_width):
                        new_score = score + topk_probs[0, j].item()
                        new_seq = seq + [topk_indices[0, j].item()]
                        new_beams.append((new_score, new_seq, new_h))
                
                # Keep top beams
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
                
                # Check if all beams ended
                if all(b[1][-1] == self.eos_idx for b in beams):
                    break
            
            # Get best sequence (excluding <sos>)
            best_seq = beams[0][1][1:]
            # Remove <eos> if present
            if best_seq and best_seq[-1] == self.eos_idx:
                best_seq = best_seq[:-1]
            results.append(best_seq)
        
        return results


def create_rnn_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    config: dict = None
) -> Seq2SeqRNN:
    """Factory function to create RNN model"""
    default_config = {
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'rnn_type': 'gru',
        'attention_type': 'dot',
        'dropout': 0.3,
        'padding_idx': 0,
        'sos_idx': 2,
        'eos_idx': 3
    }
    
    if config:
        default_config.update(config)
    
    return Seq2SeqRNN(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **default_config
    )


if __name__ == "__main__":
    # Test model
    model = create_rnn_model(
        src_vocab_size=10000,
        tgt_vocab_size=8000,
        config={'attention_type': 'additive'}
    )
    
    batch_size = 4
    src_len = 20
    tgt_len = 15
    
    src = torch.randint(4, 10000, (batch_size, src_len))
    src_lens = torch.tensor([20, 18, 15, 10])
    tgt = torch.randint(4, 8000, (batch_size, tgt_len))
    
    # Forward pass
    outputs = model(src, src_lens, tgt, teacher_forcing_ratio=0.5)
    print(f"Output shape: {outputs.shape}")
    
    # Greedy decode
    preds, attn = model.greedy_decode(src, src_lens, max_len=20)
    print(f"Predictions shape: {preds.shape}")
    
    # Beam search
    beam_results = model.beam_search_decode(src, src_lens, beam_width=3, max_len=20)
    print(f"Beam search results: {len(beam_results)} sequences")
