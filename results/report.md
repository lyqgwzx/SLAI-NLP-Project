# Neural Machine Translation Experiment Report

## 1. Overview

This report presents experimental results for Chinese-English neural machine translation using RNN-based and Transformer-based architectures.

## 2. Results Summary

| Experiment | BLEU-4 (Greedy) | BLEU-4 (Beam) | Parameters |
|------------|-----------------|---------------|------------|
| rnn_lstm_additive | 3.08 | 4.26 | 26.2M |
| rnn_gru_multiplicative | 3.19 | 4.04 | 23.8M |
| transformer_pos_learned | 2.50 | 3.83 | 19.1M |
| transformer_scale_small | 2.47 | 3.81 | 4.6M |

## 3. RNN Model Analysis

### 3.1 Attention Mechanism Comparison

| Attention | BLEU-4 | Description |
|-----------|--------|-------------|
| dot | 2.62 | Requires same encoder/decoder dimensions |
| multiplicative | 3.19 | Learnable weight matrix |
| additive | 3.08 | Most expressive, higher computation |

### 3.2 Teacher Forcing Comparison

| TF Ratio | BLEU-4 | Description |
|----------|--------|-------------|
| 100% | 2.48 | Full teacher forcing |
| 50% | 3.23 | Mixed strategy |
| 0% | 2.41 | Free running |

## 4. Transformer Model Analysis

### 4.1 Positional Encoding Comparison

| Encoding | BLEU-4 | Description |
|----------|--------|-------------|
| sinusoidal | 2.33 | Fixed function |
| learned | 2.50 | Learnable embeddings |
| rope | 2.37 | Rotary position embedding |

### 4.2 Normalization Comparison

| Normalization | BLEU-4 |
|---------------|--------|
| LayerNorm | 2.33 |
| RMSNorm | 2.39 |

### 4.3 Model Scale Comparison

| Scale | BLEU-4 | Parameters |
|-------|--------|------------|
| Small | 2.47 | 4.6M |
| Base | 3.47 | 14.8M |
| Large | 0.00 | 59.0M |

## 5. Conclusion

- Best model: rnn_lstm_additive (BLEU-4=4.26)
- Additive attention performs well for RNN models
- Learned positional encoding shows competitive performance
- Large Transformer models may overfit on small datasets
