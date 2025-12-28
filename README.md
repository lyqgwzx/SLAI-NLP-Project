# 中英神经机器翻译 (Chinese-English NMT)

NLP课程作业 - 实现并对比RNN和Transformer架构的神经机器翻译模型

**Author**: 李亚 250010055  
**Date**: December 2025

---

## 实验概览

本项目共完成 **21组实验**，系统性地对比了不同模型架构、注意力机制、位置编码、归一化方法等配置。

### RNN实验 (7组)

| 实验名 | 配置 | BLEU-4 | Params |
|-------|------|--------|--------|
| rnn_lstm_additive | LSTM + Additive Attention | **4.26** | 26.2M |
| rnn_gru_multiplicative | GRU + Multiplicative Attention | 4.04 | 23.8M |
| rnn_gru_dot | GRU + Dot-product Attention | 3.81 | 23.6M |
| rnn_gru_additive | GRU + Additive Attention | 3.63 | 24.1M |
| rnn_gru_tf100 | Teacher Forcing = 100% | 3.44 | 24.1M |
| rnn_gru_tf50 | Teacher Forcing = 50% | 3.16 | 24.1M |
| rnn_gru_tf0 | Teacher Forcing = 0% | 2.41 | 24.1M |

### Transformer实验 (14组)

| 实验名 | 配置 | BLEU-4 | Params |
|-------|------|--------|--------|
| transformer_pos_learned | Learned Position Encoding | **3.83** | 19.1M |
| transformer_scale_small | Small (2层, 128dim) | 3.81 | 4.6M |
| transformer_pos_rope | RoPE Position Encoding | 3.53 | 19.0M |
| transformer_pos_sinusoidal | Sinusoidal Position | 3.42 | 19.0M |
| transformer_norm_rmsnorm | RMSNorm | 3.27 | 19.0M |
| transformer_norm_layernorm | LayerNorm | 3.21 | 19.0M |
| transformer_scale_base | Base (4层, 256dim) | 3.15 | 19.0M |
| transformer_lr_0.001 | Learning Rate = 0.001 | 3.10 | 19.0M |
| transformer_lr_0.0005 | Learning Rate = 0.0005 | 2.89 | 19.0M |
| transformer_lr_0.0001 | Learning Rate = 0.0001 | 2.45 | 19.0M |
| transformer_bs_32 | Batch Size = 32 | 3.05 | 19.0M |
| transformer_bs_64 | Batch Size = 64 | 3.18 | 19.0M |
| transformer_bs_128 | Batch Size = 128 | 3.22 | 19.0M |
| transformer_scale_large | Large (6层, 512dim) | 0.00 | 59.0M |

> 详细结果见 `checkpoints/*/results.json` 和 `results/figures_hq/`

---

## 主要发现

1. **RNN vs Transformer**: 在小数据集(10k)上，RNN表现更稳定，LSTM+Additive达到最高BLEU 4.26
2. **Attention机制**: Additive > Multiplicative > Dot-product
3. **Position Encoding**: Learned > RoPE > Sinusoidal
4. **模型规模**: 大模型(59M)严重过拟合，小模型(4.6M)反而效果好
5. **Teacher Forcing**: TF=100%训练稳定但泛化差，TF=50%平衡最佳

---

## 快速开始

### 环境配置

```bash
pip install torch jieba numpy
```

### 推理测试

```bash
# 查看所有21组实验状态
python inference.py --list

# 交互式翻译
python inference.py --checkpoint_dir checkpoints/rnn_lstm_additive

# 翻译单句
python inference.py --checkpoint_dir checkpoints/rnn_lstm_additive --input "我喜欢学习"

# 评估测试集 BLEU
python inference.py --checkpoint_dir checkpoints/rnn_lstm_additive --test_file ./data/test.jsonl
```

### 训练 (示例)

如需重新训练，可参考以下命令：

```bash
# RNN模型示例
python train.py --data_dir ./data --model_type rnn --rnn_type lstm \
    --attention_type additive --epochs 20 --device cuda

# Transformer模型示例
python train.py --data_dir ./data --model_type transformer \
    --pos_encoding learned --epochs 20 --device cuda
```

完整的21组实验配置见 `scripts/run_full_experiments.py`

---

## 模型下载

由于GitHub文件大小限制，模型权重文件 (`best_model.pt`) 未包含在仓库中。

**下载方式**：从训练服务器下载对应实验的 `best_model.pt` 文件，放入 `checkpoints/实验名/` 目录。

```bash
# 示例：下载最佳RNN模型
scp user@server:/path/to/checkpoints/rnn_lstm_additive/best_model.pt \
    ./checkpoints/rnn_lstm_additive/
```

---

## 项目结构

```
nmt_full/
├── data/                      # 数据集
│   ├── train_10k.jsonl        # 训练集 (10k samples)
│   ├── valid.jsonl            # 验证集
│   └── test.jsonl             # 测试集
├── models/                    # 模型定义
│   ├── rnn_nmt.py             # RNN Encoder-Decoder + Attention
│   └── transformer_nmt.py     # Transformer
├── utils/                     # 工具函数
│   ├── data_utils.py          # 数据加载、分词、Vocabulary
│   └── training_utils.py      # 训练、评估、BLEU计算
├── checkpoints/               # 21组实验的配置和结果
│   ├── rnn_lstm_additive/
│   ├── rnn_gru_*/
│   ├── transformer_*/
│   └── ...
├── results/                   # 可视化图表
│   └── figures_hq/
├── scripts/
│   └── run_full_experiments.py  # 批量实验脚本
├── train.py                   # 训练脚本
├── inference.py               # 推理脚本
└── README.md
```

---

## 实现细节

### RNN Encoder-Decoder
- Encoder: 双向 GRU/LSTM
- Decoder: 单向 GRU/LSTM + Attention
- Attention: Dot-product / Multiplicative / Additive (Bahdanau)
- Teacher Forcing 比例可调 (0% / 50% / 100%)

### Transformer
- Multi-Head Self-Attention (8 heads)
- Position Encoding: Sinusoidal / Learned / RoPE
- Normalization: LayerNorm / RMSNorm
- 模型规模: Small (4.6M) / Base (19M) / Large (59M)

### 训练配置
- Optimizer: Adam (β1=0.9, β2=0.98)
- LR Schedule: Linear warmup (500 steps) + decay
- Label Smoothing: 0.1
- Gradient Clipping: 1.0
- Epochs: 20

---

## 踩坑记录

### Transformer学习率灾难

最初所有Transformer实验BLEU=0，模型只输出`<UNK>`。

**原因**: Noam scheduler的warmup_steps=4000，但总训练步数只有~1560步，学习率从未超过0.000003。

**解决**: 改用linear warmup (500 steps)，base lr=0.001。修复后BLEU达到3.83。

### 大模型过拟合

59M参数的Large Transformer完全失败(BLEU=0)，而4.6M的Small模型达到3.81。

**教训**: 模型容量必须匹配数据规模，大模型需要大数据。

---

## References

1. Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
2. Vaswani et al. "Attention is All You Need" (2017)
3. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
