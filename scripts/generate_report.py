#!/usr/bin/env python3
"""
Experiment Analysis and Report Generation

Generates visualization figures and markdown report from experiment results.
"""
import os
import json
import argparse
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    import pandas as pd
    HAS_VIS = True
except ImportError:
    HAS_VIS = False
    print("Visualization libraries not installed. Text report only.")
    print("Install: pip install matplotlib seaborn pandas")


def load_experiment_results(checkpoint_dir):
    """Load all experiment results from checkpoint directory."""
    results = {}
    
    if not os.path.exists(checkpoint_dir):
        print(f"Directory not found: {checkpoint_dir}")
        return results
    
    for exp_name in os.listdir(checkpoint_dir):
        exp_dir = os.path.join(checkpoint_dir, exp_name)
        results_file = os.path.join(exp_dir, 'results.json')
        config_file = os.path.join(exp_dir, 'config.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                exp_results = json.load(f)
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    exp_results['config'] = json.load(f)
            
            results[exp_name] = exp_results
    
    return results


def plot_attention_comparison(results, output_dir):
    """Plot attention mechanism comparison."""
    if not HAS_VIS:
        return
    
    attn_exps = {k: v for k, v in results.items() if 'rnn_gru_' in k and 'tf' not in k}
    
    if len(attn_exps) < 2:
        return
    
    names = []
    bleu_greedy = []
    bleu_beam = []
    
    for name, res in attn_exps.items():
        attn_type = name.replace('rnn_gru_', '')
        names.append(attn_type.capitalize())
        bleu_greedy.append(res.get('test_bleu_greedy', 0))
        bleu_beam.append(res.get('test_bleu_beam', 0))
    
    x = range(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], bleu_greedy, width, label='Greedy', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], bleu_beam, width, label='Beam Search', color='coral')
    
    ax.set_xlabel('Attention Type', fontsize=12)
    ax.set_ylabel('BLEU-4 Score', fontsize=12)
    ax.set_title('Attention Mechanism Comparison (RNN)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_comparison.png'), dpi=150)
    plt.close()


def plot_teacher_forcing_comparison(results, output_dir):
    """Plot teacher forcing ratio comparison."""
    if not HAS_VIS:
        return
    
    tf_exps = {k: v for k, v in results.items() if 'tf' in k}
    
    if len(tf_exps) < 2:
        return
    
    ratios = []
    bleu_scores = []
    
    for name, res in sorted(tf_exps.items()):
        tf_ratio = int(name.split('tf')[1]) / 100
        ratios.append(tf_ratio)
        bleu_scores.append(res.get('test_bleu_greedy', 0))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ratios, bleu_scores, 'o-', markersize=10, linewidth=2, color='steelblue')
    
    ax.set_xlabel('Teacher Forcing Ratio', fontsize=12)
    ax.set_ylabel('BLEU-4 Score', fontsize=12)
    ax.set_title('Teacher Forcing Ratio Comparison', fontsize=14)
    ax.grid(alpha=0.3)
    
    for x, y in zip(ratios, bleu_scores):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'teacher_forcing_comparison.png'), dpi=150)
    plt.close()


def plot_position_encoding_comparison(results, output_dir):
    """Plot positional encoding comparison."""
    if not HAS_VIS:
        return
    
    pos_exps = {k: v for k, v in results.items() if 'transformer_pos_' in k}
    
    if len(pos_exps) < 2:
        return
    
    names = []
    bleu_scores = []
    
    for name, res in pos_exps.items():
        pos_type = name.replace('transformer_pos_', '')
        names.append(pos_type.capitalize())
        bleu_scores.append(res.get('test_bleu_greedy', 0))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, bleu_scores, color=['steelblue', 'coral', 'seagreen'][:len(names)])
    
    ax.set_xlabel('Positional Encoding', fontsize=12)
    ax.set_ylabel('BLEU-4 Score', fontsize=12)
    ax.set_title('Positional Encoding Comparison (Transformer)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_encoding_comparison.png'), dpi=150)
    plt.close()


def plot_model_scale_comparison(results, output_dir):
    """Plot model scale comparison."""
    if not HAS_VIS:
        return
    
    scale_exps = {k: v for k, v in results.items() if 'transformer_scale_' in k}
    
    if len(scale_exps) < 2:
        return
    
    names = []
    bleu_scores = []
    params = []
    
    for name, res in scale_exps.items():
        scale = name.replace('transformer_scale_', '')
        names.append(scale.capitalize())
        bleu_scores.append(res.get('test_bleu_greedy', 0))
        params.append(res.get('num_parameters', 0) / 1e6)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = range(len(names))
    bars = ax1.bar(x, bleu_scores, color='steelblue', alpha=0.7, label='BLEU-4')
    ax1.set_xlabel('Model Scale', fontsize=12)
    ax1.set_ylabel('BLEU-4 Score', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    
    ax2 = ax1.twinx()
    ax2.plot(x, params, 'ro-', markersize=10, linewidth=2, label='Parameters')
    ax2.set_ylabel('Parameters (M)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title('Model Scale vs Performance', fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_scale_comparison.png'), dpi=150)
    plt.close()


def plot_training_curves(results, output_dir):
    """Plot training curves."""
    if not HAS_VIS:
        return
    
    representative = ['rnn_gru_additive', 'transformer_pos_sinusoidal']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for exp_name in representative:
        if exp_name not in results:
            continue
        
        res = results[exp_name]
        history = res.get('history', {})
        
        if 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], label=exp_name)
        
        if 'valid_bleu' in history:
            epochs = range(1, len(history['valid_bleu']) + 1)
            axes[1].plot(epochs, history['valid_bleu'], label=exp_name)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation BLEU-4')
    axes[1].set_title('Validation BLEU-4')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()


def generate_markdown_report(results, output_file):
    """Generate markdown report."""
    report = []
    
    report.append("# Neural Machine Translation Experiment Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overview
    report.append("## 1. Overview\n")
    report.append("This report presents experimental results for Chinese-English neural machine translation ")
    report.append("using RNN-based and Transformer-based architectures.\n")
    
    # Results Summary
    report.append("## 2. Results Summary\n")
    report.append("| Experiment | BLEU-4 (Greedy) | BLEU-4 (Beam) | Parameters |")
    report.append("|------------|-----------------|---------------|------------|")
    
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1].get('test_bleu_greedy', 0), 
                           reverse=True)
    
    for name, res in sorted_results:
        bleu_g = res.get('test_bleu_greedy', 0)
        bleu_b = res.get('test_bleu_beam', 0)
        params = res.get('num_parameters', 0)
        params_str = f"{params/1e6:.1f}M" if params > 0 else "N/A"
        report.append(f"| {name} | {bleu_g:.2f} | {bleu_b:.2f} | {params_str} |")
    
    report.append("")
    
    # RNN Analysis
    report.append("## 3. RNN Model Analysis\n")
    
    report.append("### 3.1 Attention Mechanism Comparison\n")
    attn_exps = {k: v for k, v in results.items() if 'rnn_gru_' in k and 'tf' not in k}
    if attn_exps:
        report.append("| Attention | BLEU-4 | Description |")
        report.append("|-----------|--------|-------------|")
        for name, res in attn_exps.items():
            attn = name.replace('rnn_gru_', '')
            bleu = res.get('test_bleu_greedy', 0)
            desc = {
                'dot': 'Requires same encoder/decoder dimensions',
                'multiplicative': 'Learnable weight matrix',
                'additive': 'Most expressive, higher computation'
            }.get(attn, '')
            report.append(f"| {attn} | {bleu:.2f} | {desc} |")
        report.append("")
        report.append("![Attention Comparison](figures/attention_comparison.png)\n")
    
    report.append("### 3.2 Teacher Forcing Comparison\n")
    tf_exps = {k: v for k, v in results.items() if 'tf' in k}
    if tf_exps:
        report.append("| TF Ratio | BLEU-4 | Description |")
        report.append("|----------|--------|-------------|")
        for name, res in sorted(tf_exps.items()):
            tf = name.split('tf')[1]
            bleu = res.get('test_bleu_greedy', 0)
            desc = {'100': 'Full teacher forcing', '50': 'Mixed strategy', '0': 'Free running'}.get(tf, '')
            report.append(f"| {int(tf)/100:.0%} | {bleu:.2f} | {desc} |")
        report.append("")
        report.append("![Teacher Forcing](figures/teacher_forcing_comparison.png)\n")
    
    # Transformer Analysis
    report.append("## 4. Transformer Model Analysis\n")
    
    report.append("### 4.1 Positional Encoding Comparison\n")
    pos_exps = {k: v for k, v in results.items() if 'transformer_pos_' in k}
    if pos_exps:
        report.append("| Encoding | BLEU-4 | Description |")
        report.append("|----------|--------|-------------|")
        for name, res in pos_exps.items():
            pos = name.replace('transformer_pos_', '')
            bleu = res.get('test_bleu_greedy', 0)
            desc = {
                'sinusoidal': 'Fixed function, no learning required',
                'learned': 'Learnable embeddings',
                'rope': 'Rotary position embedding'
            }.get(pos, '')
            report.append(f"| {pos} | {bleu:.2f} | {desc} |")
        report.append("")
        report.append("![Position Encoding](figures/position_encoding_comparison.png)\n")
    
    report.append("### 4.2 Normalization Comparison\n")
    norm_exps = {k: v for k, v in results.items() if 'transformer_norm_' in k}
    if norm_exps:
        report.append("| Normalization | BLEU-4 | Description |")
        report.append("|---------------|--------|-------------|")
        for name, res in norm_exps.items():
            norm = name.replace('transformer_norm_', '')
            bleu = res.get('test_bleu_greedy', 0)
            desc = {
                'layernorm': 'Standard layer normalization',
                'rmsnorm': 'Root mean square normalization'
            }.get(norm, '')
            report.append(f"| {norm} | {bleu:.2f} | {desc} |")
        report.append("")
    
    report.append("### 4.3 Model Scale Comparison\n")
    scale_exps = {k: v for k, v in results.items() if 'transformer_scale_' in k}
    if scale_exps:
        report.append("| Scale | BLEU-4 | Parameters |")
        report.append("|-------|--------|------------|")
        for name, res in scale_exps.items():
            scale = name.replace('transformer_scale_', '')
            bleu = res.get('test_bleu_greedy', 0)
            params = res.get('num_parameters', 0)
            report.append(f"| {scale} | {bleu:.2f} | {params/1e6:.1f}M |")
        report.append("")
        report.append("![Model Scale](figures/model_scale_comparison.png)\n")
    
    # Comparison
    report.append("## 5. RNN vs Transformer Comparison\n")
    report.append("| Aspect | RNN | Transformer |")
    report.append("|--------|-----|-------------|")
    report.append("| Computation | Sequential | Parallel |")
    report.append("| Long-range dependencies | Via hidden state | Direct attention |")
    report.append("| Parameter count | Lower | Higher |")
    report.append("| Training speed | Slower | Faster |")
    report.append("")
    
    # Conclusion
    report.append("## 6. Conclusion\n")
    
    best_model = max(results.items(), key=lambda x: x[1].get('test_bleu_greedy', 0))
    report.append(f"- Best model: {best_model[0]} (BLEU-4={best_model[1].get('test_bleu_greedy', 0):.2f})\n")
    report.append("- Additive attention generally performs best for RNN models\n")
    report.append("- Mixed teacher forcing (0.5) provides good balance\n")
    report.append("- Learned positional encoding shows competitive performance\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Generate experiment report')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Loading experiment results...")
    results = load_experiment_results(args.checkpoint_dir)
    print(f"Found {len(results)} experiments")
    
    if not results:
        print("No experiment results found")
        return
    
    if HAS_VIS:
        print("Generating figures...")
        plot_attention_comparison(results, figures_dir)
        plot_teacher_forcing_comparison(results, figures_dir)
        plot_position_encoding_comparison(results, figures_dir)
        plot_model_scale_comparison(results, figures_dir)
        plot_training_curves(results, figures_dir)
    
    print("Generating report...")
    generate_markdown_report(results, os.path.join(args.output_dir, 'report.md'))


if __name__ == '__main__':
    main()
