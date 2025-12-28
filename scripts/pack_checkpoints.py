#!/usr/bin/env python3
"""
云端打包脚本 - 在服务器上运行
从每个实验文件夹中选择best_model.pt和vocab文件打包

Usage (在云端服务器上运行):
    python pack_checkpoints.py --checkpoint_dir /path/to/checkpoints --output_dir /path/to/output

这个脚本会:
1. 扫描所有实验文件夹
2. 从每个文件夹中复制: best_model.pt, src_vocab.json, tgt_vocab.json
3. 打包成一个tar.gz文件，方便下载
"""
import os
import sys
import shutil
import argparse
import tarfile
from pathlib import Path


def get_checkpoint_files(exp_dir):
    """获取实验目录中的checkpoint文件列表"""
    pt_files = []
    for f in os.listdir(exp_dir):
        if f.endswith('.pt'):
            full_path = os.path.join(exp_dir, f)
            pt_files.append({
                'name': f,
                'path': full_path,
                'size': os.path.getsize(full_path),
                'mtime': os.path.getmtime(full_path)
            })
    return pt_files


def select_best_checkpoint(pt_files):
    """选择最佳的checkpoint文件"""
    # 优先选择best_model.pt
    for f in pt_files:
        if f['name'] == 'best_model.pt':
            return f
    
    # 否则选择最新的checkpoint
    if pt_files:
        return max(pt_files, key=lambda x: x['mtime'])
    
    return None


def pack_experiments(checkpoint_dir, output_dir, experiments=None):
    """打包指定的实验"""
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有实验目录
    if experiments:
        exp_dirs = [checkpoint_dir / exp for exp in experiments]
    else:
        exp_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    packed_dir = output_dir / 'packed_checkpoints'
    packed_dir.mkdir(exist_ok=True)
    
    total_size = 0
    packed_count = 0
    
    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name
        print(f"\nProcessing: {exp_name}")
        
        # 获取checkpoint文件
        pt_files = get_checkpoint_files(exp_dir)
        if not pt_files:
            print(f"  ⚠ No .pt files found, skipping")
            continue
        
        # 选择最佳checkpoint
        best_ckpt = select_best_checkpoint(pt_files)
        print(f"  Selected: {best_ckpt['name']} ({best_ckpt['size']/1024/1024:.1f} MB)")
        
        # 创建输出目录
        out_exp_dir = packed_dir / exp_name
        out_exp_dir.mkdir(exist_ok=True)
        
        # 复制文件
        files_to_copy = [
            best_ckpt['path'],  # 模型权重
            exp_dir / 'src_vocab.json',
            exp_dir / 'tgt_vocab.json',
            exp_dir / 'config.json',
            exp_dir / 'results.json'
        ]
        
        for src_file in files_to_copy:
            src_path = Path(src_file)
            if src_path.exists():
                dst_path = out_exp_dir / src_path.name
                # 如果是checkpoint但不叫best_model.pt，重命名
                if src_path.suffix == '.pt' and src_path.name != 'best_model.pt':
                    dst_path = out_exp_dir / 'best_model.pt'
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ Copied: {src_path.name} -> {dst_path.name}")
                total_size += src_path.stat().st_size
            else:
                print(f"  ✗ Missing: {src_path.name}")
        
        packed_count += 1
    
    print(f"\n{'='*50}")
    print(f"Packed {packed_count} experiments")
    print(f"Total size: {total_size/1024/1024:.1f} MB")
    
    # 创建tar.gz压缩包
    tar_path = output_dir / 'checkpoints_packed.tar.gz'
    print(f"\nCreating archive: {tar_path}")
    
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(packed_dir, arcname='checkpoints')
    
    tar_size = tar_path.stat().st_size
    print(f"Archive size: {tar_size/1024/1024:.1f} MB")
    print(f"\n✅ Done! Download: {tar_path}")
    
    return tar_path


def main():
    parser = argparse.ArgumentParser(description='Pack checkpoint files for download')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing experiment checkpoints')
    parser.add_argument('--output_dir', type=str, default='./packed',
                        help='Output directory for packed files')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiments to pack (default: all)')
    parser.add_argument('--list', action='store_true',
                        help='List all experiments and their sizes')
    
    args = parser.parse_args()
    
    if args.list:
        # 列出所有实验及其大小
        checkpoint_dir = Path(args.checkpoint_dir)
        print(f"Experiments in {checkpoint_dir}:\n")
        
        total_size = 0
        for exp_dir in sorted(checkpoint_dir.iterdir()):
            if exp_dir.is_dir():
                pt_files = get_checkpoint_files(exp_dir)
                exp_size = sum(f['size'] for f in pt_files)
                total_size += exp_size
                best = select_best_checkpoint(pt_files)
                best_info = f" (best: {best['name']})" if best else ""
                print(f"  {exp_dir.name}: {len(pt_files)} files, {exp_size/1024/1024:.1f} MB{best_info}")
        
        print(f"\nTotal: {total_size/1024/1024:.1f} MB")
        return
    
    pack_experiments(args.checkpoint_dir, args.output_dir, args.experiments)


if __name__ == '__main__':
    main()
