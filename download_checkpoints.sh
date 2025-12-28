#!/bin/bash
# download_checkpoints.sh
# 从云端服务器下载checkpoint文件
#
# 使用方法:
#   chmod +x download_checkpoints.sh
#   ./download_checkpoints.sh
#
# 或者单独下载某个实验:
#   ./download_checkpoints.sh rnn_lstm_additive

# ========== 配置 ==========
# 修改这些变量以匹配你的服务器配置
SERVER_USER="root"                    # 或者你的用户名
SERVER_HOST="118.145.32.129"
SERVER_PORT="22"                      # SSH端口，如果是22可以省略 -P 参数
REMOTE_PATH="/data/250010055/250010055/nmt_project/checkpoints"
LOCAL_PATH="./checkpoints"

# 要下载的实验（按重要性排序）
# 如果只想下载几个代表性实验，可以只保留前几个
EXPERIMENTS=(
    "rnn_lstm_additive"           # 最佳RNN模型 BLEU=4.26
    "transformer_pos_learned"     # 最佳Transformer BLEU=3.83
    "transformer_scale_small"     # 最高效模型 BLEU=3.81
    "rnn_gru_multiplicative"      # GRU对比
    "rnn_gru_dot"
    "rnn_gru_additive"
    "rnn_gru_tf100"
    "rnn_gru_tf50"
    "rnn_gru_tf0"
    "transformer_pos_rope"
    "transformer_pos_sinusoidal"
    "transformer_norm_layernorm"
    "transformer_norm_rmsnorm"
    "transformer_scale_base"
    "transformer_scale_large"
    "transformer_lr_0.001"
    "transformer_lr_0.0005"
    "transformer_lr_0.0001"
    "transformer_bs_32"
    "transformer_bs_64"
    "transformer_bs_128"
)

# 需要下载的文件
FILES_TO_DOWNLOAD=(
    "best_model.pt"
    "src_vocab.json"
    "tgt_vocab.json"
)

# ========== 函数 ==========
download_experiment() {
    local exp_name=$1
    local remote_dir="${REMOTE_PATH}/${exp_name}"
    local local_dir="${LOCAL_PATH}/${exp_name}"
    
    echo "----------------------------------------"
    echo "Downloading: ${exp_name}"
    
    # 创建本地目录
    mkdir -p "${local_dir}"
    
    # 下载每个文件
    for file in "${FILES_TO_DOWNLOAD[@]}"; do
        local remote_file="${remote_dir}/${file}"
        local local_file="${local_dir}/${file}"
        
        if [ -f "${local_file}" ]; then
            echo "  ✓ ${file} (already exists)"
        else
            echo "  ↓ Downloading ${file}..."
            scp -P ${SERVER_PORT} "${SERVER_USER}@${SERVER_HOST}:${remote_file}" "${local_file}" 2>/dev/null
            
            if [ $? -eq 0 ]; then
                echo "  ✓ ${file} downloaded"
            else
                echo "  ✗ ${file} failed"
            fi
        fi
    done
}

list_remote_experiments() {
    echo "Listing experiments on remote server..."
    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "ls -la ${REMOTE_PATH}"
}

# ========== 主程序 ==========
echo "======================================"
echo "NMT Checkpoint Download Script"
echo "======================================"
echo "Server: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}"
echo "Remote: ${REMOTE_PATH}"
echo "Local:  ${LOCAL_PATH}"
echo ""

# 如果指定了实验名称，只下载该实验
if [ ! -z "$1" ]; then
    if [ "$1" == "--list" ]; then
        list_remote_experiments
        exit 0
    fi
    
    download_experiment "$1"
    echo ""
    echo "Done! You can now run:"
    echo "  python demo.py --exp $1"
    exit 0
fi

# 否则，询问下载哪些实验
echo "Available experiments to download:"
echo ""
echo "  1) Essential only (rnn_lstm_additive, transformer_pos_learned) ~200MB"
echo "  2) Top 5 experiments ~500MB"
echo "  3) All experiments ~1-2GB"
echo "  4) Custom selection"
echo ""
read -p "Select option [1-4]: " choice

case $choice in
    1)
        EXPERIMENTS=("rnn_lstm_additive" "transformer_pos_learned")
        ;;
    2)
        EXPERIMENTS=("rnn_lstm_additive" "transformer_pos_learned" "transformer_scale_small" "rnn_gru_multiplicative" "rnn_gru_dot")
        ;;
    3)
        # 使用全部
        ;;
    4)
        echo "Enter experiment names (space-separated):"
        read -a EXPERIMENTS
        ;;
    *)
        echo "Invalid option. Using essential only."
        EXPERIMENTS=("rnn_lstm_additive" "transformer_pos_learned")
        ;;
esac

echo ""
echo "Will download ${#EXPERIMENTS[@]} experiments:"
for exp in "${EXPERIMENTS[@]}"; do
    echo "  - ${exp}"
done
echo ""
read -p "Continue? [y/N]: " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

# 下载所有选中的实验
for exp in "${EXPERIMENTS[@]}"; do
    download_experiment "${exp}"
done

echo ""
echo "======================================"
echo "Download complete!"
echo "======================================"
echo ""
echo "Run demo:"
echo "  python demo.py --status    # 查看状态"
echo "  python demo.py             # 交互式翻译"
echo "  python demo.py --eval      # 评估测试集"
