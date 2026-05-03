#!/usr/bin/env bash
# setup_gpu_instance.sh — one-command environment setup for Lambda Labs or RunPod
# Works on Ubuntu 22.04 + CUDA 12.x instances.
# Usage: bash <(curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/learned-compiler-memory/main/scripts/setup_gpu_instance.sh)
# Or:    bash scripts/setup_gpu_instance.sh

set -euo pipefail

# ── Detect workspace root (RunPod uses /workspace, Lambda uses /home/ubuntu) ──
if [[ -d /workspace ]]; then
    ROOT=/workspace/mcm
else
    ROOT="$HOME/mcm"
fi
echo "→ Using root: $ROOT"
mkdir -p "$ROOT"/{data/synthetic,checkpoints,scripts,src}

# ── System packages ──
echo "→ Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop git rsync curl wget unzip

# ── Python deps ──
echo "→ Installing Python packages..."
pip install -q --upgrade \
    transformers \
    peft \
    trl \
    bitsandbytes \
    datasets \
    accelerate \
    sentencepiece \
    huggingface_hub

# ── Clone repo (edit URL if private) ──
# Uncomment and set your repo URL:
# git clone https://github.com/YOUR_ORG/learned-compiler-memory "$ROOT/repo"
# cp "$ROOT/repo/scripts/train_mcm.py" "$ROOT/scripts/"

# ── Print installed versions ──
echo ""
echo "=== Installed versions ==="
python -c "
import torch, transformers, peft, trl, bitsandbytes, accelerate
print(f'  torch:          {torch.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  trl:            {trl.__version__}')
print(f'  bitsandbytes:   {bitsandbytes.__version__}')
print(f'  accelerate:     {accelerate.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM:           {total:.1f} GB')
    print(f'  bf16 support:   {torch.cuda.is_bf16_supported()}')
"

echo ""
echo "=== Setup complete ==="
echo "  Workspace:  $ROOT"
echo ""
echo "Next steps:"
echo "  1. Upload data:      rsync -avz data/synthetic/ \$(hostname -I | awk '{print \$1}'):$ROOT/data/synthetic/"
echo "  2. Copy train script: cp scripts/train_mcm.py $ROOT/scripts/"
echo "  3. Start training:    tmux new -s train"
echo "                        python $ROOT/scripts/train_mcm.py \\"
echo "                            --base_model Qwen/Qwen2.5-3B-Instruct \\"
echo "                            --data_dir $ROOT/data/synthetic \\"
echo "                            --output_dir $ROOT/checkpoints/mcm-write-v1 \\"
echo "                            --epochs 3 \\"
echo "                            --per_device_batch_size 4 \\"
echo "                            --grad_accumulation_steps 8 \\"
echo "                            --max_seq_length 2048 \\"
echo "                            --save_steps 300"
echo ""
echo "  Detach tmux: Ctrl+B, D   |   Re-attach: tmux attach -t train"
