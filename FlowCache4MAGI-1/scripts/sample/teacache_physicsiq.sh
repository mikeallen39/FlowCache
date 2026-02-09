#!/bin/bash

# TeaCache PhysicsIQ sampling script
# Usage: bash teacache_physicsiq.sh [yaml_config_path]
# Default config: yaml_config/sample/teacache_physicsiq.yaml

export DEVICES="0,1,2,3,4,5,6,7"

export PAD_HQ=1
export PAD_DURATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"

MAGI_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
export MAGI_ROOT="$MAGI_ROOT"

export XDG_CACHE_HOME="/path/to/tmp"
mkdir -p "$XDG_CACHE_HOME"

# YAML config file path (can be overridden via command line argument)
YAML_CONFIG="${1:-yaml_config/sample/teacache_physicsiq.yaml}"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "❌ YAML config file not found: $YAML_CONFIG"
    exit 1
fi

echo "📋 Using YAML config: $YAML_CONFIG"

# Create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/teacache_physicsiq_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🚀 Starting multi-GPU benchmark sampling"
echo "🎮 GPUs: $DEVICES"

# Run sampling
python sample_video.py "$YAML_CONFIG"

if [ $? -eq 0 ]; then
    echo "✅ Sampling completed successfully."
else
    echo "❌ Sampling failed. Check log: $LOG_FILE"
    exit 1
fi

echo "---"
echo "🎉 All sampling tasks completed."
