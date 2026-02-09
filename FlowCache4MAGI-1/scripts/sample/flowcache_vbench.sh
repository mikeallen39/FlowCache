#!/bin/bash

# FlowCache VBench sampling script
# Usage: bash flowcache_vbench.sh [yaml_config_path]
# Default config: yaml_config/sample/flowcache_vbench.yaml

export DEVICES="1,3,4,6"

export PAD_HQ=1
export PAD_DURATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"

MAGI_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
export MAGI_ROOT="$MAGI_ROOT"

# YAML config file path (can be overridden via command line argument)
YAML_CONFIG="${1:-yaml_config/sample/flowcache_vbench.yaml}"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "❌ YAML config file not found: $YAML_CONFIG"
    exit 1
fi

echo "📋 Using YAML config: $YAML_CONFIG"

# Create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flowcache_vbench_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🚀 Starting multi-GPU benchmark sampling"
echo "🎮 GPUs: $DEVICES"

# Define list of dimensions to process
DIMENSIONS=("overall_consistency" "subject_consistency" "scene")

echo "🔢 Total dimensions to process: ${#DIMENSIONS[@]}"
echo "📋 Dimensions: ${DIMENSIONS[*]}"

# Loop through each dimension
for DIMENSION in "${DIMENSIONS[@]}"; do
    echo "🔍 Processing dimension: $DIMENSION"

    # Use Python to temporarily modify the dimension in YAML, then run sampling
    python3 -c "
import yaml
import sys

# Read YAML config
with open('$YAML_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Modify dimension
config['dimension'] = '$DIMENSION'

# Save to temporary file
temp_config = '$YAML_CONFIG.tmp'
with open(temp_config, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(temp_config)
" > /tmp/temp_config_path.txt

    TEMP_CONFIG=$(cat /tmp/temp_config_path.txt)
    python sample_video.py "$TEMP_CONFIG"
    rm "$TEMP_CONFIG"

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $DIMENSION"
    else
        echo "❌ Failed: $DIMENSION"
        echo "🛑 Script paused due to error. Fix the issue and rerun."
        exit 1
    fi

    echo "---"
done

echo "🎉 All sampling tasks completed."
