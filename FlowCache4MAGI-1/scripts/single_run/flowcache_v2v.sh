# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export MASTER_ADDR=localhost
export MASTER_PORT=6001
export GPUS_PER_NODE=1
export NNODES=1
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=7

export PAD_HQ=1
export PAD_DURATION=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"

MAGI_ROOT=$(git rev-parse --show-toplevel)


OUTPUT_NAME=flowcache
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
EXP_DIR="/path/to/output/magi/${TIMESTAMP}_${OUTPUT_NAME}"
mkdir -p "$EXP_DIR"

LOG_FILE="$EXP_DIR/log_${TIMESTAMP}.log"
OUTPUT_PATH="$EXP_DIR/output.mp4"

export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
python3 inference/pipeline/flowcache.py \
    --config_file config/single_run/flowcache_v2v.json \
    --mode v2v \
    --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
    --prefix_video_path "/path/to/physicsiq/conditioning_video.mp4" \
    --output_path $OUTPUT_PATH \
    --additional_config addconfig/config.yaml \
    2>&1 | tee $LOG_FILE

# a cat sitting on the grass
