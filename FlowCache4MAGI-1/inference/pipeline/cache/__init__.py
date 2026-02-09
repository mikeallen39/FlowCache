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

"""
Cache management module for MAGI inference.

This module provides optimized implementations for:
- TeaCache: Full output reuse (all chunks together)
- ChunkWiseCache: Per-chunk output reuse (used in FlowCache)
- KVCacheCompressor: Dynamic KV cache compression
"""

from .base import CacheStrategy, OutputCache, KVCompressor
from .cachereuse import TeaCache, ChunkWiseCache
from .kv_compressor import KVCacheCompressor
from .utils import generate_dynamic_kv_range

__all__ = [
    "CacheStrategy",
    "OutputCache",
    "KVCompressor",
    "TeaCache",
    "ChunkWiseCache",
    "generate_dynamic_kv_range",
]
