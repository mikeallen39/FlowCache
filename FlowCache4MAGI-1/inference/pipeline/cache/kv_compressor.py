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
KV Cache Compression module.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from .base import KVCompressor
from .utils import (
    identify_compressible_chunks,
    check_compress_condition,
    get_latent_spatial_dims,
)


class KVCacheCompressor(KVCompressor):
    """
    Manages KV cache compression for memory-efficient inference.

    This compressor identifies clean chunks (completed denoising) and compresses
    their KV caches using the configured compression strategy (e.g., R1KV).

    Attributes:
        total_cache_len: Total cache capacity in tokens
        tokens_per_chunk: Number of tokens per chunk
        budget_cache_len: Target cache size after compression
        compression_config: Configuration for compression strategy
        kv_compressed: Whether compression has been performed
        chunk_query_states: Query states for each layer (used for compression)
    """

    def __init__(
        self,
        total_cache_len: int,
        tokens_per_chunk: int,
        budget_chunk_nums: int,
        window_size: int = 4,
        compression_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the KV cache compressor.

        Args:
            total_cache_len: Total cache capacity in tokens
            tokens_per_chunk: Number of tokens per chunk
            budget_chunk_nums: Target number of chunks after compression
            window_size: Window size for denoising stages
            compression_config: Configuration for compression strategy
        """
        super().__init__(enabled=True)
        self.total_cache_len = total_cache_len
        self.tokens_per_chunk = tokens_per_chunk
        self.budget_cache_len = (budget_chunk_nums - 1) * tokens_per_chunk
        self.window_size = window_size
        self.compression_config = compression_config or {}

        self.kv_compressed = False
        self.chunk_query_states: Dict[int, torch.Tensor] = {}

    def reset(self):
        """Reset compression state."""
        self.kv_compressed = False
        self.chunk_query_states.clear()

    def should_compress(
        self,
        tracker,
        chunk_num: int,
        chunk_start: int,
        transport_input,
        chunk_denoise_count: Dict[int, int],
        **kwargs
    ) -> bool:
        """
        Check if compression should be triggered.

        Args:
            tracker: ChunkKVRangeTracker instance
            chunk_num: Total number of chunks
            chunk_start: Current chunk being processed
            transport_input: Transport input
            chunk_denoise_count: Denoising steps per chunk

        Returns:
            True if compression should be performed
        """
        return check_compress_condition(
            tracker=tracker,
            total_cache_len=self.total_cache_len,
            chunk_num=chunk_num,
            chunk_start=chunk_start,
            transport_input=transport_input,
            chunk_denoise_count=chunk_denoise_count,
            window_size=self.window_size
        )

    def compress(
        self,
        model,
        inference_params,
        tracker,
        transport_input,
        chunk_start: int,
        chunk_denoise_count: Dict[int, int],
        query_states_dict: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[int, Tuple[int, int]]:
        """
        Perform KV cache compression.

        Args:
            model: DiT model with videodit_blocks
            inference_params: Inference parameters containing KV cache
            tracker: ChunkKVRangeTracker instance
            transport_input: Transport input
            chunk_start: Current chunk being processed
            chunk_denoise_count: Denoising steps per chunk

        Returns:
            Dictionary mapping chunk_id to (start, end) ranges after compression
        """
        # Identify chunks to compress
        chunk_offset = self._get_chunk_offset(transport_input)
        clean_chunk_ids, active_chunk_ids = identify_compressible_chunks(
            tracker=tracker,
            chunk_start=chunk_start,
            transport_input=transport_input,
            chunk_denoise_count=chunk_denoise_count,
            chunk_offset=chunk_offset
        )

        if len(clean_chunk_ids) < 2:
            # Need at least 2 chunks to compress
            return {}

        # Compress for each layer
        final_chunk_ids = []
        final_lengths = []

        for layer in model.videodit_blocks.layers:
            if not hasattr(layer.self_attention, 'kv_cluster'):
                continue
            
            # import pdb; pdb.set_trace()
            layer_result = self._compress_layer(
                layer=layer,
                inference_params=inference_params,
                tracker=tracker,
                clean_chunk_ids=clean_chunk_ids,
                active_chunk_ids=active_chunk_ids,
                transport_input=transport_input,
                query_states_dict=query_states_dict
            )

            # Store result from first layer for chunk metadata
            if layer.self_attention.layer_number == 0:
                final_chunk_ids = layer_result['chunk_ids']
                final_lengths = layer_result['lengths']

        # Update tracker ranges (shared across layers)
        new_ranges = self._compute_new_ranges(
            final_chunk_ids, final_lengths
        )
        tracker.update_ranges_after_compression(new_ranges)

        # Mark as compressed
        self.kv_compressed = True

        return new_ranges

    def _compress_layer(
        self,
        layer,
        inference_params,
        tracker,
        clean_chunk_ids: List[int],
        active_chunk_ids: List[int],
        transport_input,
        query_states_dict: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Compress KV cache for a single layer.

        Args:
            layer: Transformer layer
            inference_params: Inference parameters
            tracker: ChunkKVRangeTracker
            clean_chunk_ids: Chunks to compress
            active_chunk_ids: Chunks to keep uncompressed
            transport_input: Transport input
            query_states_dict: Query states for each layer (from transport)

        Returns:
            Dictionary with compression results
        """
        kv_cluster = layer.self_attention.kv_cluster
        layer_num = layer.self_attention.layer_number

        # Extract KV caches for clean chunks
        clean_kv_list = []
        clean_lengths = []
        for cid in clean_chunk_ids:
            s, e = tracker.get_range(cid)
            chunk_kv = inference_params.key_value_memory_dict[layer_num][s:e, ...]
            clean_kv_list.append(chunk_kv)
            clean_lengths.append(e - s)

        # Concatenate and split into key and value
        clean_kv = torch.cat(clean_kv_list, dim=0)
        key_clean, value_clean = torch.chunk(clean_kv, 2, dim=-1)

        # Extract KV caches for active chunks
        active_kv_list = []
        active_lengths = []
        for cid in active_chunk_ids:
            s, e = tracker.get_range(cid)
            chunk_kv = inference_params.key_value_memory_dict[layer_num][s:e, ...]
            active_kv_list.append(chunk_kv)
            active_lengths.append(e - s)

        # Get query states for compression
        query_states = query_states_dict.get(layer_num) if query_states_dict else None
        if query_states is None:
            raise RuntimeError(f"Query states not available for layer {layer_num}")

        # Set compression budget
        total_clean_tokens = sum(clean_lengths)
        kv_cluster.budget = max(
            total_clean_tokens - self.tokens_per_chunk,
            self.tokens_per_chunk
        )

        # Get latent dimensions
        H, W = get_latent_spatial_dims(transport_input, layer.model_config)
        T = self.tokens_per_chunk // (H * W)

        # Perform compression
        key_compressed, value_compressed, indices = kv_cluster.update_kv(
            key_states=key_clean,
            query_states=query_states,
            value_states=value_clean,
            clean_chunk_tokens=total_clean_tokens,
            latent_size_t=T,
            latent_size_h=H,
            latent_size_w=W,
        )

        # Reassemble KV cache
        final_kv_parts = []
        final_chunk_ids = []
        final_lengths = []

        # Add compressed part
        compressed_kv = torch.cat([key_compressed, value_compressed], dim=-1)
        final_kv_parts.append(compressed_kv)

        # Compute compressed lengths per chunk
        all_lengths_after_compress = self._compute_compressed_lengths(
            indices, clean_chunk_ids, clean_lengths, total_clean_tokens
        )
        final_chunk_ids.extend(clean_chunk_ids)
        final_lengths.extend(all_lengths_after_compress)

        # Add active (uncompressed) chunks
        for i, chunk_kv in enumerate(active_kv_list):
            final_kv_parts.append(chunk_kv)
            final_chunk_ids.append(active_chunk_ids[i])
            final_lengths.append(active_lengths[i])

        # Concatenate and update KV cache
        final_kv = torch.cat(final_kv_parts, dim=0)
        total_kv_len = final_kv.size(0)

        inference_params.key_value_memory_dict[layer_num][:total_kv_len, ...] = final_kv
        inference_params.key_value_memory_dict[layer_num][total_kv_len:, ...] = 0.0

        return {
            'chunk_ids': final_chunk_ids,
            'lengths': final_lengths
        }

    def _compute_compressed_lengths(
        self,
        indices: torch.Tensor,
        clean_chunk_ids: List[int],
        clean_lengths: List[int],
        total_clean_tokens: int
    ) -> List[int]:
        """
        Compute the compressed length for each chunk.

        Args:
            indices: Selected token indices [num_to_keep, num_kv_heads, head_dim]
            clean_chunk_ids: IDs of chunks that were compressed
            clean_lengths: Original lengths of compressed chunks
            total_clean_tokens: Total tokens before compression

        Returns:
            List of compressed lengths per chunk
        """
        # TODO: This has an issue - different heads keep different ranges
        # But it's fine since we attend to all previous chunks' KV cache
        indices_1d = indices[:, 0, 0]  # shape: (num_to_keep,)

        all_lengths_after_compress = []
        start_idx = 0

        for chunk_len in clean_lengths:
            end_idx = start_idx + chunk_len
            # Count selected tokens in this chunk's range
            mask = (indices_1d >= start_idx) & (indices_1d < min(end_idx, total_clean_tokens))
            kept_in_chunk = mask.sum().item()
            all_lengths_after_compress.append(kept_in_chunk)
            start_idx = end_idx

        return all_lengths_after_compress

    def _compute_new_ranges(
        self,
        chunk_ids: List[int],
        lengths: List[int]
    ) -> Dict[int, Tuple[int, int]]:
        """
        Compute new chunk ranges after compression.

        Args:
            chunk_ids: List of chunk IDs in order
            lengths: Compressed lengths for each chunk

        Returns:
            Dictionary mapping chunk_id to (start, end) range
        """
        new_ranges = {}
        current_start = 0

        for cid, length in zip(chunk_ids, lengths):
            new_end = current_start + length
            new_ranges[cid] = (current_start, new_end)
            current_start = new_end

        return new_ranges

    def _get_chunk_offset(self, transport_input) -> int:
        """
        Get the number of prefix video chunks.

        Args:
            transport_input: Transport input

        Returns:
            Number of prefix video chunks
        """
        if transport_input.prefix_video is not None:
            return transport_input.prefix_video.size(2) // transport_input.chunk_width
        return 0

    def store_query_states(self, layer_num: int, query_states: torch.Tensor):
        """
        Store query states for later compression.

        Args:
            layer_num: Layer number
            query_states: Query tensor to store
        """
        self.chunk_query_states[layer_num] = query_states

    def get_query_states(self, layer_num: int) -> Optional[torch.Tensor]:
        """
        Get stored query states for a layer.

        Args:
            layer_num: Layer number

        Returns:
            Query tensor or None if not available
        """
        return self.chunk_query_states.get(layer_num)
