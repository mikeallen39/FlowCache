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
Utility functions for cache management.
"""

import math
import torch
from typing import Dict, List, Tuple, Optional, Any
from inference.common import PackedCrossAttnParams


def generate_dynamic_kv_range(
    tracker,
    current_chunk_id: int,
    x_chunks_keys: List[int],
    chunk_token_nums: int,
    near_clean_chunk_idx: int = -1
) -> torch.Tensor:
    """
    Generate dynamic KV ranges for chunks after compression.

    This function computes the KV range each chunk should attend to,
    taking into account the compressed KV cache layout.

    Args:
        tracker: ChunkKVRangeTracker instance managing chunk ranges
        current_chunk_id: The chunk being processed
        x_chunks_keys: List of all chunk keys being processed
        chunk_token_nums: Number of tokens per chunk
        near_clean_chunk_idx: Index of the nearly-clean chunk (-1 if not present)

    Returns:
        Tensor of shape [num_chunks, 2] with KV ranges for each chunk
    """
    kv_ranges = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process normal chunks (excluding near_clean_chunk)
    normal_chunks = [chunk_id for chunk_id in x_chunks_keys if chunk_id != near_clean_chunk_idx]

    for chunk_id in normal_chunks:
        # Normal chunk: needs to see itself and all previous chunks
        all_chunk_ids = tracker.get_all_chunk_ids() + list(normal_chunks)
        chunks_to_include = [cid for cid in all_chunk_ids if cid <= chunk_id]

        # Calculate based on actual compressed ranges in tracker
        total_tokens = 0
        for cid in chunks_to_include:
            if cid in tracker.get_all_chunk_ids():
                # Use compressed actual range
                s, e = tracker.get_range(cid)
                total_tokens = max(total_tokens, e)
            else:
                # Newly entered chunk not yet registered, but size is known
                total_tokens += chunk_token_nums

        range_start = 0
        range_end = total_tokens
        kv_ranges.append([range_start, range_end])

    # Handle near_clean_chunk (always last if present)
    if near_clean_chunk_idx != -1:
        # Calculate end position of last normal chunk
        last_normal_chunk_end = 0
        all_chunk_ids = tracker.get_all_chunk_ids() + normal_chunks
        for cid in all_chunk_ids:
            if cid in tracker.get_all_chunk_ids():
                s, e = tracker.get_range(cid)
                last_normal_chunk_end = max(last_normal_chunk_end, e)
            else:
                # Newly entered chunk not yet registered
                last_normal_chunk_end += chunk_token_nums

        # near_clean_chunk range: (last_normal_chunk_end, last_normal_chunk_end + chunk_token_nums]
        range_start = last_normal_chunk_end
        range_end = last_normal_chunk_end + chunk_token_nums
        kv_ranges.append([range_start, range_end])

    return torch.tensor(kv_ranges, device=device, dtype=torch.int32)


def identify_compressible_chunks(
    tracker,
    chunk_start: int,
    transport_input,
    chunk_denoise_count: Dict[int, int],
    chunk_offset: int = 0
) -> Tuple[List[int], List[int]]:
    """
    Identify which chunks can be compressed and which should remain active.

    A chunk can be compressed if:
    - It's a prefix video chunk (always clean)
    - It's a generated chunk that has completed all denoising steps

    Args:
        tracker: ChunkKVRangeTracker instance
        chunk_start: Current chunk being processed
        transport_input: Transport input containing chunk info
        chunk_denoise_count: Dictionary mapping chunk_id to denoising steps completed
        chunk_offset: Number of prefix video chunks

    Returns:
        Tuple of (clean_chunk_ids, active_chunk_ids)
    """
    all_chunk_ids = tracker.get_all_chunk_ids()

    clean_chunks = []
    for cid in all_chunk_ids:
        if cid < chunk_offset:
            # Prefix video chunks are always clean
            clean_chunks.append(cid)
        elif cid <= chunk_start:
            # Generated chunks need to check denoising completion
            if chunk_denoise_count[cid] == transport_input.num_steps:
                clean_chunks.append(cid)

    active_chunks = [cid for cid in all_chunk_ids if cid not in clean_chunks]

    return clean_chunks, active_chunks


def check_compress_condition(
    tracker,
    total_cache_len: int,
    chunk_num: int,
    chunk_start: int,
    transport_input,
    chunk_denoise_count: Dict[int, int],
    window_size: int = 4
) -> bool:
    """
    Check if KV cache compression should be triggered.

    Compression is triggered when:
    1. Cache is full (next_free_idx >= total_cache_len)
    2. More chunks are yet to enter (registered_count < chunk_num)
    3. Next chunk is about to enter (last chunk's steps == num_steps/window_size)

    Args:
        tracker: ChunkKVRangeTracker instance
        total_cache_len: Total cache capacity in tokens
        chunk_num: Total number of chunks
        chunk_start: Current chunk being processed
        transport_input: Transport input containing parameters
        chunk_denoise_count: Dictionary mapping chunk_id to denoising steps
        window_size: Window size for denoising stages (default: 4)

    Returns:
        True if compression should be performed, False otherwise
    """
    all_chunk_ids = tracker.get_all_chunk_ids()
    if len(all_chunk_ids) == 0:
        return False

    registered_chunk_count = len(all_chunk_ids)
    cache_full = tracker.next_free_idx >= total_cache_len
    has_more_chunks = registered_chunk_count < chunk_num
    last_chunk_id = all_chunk_ids[-1]

    # Calculate steps per stage
    steps_per_stage = transport_input.num_steps // window_size
    next_chunk_will_enter = chunk_denoise_count[last_chunk_id] == steps_per_stage

    should_compress = cache_full and has_more_chunks and next_chunk_will_enter
    return should_compress


def get_embedding_and_meta_with_chunk_info(
    model_self,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    caption_dropout_mask,
    xattn_mask,
    kv_range: torch.Tensor,
    **kwargs
) -> tuple:
    """
    Compute embeddings and meta information with chunk-aware processing.

    This is a unified version of the get_embedding_and_meta function that
    properly handles chunk-based processing with dynamic KV ranges.

    Args:
        model_self: The DiT model instance
        x: Input tensor [N, C, T, H, W]
        t: Timestep tensor [N, range_num]
        y: Text conditioning tensor
        caption_dropout_mask: Dropout mask for captions
        xattn_mask: Cross-attention mask
        kv_range: KV range tensor
        **kwargs: Additional arguments including:
            - range_num: Total number of chunks
            - denoising_range_num: Number of chunks being denoised
            - slice_point: Starting chunk index
            - start_chunk_id: First chunk to process
            - end_chunk_id: Last chunk to process (exclusive)
            - distill_nearly_clean_chunk: Whether to add nearly-clean chunk
            - chunk_token_nums: Tokens per chunk
            - chunk_width: Width of each chunk in frames
            - num_steps: Total denoising steps

    Returns:
        Tuple of (x, condition, condition_map, rope, y_xattn_flat, xattn_mask_cuda,
                 H, W, ardf_meta, cross_attn_params)
    """
    # ========== Part 1: Embed x ==========
    x = model_self.x_embedder(x)  # [N, C, T, H, W]
    batch_size, _, T, H, W = x.shape

    # Prepare necessary variables
    range_num = kwargs["range_num"]
    denoising_range_num = kwargs["denoising_range_num"]
    slice_point = kwargs.get("slice_point", 0)
    frame_in_range = T // denoising_range_num

    # distill_nearly_clean_chunk adds one extra chunk
    T_total = (range_num + kwargs.get("distill_nearly_clean_chunk", False)) * frame_in_range

    # ========== Part 2: Compute rotary positional embedding ==========
    rescale_factor = math.sqrt((H * W) / (16 * 16))
    rope = model_self.rope.get_embed(
        shape=[T_total, H, W],
        ref_feat_shape=[T_total, H / rescale_factor, W / rescale_factor]
    )
    # Rope shape: (T*H*W, head_dim) - cut to current chunk range
    rope = rope[
        kwargs["start_chunk_id"] * frame_in_range * H * W :
        kwargs["end_chunk_id"] * frame_in_range * H * W
    ]

    # ========== Part 3: Embed t ==========
    assert t.shape[0] == batch_size, f"Invalid t shape: {t.shape[0]} != {batch_size}"
    assert t.shape[1] == denoising_range_num, f"Invalid t shape: {t.shape[1]} != {denoising_range_num}"

    t_flat = t.flatten()  # (N * denoising_range_num,)
    t = model_self.t_embedder(t_flat)  # (N, D)

    if model_self.engine_config.distill:
        distill_dt_scalar = 2
        if kwargs["num_steps"] == 12:
            base_chunk_step = 4
            distill_dt_factor = base_chunk_step / kwargs["distill_interval"] * distill_dt_scalar
        else:
            distill_dt_factor = kwargs["num_steps"] / 4 * distill_dt_scalar

        distill_dt = torch.ones_like(t_flat) * distill_dt_factor
        distill_dt_embed = model_self.t_embedder(distill_dt)
        t = t + distill_dt_embed

    t = t.reshape(batch_size, denoising_range_num, -1)  # (N, range_num, D)

    # ========== Part 4: Embed y, prepare condition and y_xattn_flat ==========
    y_xattn, y_adaln = model_self.y_embedder(y, model_self.training, caption_dropout_mask)

    assert xattn_mask is not None
    xattn_mask = xattn_mask.squeeze(1).squeeze(1)

    # condition: (N, range_num, D)
    y_adaln = y_adaln.squeeze(1)  # (N, D)
    condition = t + y_adaln.unsqueeze(1)

    assert condition.shape[0] == batch_size
    assert condition.shape[1] == denoising_range_num

    seqlen_per_chunk = (T * H * W) // denoising_range_num
    condition_map = torch.arange(batch_size * denoising_range_num, device=x.device)
    condition_map = torch.repeat_interleave(condition_map, seqlen_per_chunk)
    condition_map = condition_map.reshape(batch_size, -1).transpose(0, 1).contiguous()

    # y_xattn_flat: (total_token, D)
    y_xattn_flat = torch.masked_select(
        y_xattn.squeeze(1),
        xattn_mask.unsqueeze(-1).bool()
    ).reshape(-1, y_xattn.shape[-1])

    xattn_mask_for_cuda_graph = None

    # ========== Part 5: Prepare cross_attn_params ==========
    xattn_mask = xattn_mask.reshape(xattn_mask.shape[0], -1)
    y_index = torch.sum(xattn_mask, dim=-1)
    clip_token_nums = H * W * frame_in_range

    cu_seqlens_q = torch.Tensor(
        [0] + ([clip_token_nums] * denoising_range_num * batch_size)
    ).to(torch.int64).to(x.device)
    cu_seqlens_k = torch.cat(
        [y_index.new_tensor([0]), y_index]
    ).to(torch.int64).to(x.device)
    cu_seqlens_q = cu_seqlens_q.cumsum(-1).to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.cumsum(-1).to(torch.int32)

    assert cu_seqlens_q.shape == cu_seqlens_k.shape, \
        f"cu_seqlens_q.shape: {cu_seqlens_q.shape}, cu_seqlens_k.shape: {cu_seqlens_k.shape}"

    xattn_q_ranges = torch.cat(
        [cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1
    )
    xattn_k_ranges = torch.cat(
        [cu_seqlens_k[:-1].unsqueeze(1), cu_seqlens_k[1:].unsqueeze(1)], dim=1
    )
    assert xattn_q_ranges.shape == xattn_k_ranges.shape, \
        f"xattn_q_ranges.shape: {xattn_q_ranges.shape}, xattn_k_ranges.shape: {xattn_k_ranges.shape}"

    cross_attn_params = PackedCrossAttnParams(
        q_ranges=xattn_q_ranges,
        kv_ranges=xattn_k_ranges,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=clip_token_nums,
        max_seqlen_kv=model_self.caption_max_length,
    )

    # ========== Part 6: Prepare core_attn related q/kv range ==========
    q_range = torch.cat(
        [cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1
    )
    flat_kv = torch.unique(kv_range, sorted=True)
    max_seqlen_k = (flat_kv[-1] - flat_kv[0]).cpu().item()

    ardf_meta = dict(
        clip_token_nums=clip_token_nums,
        slice_point=slice_point,
        range_num=range_num,
        denoising_range_num=denoising_range_num,
        q_range=q_range,
        k_range=kv_range,
        max_seqlen_q=clip_token_nums,
        max_seqlen_k=max_seqlen_k,
    )

    return (x, condition, condition_map, rope, y_xattn_flat,
            xattn_mask_for_cuda_graph, H, W, ardf_meta, cross_attn_params)


def compute_chunk_token_nums(
    transport_input,
    model_config,
    chunk_width: int
) -> int:
    """
    Calculate the number of tokens in one chunk.

    Args:
        transport_input: Transport input containing latent dimensions
        model_config: Model configuration
        chunk_width: Number of frames per chunk

    Returns:
        Number of tokens per chunk
    """
    patch_size = model_config.patch_size
    latent_h = transport_input.latent_size[3] // patch_size
    latent_w = transport_input.latent_size[4] // patch_size

    return chunk_width * latent_h * latent_w


def get_latent_spatial_dims(
    transport_input,
    model_config
) -> Tuple[int, int]:
    """
    Get the spatial dimensions of latent in patch units.

    Args:
        transport_input: Transport input containing latent dimensions
        model_config: Model configuration

    Returns:
        Tuple of (height_patches, width_patches)
    """
    patch_size = model_config.patch_size
    h = transport_input.latent_size[3] // patch_size
    w = transport_input.latent_size[4] // patch_size
    return h, w
