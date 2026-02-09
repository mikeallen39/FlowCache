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
Cache reuse implementations for output optimization.

This module provides two caching strategies:
- TeaCache: Full output reuse (all chunks together)
- ChunkWiseCache: Per-chunk output reuse (for FlowCache)
"""

from einops import rearrange
import torch
from typing import Dict, List, Optional, Tuple
from .base import OutputCache


class TeaCache(OutputCache):
    """
    TeaCache implementation with full output reuse.

    This cache computes the relative L1 distance between current and previous
    modulated inputs. When the accumulated distance is below threshold, the
    output is reused and only the residual is applied.

    All chunks are treated as a single unit for reuse decisions.

    Attributes:
        rel_l1_thresh: Threshold for relative L1 distance
        warmup_steps: Number of warmup steps before reuse can happen
        log: Whether to log reuse decisions
        accumulated_rel_l1_distance: Accumulated relative L1 distance
        previous_modulated_input: Previous input features
        previous_residual: Previous residual for reuse
        reuse_times: Number of times output was reused
        previous_output: Output from previous stage
        cnt: Current step counter
        num_steps: Total number of steps
    """

    def __init__(
        self,
        rel_l1_thresh: float = 0.01,
        warmup_steps: int = 0,
        log: bool = False
    ):
        super().__init__(enabled=True)
        self.rel_l1_thresh = rel_l1_thresh
        self.warmup_steps = warmup_steps
        self.log = log

        # State variables
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.reuse_times = 0
        self.previous_output = None
        self.cnt = 0
        self.num_steps = 0
        self.should_calc = True

    def reset(self):
        """Reset all cache state."""
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.reuse_times = 0
        self.previous_output = None
        self.cnt = 0
        self.should_calc = True

    def compute_feature_metric(
        self,
        x: torch.Tensor,
        x_embedder,
        x_rescale_factor: float,
        half_channel_vae: bool,
        params_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Compute feature metric from input tensor.

        Args:
            x: Input tensor [N, C, T, H, W]
            x_embedder: Model's x_embedder module
            x_rescale_factor: Rescale factor for x
            half_channel_vae: Whether VAE uses half channels
            params_dtype: Model's parameter dtype for final conversion

        Returns:
            Feature tensor of shape [(T*H*W), N, C]
        """
        metric_x = x.clone()
        metric_x = metric_x * x_rescale_factor

        if half_channel_vae:
            assert metric_x.shape[1] == 16, "Expected 16 channels for half-channel VAE"
            metric_x = torch.cat([metric_x, metric_x], dim=1)

        metric_x = metric_x.float()
        metric_x = x_embedder(metric_x)
        metric_x = metric_x.to(params_dtype)
        metric_x = rearrange(metric_x, "N C T H W -> (T H W) N C").contiguous()

        return metric_x

    def should_reuse(
        self,
        chunk_id: int,
        step: int,
        current_features: torch.Tensor,
        denoise_step_per_stage: int,
        num_chunks_current: int,
        num_chunks_previous: int,
        **kwargs
    ) -> bool:
        """
        Determine whether to reuse output based on feature similarity.

        Args:
            chunk_id: Current chunk ID (not used in simple mode)
            step: Current denoising step
            current_features: Current input features
            denoise_step_per_stage: Steps per denoising stage
            num_chunks_current: Number of chunks in current stage
            num_chunks_previous: Number of chunks in previous stage

        Returns:
            True if output should be reused, False if should calculate
        """
        # Always calculate first and last steps, and during warmup
        if self.cnt == 0 or self.cnt == self.num_steps - 1 or self.cnt < self.warmup_steps:
            self.should_calc = True
            self.accumulated_rel_l1_distance = 0
            if self.log:
                print(f"Calculate output at step {self.cnt}")
            return False

        # Compute feature difference
        a1 = current_features.clone()
        a2 = self.previous_modulated_input.clone()

        # Handle chunk count changes across stages
        if self.cnt % denoise_step_per_stage == 0:
            dim1 = a1.shape[0]
            dim2 = a2.shape[0]

            if dim1 > dim2:
                # Next stage has more chunks, truncate to match
                a1 = a1[:dim2]
            elif dim1 < dim2:
                # Next stage has fewer chunks, take tail part
                a2 = a2[-dim1:]

        # Compute relative L1 distance
        rel_l1 = ((a1 - a2).abs().mean() / a2.abs().mean()).cpu().item()
        self.accumulated_rel_l1_distance += rel_l1

        # Decide whether to reuse
        if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
            if self.cnt % denoise_step_per_stage == 0 and dim1 > dim2:
                # Only calculate new chunk when crossing stage
                self.should_calc = True
                if self.log:
                    print(f"Partly reuse output at step {self.cnt}, only calculate new chunk")
                return False
            else:
                # Full reuse
                self.reuse_times += 1
                if self.log:
                    print(f"Reuse output at step {self.cnt}")
                self.should_calc = False
                return True
        else:
            # Threshold exceeded, recalculate
            if self.log:
                print(f"Calculate output at step {self.cnt}")
            self.should_calc = True
            self.accumulated_rel_l1_distance = 0
            return False

    def update_residual(self, chunk_id: int, residual: torch.Tensor):
        """
        Update the residual for reuse.

        Args:
            chunk_id: Chunk ID (not used in simple mode, residual applies to all)
            residual: Residual tensor to store
        """
        self.previous_residual = residual

    def get_residual(self, chunk_id: int) -> Optional[torch.Tensor]:
        """
        Get the stored residual.

        Args:
            chunk_id: Chunk ID (not used in simple mode)

        Returns:
            The residual tensor or None
        """
        return self.previous_residual

    def increment_step(self):
        """Increment step counter and print statistics if done."""
        self.cnt += 1
        if self.cnt == self.num_steps:
            print(f"Reuse output account for {self.reuse_times} / {self.num_steps} steps, "
                  f"ratio: {self.reuse_times / self.num_steps:.2%}")
            self.cnt = 0

    def store_previous_features(self, features: torch.Tensor):
        """Store current features as previous for next step."""
        self.previous_modulated_input = features.clone()

    def get_previous_features(self) -> Optional[torch.Tensor]:
        """Get the stored previous features."""
        return self.previous_modulated_input

    def prepare_for_next_stage(self):
        """Store output for use in next stage."""
        pass  # Handled in integrate_velocity


class ChunkWiseCache(OutputCache):
    """
    Chunk-wise output cache implementation for FlowCache.

    This cache tracks reuse decisions separately for each chunk, allowing
    finer-grained control over which chunks to skip.

    Attributes:
        rel_l1_thresh: Threshold for relative L1 distance
        warmup_steps: Number of warmup steps per chunk before reuse can happen
        discard_nearly_clean_chunk: Whether to skip nearly-clean chunk
        log: Whether to log reuse decisions
        chunk_accumulated_rel_l1: Per-chunk accumulated L1 distance
        chunk_reuse_flags: Per-chunk reuse flags for current step
        prev_metric_chunks: Previous features per chunk
        previous_residual: Per-chunk residuals
    """

    def __init__(
        self,
        rel_l1_thresh: float = 0.01,
        warmup_steps: int = 0,
        discard_nearly_clean_chunk: bool = False,
        log: bool = False
    ):
        super().__init__(enabled=True)
        self.rel_l1_thresh = rel_l1_thresh
        self.warmup_steps = warmup_steps
        self.discard_nearly_clean_chunk = discard_nearly_clean_chunk
        self.log = log

        # State variables
        self.chunk_accumulated_rel_l1: Dict[int, float] = {}
        self.chunk_reuse_flags: Dict[int, bool] = {}
        self.prev_metric_chunks: Dict[int, torch.Tensor] = {}
        self.previous_residual: Dict[int, torch.Tensor] = {}

        self.cnt = 0
        self.num_steps = 0

    def reset(self):
        """Reset all cache state."""
        self.chunk_accumulated_rel_l1.clear()
        self.chunk_reuse_flags.clear()
        self.prev_metric_chunks.clear()
        self.previous_residual.clear()
        self.cnt = 0

    def initialize_chunk_state(self, chunk_num: int):
        """Initialize state for all chunks."""
        if len(self.chunk_accumulated_rel_l1) != chunk_num:
            self.chunk_accumulated_rel_l1 = {i: 0.0 for i in range(chunk_num)}
            self.previous_residual = {i: None for i in range(chunk_num)}

        # Reset reuse flags for each step
        self.chunk_reuse_flags = {i: False for i in range(chunk_num)}

    def compute_feature_metric(
        self,
        x: torch.Tensor,
        x_embedder,
        x_rescale_factor: float,
        half_channel_vae: bool,
        chunk_token_nums: int,
        params_dtype: torch.dtype,
        offset: int = 0,
        fwd_extra_1st_chunk: bool = False,
        distill_nearly_clean_chunk: bool = False
    ) -> Tuple[Dict[int, torch.Tensor], int]:
        """
        Compute feature metric for each chunk.

        Following source code logic:
        1. Compute metric_x from input x
        2. Handle fwd_extra_1st_chunk: slice off first chunk if needed
        3. Handle distill_nearly_clean_chunk: slice off last chunk if needed
        4. Split into chunks

        Args:
            x: Input tensor [N, C, T, H, W]
            x_embedder: Model's x_embedder module
            x_rescale_factor: Rescale factor for x
            half_channel_vae: Whether VAE uses half channels
            chunk_token_nums: Number of tokens per chunk
            params_dtype: Model's parameter dtype for final conversion
            offset: Offset for chunk_id (to match x_chunks indexing)
            fwd_extra_1st_chunk: Whether to slice off first chunk (always False)
            distill_nearly_clean_chunk: Whether to slice off last chunk

        Returns:
            Tuple of (metric_chunks dict, num_chunks_for_x)
        """
        from einops import rearrange

        # 1. Compute metric_x from input x
        metric_x = x.clone()
        metric_x = metric_x * x_rescale_factor

        if half_channel_vae:
            assert metric_x.shape[1] == 16
            metric_x = torch.cat([metric_x, metric_x], dim=1)

        metric_x = metric_x.float()
        metric_x = x_embedder(metric_x)
        metric_x = metric_x.to(params_dtype)
        metric_x = rearrange(metric_x, "N C T H W -> (T H W) N C").contiguous()

        # 2. Handle fwd_extra_1st_chunk: slice off first chunk if needed
        # Note: fwd_extra_1st_chunk is always False in current implementation
        if fwd_extra_1st_chunk:
            metric_x = metric_x[chunk_token_nums:, :, :]

        # 3. Handle distill_nearly_clean_chunk: slice off last chunk if needed
        if distill_nearly_clean_chunk:
            metric_x = metric_x[:-chunk_token_nums, :, :]

        # 4. Split into chunks
        assert metric_x.shape[0] % chunk_token_nums == 0
        num_chunks = metric_x.shape[0] // chunk_token_nums

        metric_chunks = {}
        for i in range(num_chunks):
            start = i * chunk_token_nums
            end = start + chunk_token_nums
            metric_chunks[offset + i] = metric_x[start:end]

        # Return num_chunks for x_chunks iteration (matching source code)
        return metric_chunks, num_chunks

    def should_reuse(
        self,
        chunk_id: int,
        step: int,
        current_features: torch.Tensor,
        chunk_denoise_count: Dict[int, int],
        current_num_chunks: int,
        previous_num_chunks: int,
        **kwargs
    ) -> bool:
        """
        Determine whether to reuse output for a specific chunk.

        Args:
            chunk_id: The chunk ID to check
            step: Current denoising step
            current_features: Current features for all chunks
            chunk_denoise_count: Denoising steps completed per chunk
            current_num_chunks: Number of chunks in current stage
            previous_num_chunks: Number of chunks in previous stage

        Returns:
            True if output should be reused, False otherwise
        """
        # First and last steps always calculate
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            return False

        # Check if chunk exists in both current and previous
        if chunk_id not in current_features or chunk_id not in self.prev_metric_chunks:
            return False

        # Apply warmup: skip reuse during warmup period
        if self._should_skip_reuse(chunk_id, chunk_denoise_count):
            self.chunk_accumulated_rel_l1[chunk_id] = 0.0
            return False

        # Compute relative L1 distance
        curr_feat = current_features[chunk_id]
        prev_feat = self.prev_metric_chunks[chunk_id]

        diff = (curr_feat - prev_feat).abs().mean()
        denom = prev_feat.abs().mean() + 1e-8
        rel_l1 = (diff / denom).item()

        # Accumulate and check threshold
        accumulated = self.chunk_accumulated_rel_l1[chunk_id] + rel_l1

        if accumulated < self.rel_l1_thresh:
            self.chunk_accumulated_rel_l1[chunk_id] = accumulated
            self.chunk_reuse_flags[chunk_id] = True
            return True
        else:
            self.chunk_accumulated_rel_l1[chunk_id] = 0.0
            self.chunk_reuse_flags[chunk_id] = False
            return False

    def _should_skip_reuse(
        self,
        chunk_id: int,
        chunk_denoise_count: Dict[int, int]
    ) -> bool:
        """
        Check if reuse should be skipped for this chunk.

        During warmup period, chunks are always recalculated.

        Args:
            chunk_id: Chunk to check
            chunk_denoise_count: Steps completed per chunk

        Returns:
            True if should skip reuse (i.e., in warmup period)
        """
        return chunk_denoise_count[chunk_id] < self.warmup_steps

    def update_residual(self, chunk_id: int, residual: torch.Tensor):
        """Update the residual for a specific chunk."""
        self.previous_residual[chunk_id] = residual

    def get_residual(self, chunk_id: int) -> Optional[torch.Tensor]:
        """Get the stored residual for a chunk."""
        return self.previous_residual.get(chunk_id)

    def store_previous_features(self, metric_chunks: Dict[int, torch.Tensor]):
        """Store current features as previous for next step."""
        self.prev_metric_chunks = {
            i: f.clone().detach() for i, f in metric_chunks.items()
        }

    def increment_step(self):
        """Increment step counter."""
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    def set_total_steps(self, num_steps: int):
        """Set total number of steps."""
        self.num_steps = num_steps
