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
Base classes for cache management strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch


class CacheStrategy(ABC):
    """
    Abstract base class for cache management strategies.

    All cache implementations should inherit from this class and implement
    the required methods.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the cache strategy.

        Args:
            enabled: Whether this cache strategy is enabled
        """
        self.enabled = enabled

    @abstractmethod
    def reset(self):
        """
        Reset the cache state.

        This method should clear all internal state and prepare the cache
        for a new inference run.
        """
        pass

    def reset_if_enabled(self):
        """Reset the cache if it is enabled."""
        if self.enabled:
            self.reset()


class OutputCache(CacheStrategy):
    """
    Abstract base class for output reuse strategies.

    Output caching strategies determine when model outputs can be reused
    based on input similarity metrics.
    """

    @abstractmethod
    def should_reuse(
        self,
        chunk_id: int,
        step: int,
        current_features: torch.Tensor,
        **kwargs
    ) -> bool:
        """
        Determine whether the output for a chunk should be reused.

        Args:
            chunk_id: The ID of the current chunk
            step: The current denoising step
            current_features: Feature tensor for the current input
            **kwargs: Additional arguments specific to the implementation

        Returns:
            True if the output should be reused, False otherwise
        """
        pass

    @abstractmethod
    def update_residual(
        self,
        chunk_id: int,
        residual: torch.Tensor
    ):
        """
        Update the residual for a chunk.

        When outputs are reused, the residual from the previous step is
        applied to the current input.

        Args:
            chunk_id: The ID of the chunk
            residual: The residual tensor to store
        """
        pass

    @abstractmethod
    def get_residual(self, chunk_id: int) -> Optional[torch.Tensor]:
        """
        Get the stored residual for a chunk.

        Args:
            chunk_id: The ID of the chunk

        Returns:
            The residual tensor if available, None otherwise
        """
        pass


class KVCompressor(CacheStrategy):
    """
    Abstract base class for KV cache compression strategies.

    KV cache compression manages memory usage by selectively compressing
    KV caches from completed chunks.
    """

    @abstractmethod
    def should_compress(
        self,
        current_chunk_id: int,
        cache_used: int,
        cache_capacity: int,
        **kwargs
    ) -> bool:
        """
        Determine whether KV cache compression should be triggered.

        Args:
            current_chunk_id: The ID of the most recently completed chunk
            cache_used: Current KV cache usage in tokens
            cache_capacity: Total KV cache capacity in tokens
            **kwargs: Additional arguments specific to the implementation

        Returns:
            True if compression should be performed, False otherwise
        """
        pass

    @abstractmethod
    def compress(
        self,
        inference_params,
        chunk_tracker,
        clean_chunk_ids: list,
        active_chunk_ids: list,
        **kwargs
    ) -> Dict[int, Tuple[int, int]]:
        """
        Compress KV caches for specified chunks.

        Args:
            inference_params: Inference parameters containing KV cache
            chunk_tracker: Tracker managing chunk ranges
            clean_chunk_ids: List of chunk IDs to compress
            active_chunk_ids: List of chunk IDs to keep uncompressed
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping chunk_id to (start, end) ranges after compression
        """
        pass
