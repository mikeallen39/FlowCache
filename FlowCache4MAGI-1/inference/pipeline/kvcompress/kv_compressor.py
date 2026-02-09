import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os

from inference.pipeline.kvcompress.utils import cal_similarity, compute_attention_scores


class KVCompressor:
    def __init__(
        self,
        kernel_size=7,
        mix_lambda=0.07,
        compress_strategy="token",
        query_granularity="chunk",
        score_weighting_method="default",
        power=3,
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        assert compress_strategy in ["token", "frame", "chunk"]
        assert query_granularity in ["token", "frame", "chunk"]
        self.compress_strategy = compress_strategy
        self.query_granularity = query_granularity
        self.score_weighting_method = score_weighting_method
        self.power = power

    def update_kv(
        # The passed kv is the kv cache of all chunks
        self,
        key_states,
        query_states,
        value_states,
        clean_chunk_tokens,
        latent_size_t,
        latent_size_h,
        latent_size_w
    ):
        if self.query_granularity == "token":
            # Take 50 tokens
            query_states = query_states[- 50 : ]
        elif self.query_granularity == "frame":
            query_states = query_states[- latent_size_h * latent_size_w : ]
        elif self.query_granularity == "chunk":
            pass
        else:
            raise ValueError("Invalid query granularity")
        
        if self.compress_strategy == "token":
            return self.update_kv_token(
                key_states,
                query_states,
                value_states,
                clean_chunk_tokens,
                each_chunk_tokens=latent_size_t * latent_size_h * latent_size_w,
            )
        elif self.compress_strategy == "frame":
            return self.update_kv_frame_chunk(
                key_states,
                query_states,
                value_states,
                clean_chunk_tokens,
                together_size=latent_size_h * latent_size_w,
            )
        elif self.compress_strategy == "chunk":
            return self.update_kv_frame_chunk(
                key_states,
                query_states,
                value_states,
                clean_chunk_tokens,
                together_size=latent_size_t * latent_size_h * latent_size_w,
            )
        else:
            raise ValueError("Invalid compress strategy")

    def update_kv_token(
        self,
        key_states,
        query_states,
        value_states,
        clean_chunk_tokens,
        each_chunk_tokens,
    ):
        each_chunk_tokens = int(each_chunk_tokens)
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[0]

        attn_weights = compute_attention_scores(query_states, key_states[:clean_chunk_tokens])
        attn_weights_sum = (
            nn.functional.softmax(
                attn_weights[:, :, : clean_chunk_tokens],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)
            .to(query_states.dtype)
        )

        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).to('cpu')

        similarity_cos = cal_similarity(key_states[:clean_chunk_tokens, :, :]).to('cpu')

        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)

        # Ensure final score is non-negative for weighting
        min_scores_per_head = final_score.min(dim=-1, keepdim=True).values  # (num_kv_heads, 1)
        final_score = final_score - min_scores_per_head

        # Note that final_score contains negative numbers
        # Apply different weighting methods to final_score, relatively making tokens at later positions more likely to be selected
        if self.score_weighting_method == "no_weight":
            print("Using no weighting method")
            pass

        elif self.score_weighting_method == "hard_code":
            print("Using hard code weighting method")
            final_score[:, :each_chunk_tokens] -= 1e6
        
        elif self.score_weighting_method == "exponential":
            print("Using exponential weighting method")
            seq_len = final_score.shape[1]
            positions = torch.arange(seq_len, dtype=torch.float32, device=final_score.device) / (seq_len - 1) if seq_len > 1 else torch.zeros(1)
            decay_rate = 2.0
            # Normalize to [0.1, 1.0] range
            exponential_values = 1 - torch.exp(-decay_rate * positions)
            max_value = 1 - torch.exp(torch.tensor(-decay_rate, device=final_score.device))  # Value when positions=1
            weights = 0.1 + 0.9 * (exponential_values / max_value)
            final_score = final_score * weights.unsqueeze(0)

        elif self.score_weighting_method == "polynomial":
            print(f"Using polynomial weighting method, power={self.power}")
            seq_len = final_score.shape[1]
            positions = torch.arange(seq_len, dtype=torch.float32, device=final_score.device) / (seq_len - 1) if seq_len > 1 else torch.zeros(1)
            # Normalize to [0.1, 1.0] range
            weights = 0.1 + 0.9 * (positions ** self.power)
            final_score = final_score * weights.unsqueeze(0)
        elif self.score_weighting_method == "upper_convex_polynomial":
            print(f"Using upper convex polynomial weighting method, power={self.power}")
            seq_len = final_score.shape[1]
            positions = torch.arange(seq_len, dtype=torch.float32, device=final_score.device) / (seq_len - 1) if seq_len > 1 else torch.zeros(1)
            max_value = 2.0
            # Construct upper convex n-th degree polynomial: w(x) = max_value * (1 - (1-x)^n)
            weights = max_value * (1 - (1 - positions) ** self.power)
            final_score = final_score * weights.unsqueeze(0)

        elif self.score_weighting_method == "gaussian":
            print("Using gaussian weighting method")
            # Emphasize previous information more
            seq_len = final_score.shape[1]

            positions = torch.arange(seq_len, dtype=torch.float32, device=final_score.device)
            sigma = seq_len / 4.0  # ← Adjustable! Smaller values emphasize the beginning more

            gaussian_decay = torch.exp(-0.5 * (positions / sigma) ** 2)
            min_decay = torch.exp(torch.tensor(-0.5 * ((seq_len - 1) / sigma) ** 2, device=final_score.device))

            # Map [min_decay, 1.0] → [0.1, 1.0]
            weights = 0.1 + 0.9 * ((gaussian_decay - min_decay) / (1.0 - min_decay))
            final_score = final_score * weights.unsqueeze(0)
        else:
            raise ValueError(f"Unknown score weighting method: {self.score_weighting_method}")

        # Calculate number of tokens to keep
        num_to_keep = self.budget

        # Select top-k tokens
        try:
            indices = final_score.topk(num_to_keep, dim=-1).indices  # shape: (num_kv_heads, num_to_keep)
            del final_score
        except RuntimeError:
            import pdb; pdb.set_trace()
        indices = indices.unsqueeze(-1).expand(-1, -1, head_dim).permute(1, 0, 2)  # shape: (num_to_keep, num_kv_heads, head_dim)

        indices = indices.to(key_states.device)

        # Compress non-recent parts
        k_past_compress = key_states[:clean_chunk_tokens, :, :].gather(dim=0, index=indices)
        v_past_compress = value_states[:clean_chunk_tokens, :, :].gather(dim=0, index=indices)
        k_cur = key_states[clean_chunk_tokens :, :, :]
        v_cur = value_states[clean_chunk_tokens :, :, :]

        key_compress = torch.cat([k_past_compress, k_cur], dim=0)
        value_compress = torch.cat([v_past_compress, v_cur], dim=0)

        return key_compress, value_compress, indices

    def update_kv_frame_chunk(
        self,
        key_states,
        query_states,
        value_states,
        clean_chunk_tokens,
        together_size,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[0]

        # ========== Compression Logic ==========

        # Step 1: Compute attention weights
        attn_weights = compute_attention_scores(query_states, key_states)
        attn_weights_sum = (
            nn.functional.softmax(
                attn_weights[:, :, : clean_chunk_tokens],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)  # shape: (num_kv_heads, clean_chunk_tokens)
            .to(query_states.dtype)
        )

        # Step 2: Pooling to get "importance" of each token
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).to('cpu')  # shape: (num_kv_heads, clean_chunk_tokens)

        # Step 3: Compute similarity between tokens
        similarity_cos = cal_similarity(key_states[:clean_chunk_tokens, :, :]).to('cpu')

        # Step 4: Compute final score for each token
        final_score_per_token = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
        # shape: (num_kv_heads, clean_chunk_tokens)

        # ========== Frame-wise or Chunk-wise Aggregation ==========
        # In the code below, chunk is also referred to as frame; they are conceptually consistent, just differing in how many tokens are aggregated into one frame/chunk

        assert clean_chunk_tokens % together_size == 0
        num_frames = clean_chunk_tokens // together_size

        # Reshape to (num_kv_heads, num_frames, together_size)
        score_frames = final_score_per_token.view(
            key_states.shape[1], num_frames, together_size
        )

        # Aggregate scores for each frame
        frame_scores = score_frames.mean(dim=-1)  # shape: (num_kv_heads, num_frames)

        # Calculate number of frames to keep
        assert self.budget % together_size == 0
        num_frames_to_keep = self.budget // together_size

        # Select top-k frames for each head
        frame_indices = frame_scores.topk(num_frames_to_keep, dim=-1).indices
        # shape: (num_kv_heads, num_frames_to_keep)
       

        # Convert frame_indices to token indices
        # frame_indices: frame id selected by each head

        # offset: [0, 1, ..., together_size-1]
        token_offsets = torch.arange(together_size, device=key_states.device)
        frame_indices_expanded = frame_indices.unsqueeze(-1) * together_size
        token_indices_per_head = frame_indices_expanded + token_offsets  # shape: (num_heads, num_frames_to_keep, together_size)
        token_indices_flat = token_indices_per_head.view(key_states.shape[1], -1)  # (num_heads, K * together_size)
        indices_gather = token_indices_flat.permute(1, 0).unsqueeze(-1).expand(-1, -1, head_dim)  # shape: (kept_tokens, num_kv_heads, head_dim)

        # Gather from key/value
        k_past_compress = key_states[:clean_chunk_tokens, :, :].gather(dim=0, index=indices_gather)
        v_past_compress = value_states[:clean_chunk_tokens, :, :].gather(dim=0, index=indices_gather)

        # ========== Concatenate Recent Parts ==========
        k_cur = key_states[clean_chunk_tokens:, :, :]
        v_cur = value_states[clean_chunk_tokens:, :, :]

        key_compress = torch.cat([k_past_compress, k_cur], dim=0)
        value_compress = torch.cat([v_past_compress, v_cur], dim=0)

        return key_compress, value_compress, indices_gather  # token indices


def plot_tensor_values(tensor_1d: torch.Tensor, title: str = "Tensor Values", save_path: str = None,
                       xlabel: str = "Position", ylabel: str = "Value", figsize: tuple = (10, 6)):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping plot")
        return

    # Ensure input is a 1D tensor
    if tensor_1d.dim() != 1:
        raise ValueError(f"Input tensor must be 1D, got {tensor_1d.dim()}D")

    # Convert to numpy array
    values = tensor_1d.detach().cpu().float().numpy()
    positions = np.arange(len(values))

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(positions, values, 'b-', linewidth=2, markersize=4, alpha=0.8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.close()