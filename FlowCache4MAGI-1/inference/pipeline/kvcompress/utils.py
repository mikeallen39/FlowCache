import math
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict

#################################################################
###################### kv cache utilities #######################
#################################################################

def compute_attention_scores(query_states, key_states_cpu, pooling="max"):
    """
    query_states: [q_len, q_heads, head_dim] on GPU
    key_states_cpu: [kv_len, kv_heads, head_dim] on CPU
    """

    q_len, q_heads, head_dim = query_states.shape
    kv_len, kv_heads, _ = key_states_cpu.shape
    query_group_size = q_heads // kv_heads

    device = query_states.device  # GPU

    # print(f"Before computing attention scores, GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")

    if query_group_size == 1:
        chunk_size = 12150

        attn_weights = torch.empty(kv_heads, q_len, kv_len, device=device, dtype=query_states.dtype)

        for i in range(0, kv_len, chunk_size):
            end_i = min(i + chunk_size, kv_len)
            k_chunk = key_states_cpu[i:end_i].to(device)  # Transfer small chunk to GPU

            attn_chunk = torch.bmm(
                query_states.transpose(0, 1),  # [kv_heads, q_len, head_dim]
                k_chunk.transpose(1, 2)        # [kv_heads, head_dim, chunk_size]
            ) / math.sqrt(head_dim)            # [kv_heads, q_len, chunk_size]

            attn_weights[:, :, i:end_i] = attn_chunk
            del k_chunk, attn_chunk

        return attn_weights

    else:
        # query_states: [q_len, q_heads, head_dim] -> reshape to group
        # We group by query_group, but still compute key in chunks
        query_states = query_states.view(q_len, kv_heads, query_group_size, head_dim)
        # [q_len, kv_heads, g, head_dim] -> permute to [kv_heads, g, q_len, head_dim]
        query_states = query_states.permute(1, 2, 0, 3).contiguous()  # [kv_heads, g, q_len, head_dim]

        if pooling == "mean":
            attn_weights_sum = None
            count = 0
        elif pooling == "max":
            attn_weights_max = None
        else:
            raise ValueError("Pooling method not supported")

        for g in range(query_group_size):
            q_group = query_states[:, g, :, :]  # [kv_heads, q_len, head_dim]

            chunk_size = 12150
            group_attn = torch.empty(kv_heads, q_len, kv_len, device=device, dtype=query_states.dtype)

            for i in range(0, kv_len, chunk_size):
                end_i = min(i + chunk_size, kv_len)
                k_chunk = key_states_cpu[i:end_i].to(device)  # [chunk_size, kv_heads, head_dim]
                k_chunk = k_chunk.permute(1, 2, 0)  # [kv_heads, head_dim, chunk_size]
                attn_chunk = torch.bmm(q_group, k_chunk) / math.sqrt(head_dim)
                group_attn[:, :, i:end_i] = attn_chunk
                del k_chunk, attn_chunk

            # Apply pooling over query_group_size dimension
            if pooling == "mean":
                if attn_weights_sum is None:
                    attn_weights_sum = group_attn
                else:
                    attn_weights_sum += group_attn
                count += 1
            elif pooling == "max":
                if attn_weights_max is None:
                    attn_weights_max = group_attn
                else:
                    attn_weights_max = torch.max(attn_weights_max, group_attn)

            del group_attn

        if pooling == "mean":
            attn_weights = attn_weights_sum / count
            del attn_weights_sum
        elif pooling == "max":
            attn_weights = attn_weights_max
            del attn_weights_max

        return attn_weights


# def cal_similarity(
#     key_states,
# ):
#     # key_states shape: [kv_len, kv_heads, head_dim]
#     start = time.time()
#     k = key_states.permute(1, 0, 2).to('cuda')  # shape: [kv_heads, kv_len, head_dim]
#     num_heads = k.shape[0]

#     k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
#     similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2)).to('cpu')

#     for h in range(num_heads):
#         similarity_cos[h].fill_diagonal_(0.0)

#     end = time.time()
#     return similarity_cos.mean(dim=1).softmax(dim=-1)


def cal_similarity(
    key_states,
):
    # [kv_len, H, D] → [H, kv_len, D]
    k = key_states.permute(1, 0, 2).to('cuda')
    H, L, D = k.shape

    # L2 normalize each key vector per head
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)   # [H, L, D]

    # Step 1: Compute sum of all keys per head → [H, D]
    k_sum = k_norm.sum(dim=1)   # Σ_j k_j

    # Step 2: For each key i, compute k_i ⋅ (Σ_j k_j) → [H, L]
    # That is: (k_norm @ k_sum.T) → use bmm for batch
    # k_norm: [H, L, D], k_sum.unsqueeze(-1): [H, D, 1] → bmm → [H, L, 1]
    dot_with_sum = torch.bmm(k_norm, k_sum.unsqueeze(-1)).squeeze(-1)  # [H, L]

    # Step 3: Apply correction for diagonal (since cos(k_i, k_i) = 1 was included in sum)
    # Original: fill_diagonal_(0) then mean(dim=1) ⇒ (total_sum - 1) / L
    if L == 1:
        mean_sim = torch.zeros(H, 1, device=k.device)  # or handle specially
    else:
        mean_sim = (dot_with_sum - 1.0) / L   # [H, L] ← strictly equivalent to original

    avg_sim = mean_sim

    # Step 5: Softmax → final importance-like distribution
    result = avg_sim.softmax(dim=-1).to('cpu')  # move small result to CPU

    return result


class ChunkKVRangeTracker:
    def __init__(self, total_cache_len: int, clip_token_nums: int, max_batch_size: int):
        self.total_cache_len = total_cache_len
        self.clip_token_nums = clip_token_nums
        self.max_batch_size = max_batch_size
        self.tokens_per_chunk = clip_token_nums * max_batch_size
        self.chunk_ranges: Dict[int, Tuple[int, int]] = {}  # chunk_id -> (start, end)
        self.next_free_idx = 0  # For sequential allocation when not compressed
        self.registered_chunks_ordered: List[int] = []  # Maintain registration order for compression and concatenation

    def register_chunks(self, chunk_ids: List[int]):
        """Batch register multiple chunks and allocate original space"""
        for cid in chunk_ids:
            if cid in self.chunk_ranges:
                continue
            start = self.next_free_idx
            end = start + self.tokens_per_chunk
            if end > self.total_cache_len:
                import pdb; pdb.set_trace()
                raise ValueError("KV cache is full")
            self.chunk_ranges[cid] = (start, end)
            self.registered_chunks_ordered.append(cid)
            self.next_free_idx = end

    def get_range(self, chunk_id: int) -> Tuple[int, int]:
        if chunk_id not in self.chunk_ranges:
            raise KeyError(f"Chunk {chunk_id} not registered. Call register_chunks first.")
        return self.chunk_ranges[chunk_id]

    def get_all_ranges_previous(self, current_chunk_ids: List[int]) -> List[Tuple[int, int]]:
        # Get KV ranges of all previous chunks
        ranges = []
        if len(current_chunk_ids) > 0:
            min_chunk_id = min(current_chunk_ids)
            for cid in self.registered_chunks_ordered:
                if cid >= min_chunk_id:
                    continue
                ranges.append(self.chunk_ranges[cid])
        else:
            # To adapt to MAGI-1's original logic, should return ranges of all registered chunks
            for cid in self.registered_chunks_ordered:
                ranges.append(self.chunk_ranges[cid])
        return ranges

    def get_all_chunk_ids(self) -> List[int]:
        return self.registered_chunks_ordered.copy()
    
    def update_ranges_after_compression(self, new_ranges: Dict[int, Tuple[int, int]]):
        """Update each chunk's range based on actual compressed length"""
        # Update chunk_ranges
        for cid, (start, end) in new_ranges.items():
            if cid in self.chunk_ranges:
                self.chunk_ranges[cid] = (start, end)

        # Update next_free_idx to maximum end
        if new_ranges:
            self.next_free_idx = max(end for start, end in new_ranges.values())
        else:
            self.next_free_idx = 0