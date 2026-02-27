import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['q_len', 'kv_len', 'head_dim'],
)
@triton.jit
def qk_dot_kernel_optimized(
    Q, K, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_oz, stride_oh, stride_om, stride_on,
    num_q_heads, num_kv_heads, group_size, q_len, kv_len, head_dim,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Optimized Q @ K^T * scale kernel with L2 cache swizzling.
    """
    # Grid decomposition with swizzling
    pid = tl.program_id(0).to(tl.int64)
    pid_zh = tl.program_id(1).to(tl.int64)

    batch_id = pid_zh // num_q_heads
    q_head_id = pid_zh % num_q_heads
    kv_head_id = q_head_id // group_size

    grid_m = tl.cdiv(q_len, BLOCK_M)
    grid_n = tl.cdiv(kv_len, BLOCK_N)

    # L2 Cache Swizzling: reorder blocks for better cache reuse
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Q pointers: (BLOCK_M, BLOCK_K)
    q_ptrs = (Q + batch_id * stride_qz + q_head_id * stride_qh +
              offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # K pointers: (BLOCK_N, BLOCK_K) - will be transposed in dot
    k_ptrs = (K + batch_id * stride_kz + kv_head_id * stride_kh +
              offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk)

    # Pre-compute masks that don't change in the loop (Loop Hoisting)
    q_mask = offs_m[:, None] < q_len
    k_mask_n = offs_n[None, :] < kv_len

    # Accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Main loop over K dimension with pointer advancement
    for k in range(0, head_dim, BLOCK_K):
        # Boundary check for K dimension (critical for correctness when head_dim % BLOCK_K != 0)
        k_mask = (offs_k[None, :] + k) < head_dim
        q_mask_combined = q_mask & k_mask
        k_mask_combined = k_mask_n & k_mask.T  # Transpose for correct shape

        q = tl.load(q_ptrs, mask=q_mask_combined, other=0.0)
        k_val = tl.load(k_ptrs, mask=k_mask_combined, other=0.0)
        acc += tl.dot(q, k_val)

        # Pointer advancement (faster than multiplication in each iteration)
        q_ptrs += BLOCK_K * stride_qk
        k_ptrs += BLOCK_K * stride_kk

    # Fused scale
    acc = acc * SCALE

    # Output pointers
    o_ptrs = (Out + batch_id * stride_oz + q_head_id * stride_oh +
              offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

    # Boundary mask
    mask = (offs_m[:, None] < q_len) & (offs_n[None, :] < kv_len)
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=mask)


def qk_dot_triton(q: torch.Tensor, k: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Compute Q @ K^T * scale using optimized Triton kernel.
    Supports GQA/MQA where num_q_heads may differ from num_kv_heads.

    Args:
        q: Query tensor (batch, num_q_heads, q_len, head_dim), contiguous
        k: Key tensor (batch, num_kv_heads, kv_len, head_dim), contiguous
        scale: Scaling factor (e.g., 1/sqrt(head_dim))

    Returns:
        Attention scores (batch, num_q_heads, q_len, kv_len)
    """
    assert q.dim() == 4 and k.dim() == 4
    assert q.shape[0] == k.shape[0]
    assert q.shape[3] == k.shape[3]
    assert q.is_contiguous() and k.is_contiguous()

    batch_size, num_q_heads, q_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_len = k.shape[2]

    # GQA/MQA constraint: num_q_heads must be divisible by num_kv_heads
    assert num_q_heads % num_kv_heads == 0, \
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    group_size = num_q_heads // num_kv_heads

    output = torch.empty((batch_size, num_q_heads, q_len, kv_len),
                         dtype=q.dtype, device=q.device)

    # CRITICAL FIX: Use lambda to dynamically compute grid based on autotune config
    # Previously used hardcoded 128, which would cause missing computations when
    # autotune selects smaller block sizes like 64
    grid = lambda META: (
        triton.cdiv(q_len, META['BLOCK_M']) * triton.cdiv(kv_len, META['BLOCK_N']),
        batch_size * num_q_heads
    )

    qk_dot_kernel_optimized[grid](
        q, k, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        num_q_heads, num_kv_heads, group_size, q_len, kv_len, head_dim,
        SCALE=scale,
    )

    return output


def qk_dot_torch(q: torch.Tensor, k: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Compute Q @ K^T * scale using PyTorch.
    Supports GQA/MQA where num_q_heads may differ from num_kv_heads.

    Args:
        q: Query tensor (batch, num_q_heads, q_len, head_dim)
        k: Key tensor (batch, num_kv_heads, kv_len, head_dim)
        scale: Scaling factor (e.g., 1/sqrt(head_dim))

    Returns:
        Attention scores (batch, num_q_heads, q_len, kv_len)
    """
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads

    if group_size == 1:
        # Standard MHA: direct matmul
        return torch.matmul(q, k.transpose(-2, -1)) * scale
    else:
        # GQA/MQA: use reshape instead of repeat_interleave for memory efficiency
        # Q: (batch, num_q_heads, q_len, head_dim) -> (batch, num_kv_heads, group_size, q_len, head_dim)
        # K: (batch, num_kv_heads, kv_len, head_dim) -> (batch, num_kv_heads, 1, kv_len, head_dim)
        # Then broadcast matmul: (batch, num_kv_heads, group_size, q_len, kv_len)
        batch_size = q.shape[0]
        q_len = q.shape[2]
        head_dim = q.shape[3]
        kv_len = k.shape[2]

        q = q.view(batch_size, num_kv_heads, group_size, q_len, head_dim)
        k = k.unsqueeze(2)  # (batch, num_kv_heads, 1, kv_len, head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch, num_kv_heads, group_size, q_len, kv_len)
        return attn.view(batch_size, num_q_heads, q_len, kv_len)
