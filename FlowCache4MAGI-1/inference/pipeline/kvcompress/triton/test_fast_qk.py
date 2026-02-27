"""
GQA test for QK^T Triton kernel.
Config: q_heads=24, kv_heads=8
        kv_len: [8k, 16k, 32k], q_len: [2k, 4k, 8k]
        Total: 9 configurations
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from fast_qk import qk_dot_triton, qk_dot_torch


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def benchmark_config(batch_size, num_q_heads, num_kv_heads, q_len, kv_len, head_dim,
                     dtype=torch.float16, device='cuda', scale=1.0,
                     num_warmup=20, num_iters=100):
    """Benchmark a single configuration with memory tracking."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    q = torch.randn(batch_size, num_q_heads, q_len, head_dim, dtype=dtype, device=device).contiguous()
    k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, dtype=dtype, device=device).contiguous()

    # Input memory
    input_mem = (q.numel() + k.numel()) * q.element_size() / 1024 / 1024

    flops = 2 * batch_size * num_q_heads * q_len * kv_len * head_dim

    # Warmup PyTorch
    for _ in range(num_warmup):
        _ = qk_dot_torch(q, k, scale)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iters):
        _ = qk_dot_torch(q, k, scale)
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / num_iters
    torch_tflops = flops / (torch_time * 1e-3) / 1e12
    torch_peak_mem = get_memory_mb() - input_mem

    # Warmup Triton
    for _ in range(num_warmup):
        _ = qk_dot_triton(q, k, scale)
    torch.cuda.synchronize()

    # Benchmark Triton
    torch.cuda.reset_peak_memory_stats()
    start_event.record()
    for _ in range(num_iters):
        _ = qk_dot_triton(q, k, scale)
    end_event.record()
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event) / num_iters
    triton_tflops = flops / (triton_time * 1e-3) / 1e12
    triton_peak_mem = get_memory_mb() - input_mem

    return {
        'torch_time': torch_time,
        'triton_time': triton_time,
        'torch_tflops': torch_tflops,
        'triton_tflops': triton_tflops,
        'speedup': torch_time / triton_time,
        'torch_peak_mem': torch_peak_mem,
        'triton_peak_mem': triton_peak_mem,
    }


def main():
    print("=" * 110)
    print("GQA Test: q_heads=24, kv_heads=8, head_dim=128")
    print("=" * 110)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # GQA config
    batch, num_q_heads, num_kv_heads, head_dim = 1, 24, 8, 128
    kv_lens = [8192, 16384, 32768]
    q_lens = [2048, 4096, 8192]

    print(f"{'q_len':<10} {'kv_len':<10} {'PyTorch(ms)':<14} {'Triton(ms)':<14} {'Speedup':<10} {'TFLOPS':<12} {'MSE':<14} {'PeakMem(MB)':<24}")
    print(f"{'':<10} {'':<10} {'':<14} {'':<14} {'':<10} {'':<12} {'':<14} {'PyTorch':<12} {'Triton':<12}")
    print("-" * 110)

    # Store results for plotting
    labels = []
    torch_times = []
    triton_times = []

    for q_len in q_lens:
        for kv_len in kv_lens:
            try:
                torch.cuda.empty_cache()
                torch.manual_seed(42)
                q = torch.randn(batch, num_q_heads, q_len, head_dim, dtype=torch.float16, device='cuda').contiguous()
                k = torch.randn(batch, num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda').contiguous()

                # Accuracy
                out_torch = qk_dot_torch(q, k, 1.0)
                out_triton = qk_dot_triton(q, k, 1.0)
                mse = ((out_torch - out_triton) ** 2).mean().item()

                # Performance
                result = benchmark_config(batch, num_q_heads, num_kv_heads, q_len, kv_len, head_dim)

                print(f"{q_len:<10} {kv_len:<10} {result['torch_time']:<14.4f} {result['triton_time']:<14.4f} "
                      f"{result['speedup']:<10.2f}x {result['triton_tflops']:<12.2f} {mse:<14.2e} "
                      f"{result['torch_peak_mem']:<12.1f} {result['triton_peak_mem']:<12.1f}")

                # Store for plotting
                labels.append(f"q{q_len//1024}k_k{kv_len//1024}k")
                torch_times.append(result['torch_time'])
                triton_times.append(result['triton_time'])

                del q, k, out_torch, out_triton
            except Exception as e:
                print(f"{q_len:<10} {kv_len:<10} ERROR: {e}")

    print("=" * 110)

    # Plot bar chart
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, torch_times, width, label='PyTorch', color='#ff7f0e')
    bars2 = ax.bar(x + width/2, triton_times, width, label='Triton', color='#1f77b4')

    ax.set_xlabel('Configuration (q_len, kv_len)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('GQA Q@K^T Performance: PyTorch vs Triton\n(q_heads=24, kv_heads=8, head_dim=128)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('gqa_performance.png', dpi=150)
    print(f"\nBar chart saved to: gqa_performance.png")
    plt.show()


if __name__ == "__main__":
    main()
