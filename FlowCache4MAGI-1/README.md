# FLOW CACHING FOR AUTOREGRESSIVE VIDEO GENERATION

This repository provides the official implementation of **FlowCache** on **MAGI-1** model, a caching-based acceleration method for autoregressive video generation models.


## 🚀 Installation

Please follow the installation instructions provided in the [MAGI-1](https://github.com/SandAI-org/MAGI-1), as this implementation is built on top of MAGI-1.

---

## ▶️ Usage

### 1. Single Video Generation

Run accelerated generation using FlowCache:

```bash
# FlowCache for text-to-video generation
bash scripts/single_run/flowcache_t2v.sh

# FlowCache for video-to-video generation
bash scripts/single_run/flowcache_v2v.sh

# Baseline acceleration method (TeaCache) for text-to-video
bash scripts/single_run/teacache_t2v.sh

# Baseline acceleration method (TeaCache) for video-to-video
bash scripts/single_run/teacache_v2v.sh
```

### 2. Benchmark Sampling

Generate videos for evaluation on standard benchmarks:

```bash
# VBench
bash scripts/sample/flowcache_vbench.sh
bash scripts/sample/teacache_vbench.sh

# PhysicsIQ
bash scripts/sample/flowcache_physicsiq.sh
bash scripts/sample/teacache_physicsiq.sh
```

### 3. Quality Evaluation

Compute perceptual and structural similarity metrics between original and accelerated generations:

```bash
bash scripts/metric.sh
```

---

## ⚙️ Key Parameters

| Parameter | Description |
|----------|-------------|
| `rel_l1_thresh` | Relative L1 distance threshold for cache reuse decision |
| ` warmup_steps` | Number of denoising steps where reuse is disabled |
| `total_cache_chunk_nums` (`B_total`) | Total number of cache chunks maintained |
| `compress_strategy` | Granularity for selecting important KV caches: `token`, `frame`, or `chunk` |
| `query_granularity` | Granularity for importance scoring: `token`, `frame`, or `chunk` |
| `mix_lambda` | Weight balancing importance and redundancy (default: `0.07`) |
| `mode` | Generation mode: `t2v` (text-to-video), `i2v` (image-to-video), or `v2v` (video-to-video) |
| `prompt` | Input prompt for conditional generation |
| `output_path` | Path to save generated videos |
| `config_file` | Path to MAGI-1 model configuration |

---