<div align="center">

<div align="center">
  <img src="assets/FlowCache1.png" width="50%">
</div>

# Flow Caching for Autoregressive Video Generation

### ICLR 2026

**[Paper](https://openreview.net/forum?id=vko4DuhKbh)** | **[arXiv](https://arxiv.org/abs/2602.10825)** |

**The first caching framework specifically designed for autoregressive video generation**

Achieving **2.38× speedup on MAGI-1** and **6.7× on SkyReels-V2** with negligible quality degradation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

## 📋 Table of Contents

- [News](#news)
- [Overview](#overview)
- [Method](#method)
- [Main Results](#main-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Todo](#todo)
- [Contact](#contact)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## 📰 News

- 📄 **2026.02.12**: Paper available on [arXiv](https://arxiv.org/abs/2602.10825)!
- 🚀 **2026.02.10**: Code release for [MAGI-1](https://github.com/SandAI-org/MAGI-1) and [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2)!
- 🎉 **2026.01.26**: Paper accepted by ICLR 2026!

---

## 🌟 Overview

FlowCache is a caching framework designed specifically for **autoregressive video generation models**. Unlike traditional caching methods that treat all frames uniformly, FlowCache introduces a **chunkwise caching strategy** where each video chunk maintains independent caching policies, complemented by **importance-based KV cache compression** that maintains fixed memory bounds while preserving generation quality.

<div align="center">
  <img src="assets/visualization.jpg" alt="Overview" width="90%">
</div>

---

## 🔬 Method

### Key Findings

<div align="center">
  <img src="assets/key_findings.jpg" width="90%">
</div>

Our key insight: Different video chunks exhibit heterogeneous denoising states at identical timesteps, necessitating independent caching policies for optimal performance.


### Framework Overview

<div align="center">
  <img src="assets/method.jpg" width="90%">
</div>

FlowCache introduces three key innovations for training-free acceleration of autoregressive video generation:

- **Chunkwise Denoising Heterogeneity**: We identify and formalize that denoising progress varies significantly across video chunks—even at the same timestep—necessitating per-chunk caching decisions.

- **Chunkwise Adaptive Caching**: A novel design where each chunk independently decides whether to reuse or recompute based on its own similarity trajectory.

- **KV Cache Compression Tailored for Video**: We adapt importance–redundancy scoring to autoregressive video generation KV cache compression by introducing an efficient, equivalence-preserving similarity computation, thereby enhancing cache diversity without sacrificing efficiency.

These contributions collectively make FlowCache the first theoretically grounded, training-free caching framework for efficient autoregressive video generation.

For more details, please refer to the original paper.

---

## 📊 Main Results

### Quantitative Performance

#### MAGI-1 (4.5B model)

| Method | PFLOPs | Speedup | Latency (s) | VBench | LPIPS | SSIM | PSNR |
|:------|:------:|:------:|:----------:|:-----:|:-----:|:----:|:----:|
| Vanilla | 306 | **1.0×** | 2873 | 77.06% | - | - | - |
| TeaCache-slow | 294 | 1.12× | 2579 | 77.50% | 0.6211 | 0.2801 | 13.26 |
| TeaCache-fast | 225 | 1.44× | 1998 | 70.11% | 0.8160 | 0.1138 | 8.94 |
| **FlowCache-slow** | 161 | **1.86×** | 1546 | **78.96%** | 0.3160 | 0.6497 | 22.34 |
| **FlowCache-fast** | 140 | **2.38×** | 1209 | **77.93%** | 0.4311 | 0.5140 | 19.27 |

#### SkyReels-V2 (1.3B model)

| Method | PFLOPs | Speedup | Latency (s) | VBench | LPIPS | SSIM | PSNR |
|:------|:------:|:------:|:----------:|:-----:|:-----:|:----:|:----:|
| Vanilla | 113 | **1.0×** | 1540 | 83.84% | - | - | - |
| TeaCache-slow | 58 | 1.89× | 814 | 82.67% | 0.1472 | 0.7501 | 21.96 |
| TeaCache-fast | 49 | 2.2× | 686 | 80.06% | 0.3063 | 0.6121 | 18.39 |
| **FlowCache-slow** | 36 | **5.88×** | 262 | **83.12%** | 0.1225 | 0.7890 | 23.74 |
| **FlowCache-fast** | 28 | **6.7×** | 230 | **83.05%** | 0.1467 | 0.7635 | 22.95 |

---

### Visualization

<div align="center">
  <img src="assets/more_visualization1.jpg" width="90%">
  <img src="assets/more_visualization2.jpg" width="90%">
</div>

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (or 12.x)
- PyTorch 2.0+

### MAGI-1 Setup

```bash
cd FlowCache4MAGI-1
pip install -r requirements.txt
```

### SkyReels-V2 Setup

```bash
cd FlowCache4SkyReels-V2
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### MAGI-1

```bash
cd FlowCache4MAGI-1

bash scripts/single_run/flowcache_t2v.sh
```

### SkyReels-V2

```bash
cd FlowCache4SkyReels-V2

bash run_flowcache_fast.sh
```

---

## 🎯 Supported Models

| Model | Type | Status |
|:------|:-----|:------:|
| **[MAGI-1](https://github.com/SandAI-org/MAGI-1)** | 4.5B-distill | ✅ |
| **[SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2)** | 1.3B | ✅ |

---

## 📝 Todo List

- [ ] Support more autoregressive video generation models (e.g., self-forcing, causal-forcing, etc.)
- [ ] Integrate other training-free acceleration methods (e.g., quantization, etc.)

---

## 📮 Contact

For questions and collaboration inquiries, please contact the co-first authors. The following are all **WeChat IDs**:

- Yuexiao Ma: `ma18640400169`
- Xuzhe Zheng: `zhengxuzhe_`
- Jing Xu: `a2665048215`

---

## 📚 Citation

If you find FlowCache useful for your research, please cite:

```bibtex
@misc{ma2026flowcachingautoregressivevideo,
      title={Flow caching for autoregressive video generation}, 
      author={Yuexiao Ma and Xuzhe Zheng and Jing Xu and Xiwei Xu and Feng Ling and Xiawu Zheng and Huafeng Kuang and Huixia Li and Xing Wang and Xuefeng Xiao and Fei Chao and Rongrong Ji},
      year={2026},
      eprint={2602.10825},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.10825}, 
}
```

---

## 🙏 Acknowledgments

We thank the authors of the following projects for their valuable contributions:

- [MAGI-1](https://github.com/SandAI-org/MAGI-1)
- [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2)
- [TeaCache](https://github.com/ali-vilab/TeaCache)
- [R-KV](https://github.com/Zefan-Cai/R-KV)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

For questions and feedback, please open an issue on GitHub.

</div>
