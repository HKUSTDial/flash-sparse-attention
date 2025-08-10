---
name: Performance issue
about: Report performance problems or optimization opportunities
title: '[PERFORMANCE] '
labels: 'performance'
assignees: ''

---

**Performance Issue Description**
Describe the performance problem you're experiencing.

**Current Performance**
Please provide benchmark results:
- Sequence length: [e.g., 4096, 8192, 16384]
- Batch size: [e.g., 1, 2, 4]
- Number of heads: [e.g., 16, 32]
- Head dimension: [e.g., 64, 128]
- Current speed: [e.g., 15.2 ms/iteration]
- Memory usage: [e.g., 8.5 GB]

**Expected Performance**
What performance would you expect, and why?
- Expected speed: [e.g., <10 ms/iteration]
- Comparison baseline: [e.g., PyTorch SDPA, Flash Attention]

**Environment Information**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
```

**Benchmark Code**
Provide the code you used for benchmarking:
```python
# Paste your benchmark code here
```

**Profiling Information**
If you have profiling data (from nsys, nvprof, or PyTorch profiler), please include relevant excerpts.

**System Information**
- GPU model and memory: [e.g., RTX 4090 24GB]
- CUDA Compute Capability: [e.g., 8.9]
- CPU: [e.g., Intel i9-12900K]
- RAM: [e.g., 32GB DDR4]

**Additional Context**
- Is this a regression from a previous version?
- Have you tried different batch sizes or sequence lengths?
- Any specific attention patterns (causal, full, custom masks)?
