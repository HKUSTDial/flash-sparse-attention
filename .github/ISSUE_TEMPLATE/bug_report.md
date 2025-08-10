---
name: Bug report
about: Create a report to help us improve Flash-DMA
title: '[BUG] '
labels: 'bug'
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Import flash_dmattn
2. Run the following code:
```python
# Paste your code here
```
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment Information**
Please run the following and paste the output:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
```

**Additional context**
- OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12]
- Python version: [e.g. 3.9.7]
- Flash-DMA version: [e.g. 0.1.0]
- CUDA Compute Capability: [e.g. 8.6]

**Error traceback**
If applicable, add the full error traceback:
```
Paste the full traceback here
```

**Additional context**
Add any other context about the problem here, including:
- Sequence lengths and batch sizes you're using
- Whether this works with standard PyTorch SDPA
- Any custom modifications to the code
