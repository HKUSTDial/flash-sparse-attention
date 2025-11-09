# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Security Considerations

### CUDA Code Execution

Flash Sparse Attention includes CUDA kernels and C++ extensions that execute on your GPU. When using this library:

- Only install from trusted sources (official PyPI releases or verified builds)
- Be cautious when building from source with modifications
- Verify checksums when downloading pre-built binaries

### Dependencies

This library depends on:
- PyTorch (with CUDA support)
- NVIDIA CUTLASS library
- Standard Python scientific computing libraries

We recommend keeping all dependencies up to date and using virtual environments for isolation.

### Memory Safety

Our CUDA kernels are designed with memory safety in mind:
- Bounds checking is implemented where performance allows
- Memory allocation patterns are tested across different input sizes
- We use established patterns from Flash Attention and CUTLASS

However, as with any low-level CUDA code:
- Very large input tensors may cause out-of-memory errors
- Invalid input shapes may cause undefined behavior
- Custom modifications to kernel code should be thoroughly tested

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

**For security issues:**
- Email: losercheems@gmail.com
- Subject: [SECURITY] FSA Vulnerability Report
- Include: Detailed description, reproduction steps, and potential impact

**For general bugs:**
- Use our [GitHub Issues](https://github.com/SmallDoges/flash-sparse-attention/issues)
- Follow our [contributing guidelines](CONTRIBUTING.md)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution**: Depends on severity and complexity

Critical security issues will be prioritized and may result in emergency releases.

## Security Best Practices

When using Flash Sparse Attention:

1. **Environment Isolation**
   ```bash
   # Use virtual environments
   python -m venv fsa_env
   source fsa_env/bin/activate  # Linux/Mac
   # or
   fsa_env\Scripts\activate     # Windows
   ```

2. **Dependency Management**
   ```bash
   # Keep dependencies updated
   pip install --upgrade torch flash_sparse_attn
   ```

3. **Input Validation**
   ```python
   # Validate tensor shapes and dtypes before processing
   assert query.dtype in [torch.float16, torch.bfloat16, torch.float32]
   assert query.shape == key.shape == value.shape
   ```

4. **Resource Monitoring**
   ```python
   # Monitor GPU memory usage
   import torch
   print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

## Disclosure Policy

- Confirmed vulnerabilities will be disclosed responsibly
- Security fixes will be released as soon as safely possible
- CVE numbers will be requested for significant vulnerabilities
- Credit will be given to security researchers who report issues responsibly

## Contact

For security-related questions or concerns:
- Primary: losercheems@gmail.com
- Project maintainers: See [AUTHORS](AUTHORS) file

For general support:
- GitHub Issues: https://github.com/SmallDoges/flash-sparse-attention/issues
- Documentation: https://github.com/SmallDoges/flash-sparse-attention/tree/main/docs/