# Pull Request Template

## Description
Please provide a clear and concise description of your changes.

## Type of Change
Please check the relevant option(s):

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance optimization
- [ ] CUDA kernel improvement
- [ ] Code refactoring

## Related Issues
Please link any related issues:
- Fixes #(issue number)
- Related to #(issue number)

## Changes Made
Please describe the changes you made:

### Code Changes
- [ ] Modified Python API
- [ ] Updated CUDA kernels
- [ ] Changed build system
- [ ] Updated dependencies

### Documentation
- [ ] Updated README
- [ ] Updated API documentation
- [ ] Added examples
- [ ] Updated benchmarks

## Testing
Please describe the tests you ran to verify your changes:

- [ ] Existing tests pass: `python -m pytest tests/ -v`
- [ ] Added new tests for new functionality
- [ ] Benchmarks show no performance regression
- [ ] Tested on multiple GPU architectures (if applicable)

### Test Configuration
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- PyTorch: [e.g., 2.1.0]
- CUDA: [e.g., 11.8]
- GPU: [e.g., RTX 4090]

## Performance Impact
If this change affects performance, please provide benchmarks:

### Before
```
# Benchmark results before your changes
```

### After
```
# Benchmark results after your changes
```

## Breaking Changes
If this PR introduces breaking changes, please describe:
- What breaks
- How users can migrate their code
- Why the breaking change is necessary

## Checklist
Please check all that apply:

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

### CUDA-specific (if applicable)
- [ ] CUDA kernels compile without warnings
- [ ] Tested on SM 8.0+ architectures
- [ ] Memory usage has been profiled
- [ ] No memory leaks detected

## Additional Notes
Any additional information that reviewers should know:

## Screenshots (if applicable)
If your changes include visual elements or performance improvements, please add screenshots or graphs.
