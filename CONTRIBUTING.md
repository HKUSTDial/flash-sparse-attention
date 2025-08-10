# Contributing to Flash Dynamic Mask Attention

Everyone is welcome to contribute, and we value everybody's contribution. Code contributions are not the only way to help the community. Answering questions, helping others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts about the awesome projects it made possible, shout out on Twitter every time it has helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our [code of conduct](https://github.com/SmallDoges/flash-dmattn/blob/main/CODE_OF_CONDUCT.md).

## Ways to contribute

There are several ways you can contribute to Flash-DMA:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Implement new attention mechanisms or optimizations.
* Contribute to the examples, benchmarks, or documentation.
* Improve CUDA kernel performance.

If you don't know where to start, there is a special [Good First Issue](https://github.com/SmallDoges/flash-dmattn/contribute) listing. It will give you a list of open issues that are beginner-friendly and help you start contributing to open-source.

> All contributions are equally valuable to the community. 🥰

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](#create-a-pull-request) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature request. It will make it easier for us to come back to you quickly and with good feedback.

### Did you find a bug?

The Flash-DMA library is robust and reliable thanks to users who report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the library itself, and not your code.

Once you've confirmed the bug hasn't already been reported, please include the following information in your issue so we can quickly resolve it:

* Your **OS type and version** and **Python**, **PyTorch**, and **CUDA** versions.
* Your **GPU model** and **CUDA Compute Capability**.
* A short, self-contained, code snippet that allows us to reproduce the bug in less than 30s.
* The *full* traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help.

To get the environment information automatically, run:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
```

### Do you want a new feature?

If there is a new feature you'd like to see in Flash-DMA, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to performance optimization, memory efficiency, or new attention mechanisms?

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.

3. Provide a *code snippet* that demonstrates the feature's usage.

4. If the feature is related to a paper, please include a link.

## Do you want to implement a new attention mechanism?

New attention mechanisms and optimizations are constantly being developed. If you want to implement a new mechanism, please provide:

* A short description of the attention mechanism and a link to the paper.
* Link to the implementation if it is open-sourced.
* Performance benchmarks compared to existing methods.
* CUDA compute capability requirements.

## Do you want to add documentation?

We're always looking for improvements to the documentation that make it more clear and accurate. Please let us know how the documentation can be improved such as typos and any content that is missing, unclear or inaccurate.

## Create a Pull Request

Before writing any code, we strongly advise you to search through the existing PRs or issues to make sure nobody is already working on the same thing.

You will need basic `git` proficiency to contribute to Flash-DMA. You'll need **Python 3.8+** and **CUDA 11.8+** to contribute.

### Development Setup

1. Fork the [repository](https://github.com/SmallDoges/flash-dmattn) by clicking on the **Fork** button.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone https://github.com/<your Github handle>/flash-dmattn.git
   cd flash-dmattn
   git remote add upstream https://github.com/SmallDoges/flash-dmattn.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch!

4. Set up a development environment:

   ```bash
   # Ensure CUDA environment is properly set up
   export CUDA_HOME=/usr/local/cuda  # Adjust path as needed
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install pytest numpy
   ```

5. Develop the features in your branch.

   As you work on your code, you should make sure the test suite passes:

   ```bash
   python -m pytest tests/ -v
   ```

   Flash-DMA also includes performance benchmarks. Run them to ensure your changes don't regress performance:

   ```bash
   python benchmarks/forward_performance.py
   python benchmarks/forward_equivalence.py
   ```

   For CUDA development, ensure your changes compile across supported architectures:

   ```bash
   python setup.py build_ext --inplace
   ```

6. Once you're happy with your changes, add changed files using `git add` and record your changes with `git commit`:

   ```bash
   git add .
   git commit -m "A descriptive commit message"
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

7. Go to your fork on GitHub and click on **Pull Request** to open a pull request.

### Pull request checklist

☐ The pull request title should summarize your contribution.<br>
☐ If your pull request addresses an issue, please mention the issue number in the pull request description to make sure they are linked.<br>
☐ To indicate a work in progress please prefix the title with `[WIP]`.<br>
☐ Make sure existing tests pass.<br>
☐ If adding a new feature, also add tests for it.<br>
☐ If implementing new CUDA kernels, ensure they work across all supported compute capabilities (SM 8.0+).<br>
☐ All public methods must have informative docstrings.<br>
☐ Performance benchmarks should not regress significantly.<br>

### Tests

An extensive test suite is included to test the library behavior and performance. Tests can be found in the [tests](https://github.com/SmallDoges/flash-dmattn/tree/main/tests) folder and benchmarks in the [benchmarks](https://github.com/SmallDoges/flash-dmattn/tree/main/benchmarks) folder.

We use `pytest` for testing. From the root of the repository, run:

```bash
python -m pytest tests/ -v
```

For performance testing:

```bash
python -m pytest benchmarks/ -v
```

### CUDA Development Guidelines

When contributing CUDA code:

1. **Test across architectures**: Ensure your code works on SM 8.0, 9.0, and 10.0.
2. **Memory efficiency**: Profile memory usage and ensure no memory leaks.
3. **Performance**: Benchmark against existing implementations.
4. **Documentation**: Document kernel parameters and expected performance characteristics.

### Code Style

We follow standard Python code style guidelines:

* Use descriptive variable names
* Add type hints where applicable
* Follow PEP 8 guidelines
* Add docstrings to all public functions

For CUDA code:
* Use clear variable names
* Comment complex kernel logic
* Follow NVIDIA CUDA best practices

## Security

If you discover a security vulnerability, please send an e-mail to the maintainers. All security vulnerabilities will be promptly addressed.

## Questions?

If you have questions about contributing, feel free to ask in the [GitHub Discussions](https://github.com/SmallDoges/flash-dmattn/discussions) or open an issue.

Thank you for contributing to Flash Dynamic Mask Attention! 🚀
