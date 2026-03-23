# Flash Sparse Attention

This site is built with MkDocs and mkdocstrings.

It combines the existing hand-written project documentation with an auto-generated API reference extracted from Python docstrings.

## Local Preview

Install the documentation dependencies:

```bash
pip install -e .[docs]
```

Start the local documentation server:

```bash
mkdocs serve
```

Build the static site:

```bash
mkdocs build
```

## Documentation Layout

- The existing conceptual and usage documentation stays under `docs/`.
- The API page uses `mkdocstrings` to render Python module docstrings directly from the codebase.
- The recommended public API entrypoints are the Triton-facing exports from `flash_sparse_attn` and `flash_sparse_attn.ops.triton.interface`.
