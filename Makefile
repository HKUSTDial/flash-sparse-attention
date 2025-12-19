PYTHON := python

.PHONY: test style quality fixup modified_only_fixup

check_dirs := flash_sparse_attn docs

test:
	$(PYTHON) -m pytest tests

# Run ruff check/format on modified Python files only
modified_only_fixup:
	@current_branch=$$(git branch --show-current); \
	if [ "$$current_branch" = "main" ]; then \
		echo "On main branch, running 'style' target instead..."; \
		$(MAKE) style; \
	else \
		modified_py_files=$$(git diff --name-only main...HEAD | grep '\.py$$' || true); \
		if [ -n "$$modified_py_files" ]; then \
			echo "Checking/fixing files: $${modified_py_files}"; \
			$(PYTHON) -m ruff check $${modified_py_files} --fix; \
			$(PYTHON) -m ruff format $${modified_py_files}; \
		else \
			echo "No .py files were modified"; \
		fi; \
	fi

fixup: modified_only_fixup

# Run style fixes on the whole codebase
style:
	$(PYTHON) -m ruff check $(check_dirs) --fix
	$(PYTHON) -m ruff format $(check_dirs)

# Full quality gate: lint + format check + tests
quality:
	$(PYTHON) -m ruff check $(check_dirs)
	$(PYTHON) -m ruff format $(check_dirs) --check
	$(PYTHON) -m pytest tests

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

