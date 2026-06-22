.PHONY: test-build clean

# Test the package build locally
# Builds the package, validates it, tests installation, and cleans up on success
test-build: clean
	@echo "Testing package build..."
	python -m pip install --upgrade pip build twine
	python -m build
	@echo "Checking package with twine..."
	python -m twine check dist/*
	@echo "Build artifacts:"
	@ls -lh dist/
	@echo "Verifying installation..."
	python -m venv .test-venv
	.test-venv/bin/pip install --quiet dist/*.whl
	.test-venv/bin/python -c "import pyccl; print(f'pyccl version: {pyccl.__version__}'); print('Import successful!')"
	@echo "Installation verification passed!"
	@echo "Cleaning up..."
	-@rm -rf dist/
	-@rm -rf build/
	-@rm -rf .test-venv
	-@rm -rf *.egg-info
	@echo "Cleanup complete."

# Clean up build artifacts (useful after a failed test-build)
clean:
	@echo "Cleaning up build artifacts..."
	-@rm -rf dist/
	-@rm -rf build/
	-@rm -rf .test-venv
	-@rm -rf *.egg-info
	@echo "Cleanup complete."



