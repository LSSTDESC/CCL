# CCL Development Environment Setup Guide

Based on the project analysis, here's how to set up your shell session for CCL development:

## Project Overview
CCL (Core Cosmology Library) is a Python package with C/C++ extensions built using:
- **Python 3.8+** with setuptools
- **CMake** build system for C/C++ components
- **SWIG** for Python-C bindings
- **GSL** (GNU Scientific Library) and **FFTW** dependencies
- **pytest** for testing
- **flake8** for linting

## Development Environment Setup

### Option 1: Using Conda (Recommended)
This matches the CI environment and ensures all dependencies are properly configured:

```bash
# Create conda environment from the project specification
conda env create -f .github/environment.yml

# Activate the environment
conda activate test

# Install CCL in development mode
pip install -v -e .
```

### Option 2: Manual Setup with System Dependencies

```bash
# Install system dependencies (macOS with Homebrew)
brew install cmake swig gsl fftw

# Or on Linux (Ubuntu/Debian)
sudo apt-get install cmake swig libgsl-dev libfftw3-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install numpy pytest pytest-cov

# Install CCL in development mode
pip install -v -e .
```

### macOS-Specific Configuration
If on macOS, you may need to set these environment variables:

```bash
export DYLD_FALLBACK_LIBRARY_PATH=${CONDA_PREFIX}/lib
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
export CONDA_BUILD_SYSROOT=${SDKROOT}
```

## File Modification Workflow

Once installed in development mode (`pip install -e .`):

**Python files:** Changes in `pyccl/` directory are immediately active - just reimport the module.

**C/C++ files:** After modifying C files in `src/`, rebuild:
```bash
pip install -v -e .
```

## Running Tests

### Python Unit Tests
```bash
# Run all Python tests
OMP_NUM_THREADS=2 pytest -vv pyccl

# Run with coverage
OMP_NUM_THREADS=2 pytest -vv pyccl --cov=pyccl

# Run specific test file
pytest -vv pyccl/tests/test_background.py
```

### Benchmarks
```bash
# Run all benchmarks
OMP_NUM_THREADS=2 pytest -vv benchmarks

# Run specific benchmark
pytest -vv benchmarks/test_power.py
```

### C-Level Tests
For low-level C testing and debugging:

```bash
# Full CMake build with C tests
cmake -H. -Bbuild -DPYTHON_VERSION=0.0.0
make -Cbuild all

# Run C test suite
./build/check_ccl
```

## Debugging the Development Environment

### Debug Build
To compile with debug symbols for C code debugging:

```bash
# Set debug flag during installation
python setup.py develop --debug

# Or with CMake directly
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Debug
make -Cbuild all
```

### Useful Debug Commands

**Check Python installation:**
```bash
python -c "import pyccl; print(pyccl.__file__)"
python -c "import pyccl; print(pyccl.__version__)"
```

**Verify C library loading:**
```bash
python -c "import pyccl.ccllib; print(dir(pyccl.ccllib))"
```

**Check build artifacts:**
```bash
ls -la build/pyccl/
ls -la pyccl/_ccllib.so
```

**View CMake configuration:**
```bash
cat build/CMakeCache.txt
```

### Common Debug Scenarios

**If C extension fails to load:**
1. Check that `build/pyccl/_ccllib.so` was created
2. Verify it was copied to `pyccl/_ccllib.so`
3. Check library dependencies: `otool -L pyccl/_ccllib.so` (macOS) or `ldd pyccl/_ccllib.so` (Linux)

**If tests fail:**
1. Ensure environment matches `.github/workflows/ci.yml`
2. Check OpenMP settings: `echo $OMP_NUM_THREADS`
3. Run with verbose pytest: `pytest -vv -s`

**Memory/performance issues:**
```bash
# Build with optimizations
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
make -Cbuild all

# Profile with Python
python -m cProfile -o profile.stats your_test_script.py
```

## Code Quality

**Linting:**
```bash
# Run flake8 (configuration in .flake8)
flake8 --config .flake8

# Or specific directory
flake8 pyccl/
```

**Type checking:**
```bash
mypy pyccl/
```

## Quick Development Cycle

Complete workflow for making changes:

```bash
# 1. Activate environment
conda activate test  # or: source venv/bin/activate

# 2. Make changes to Python or C files

# 3. If C files changed, rebuild
pip install -v -e .

# 4. Run relevant tests
pytest -vv pyccl/tests/test_yourmodule.py

# 5. Run linting
flake8 pyccl/

# 6. Full test suite before commit
OMP_NUM_THREADS=2 pytest -vv pyccl benchmarks
```

## Additional Resources

- **Documentation**: https://ccl.readthedocs.io/en/latest/
- **Issue Tracker**: https://github.com/LSSTDESC/CCL/issues
- **CI Workflows**: See `.github/workflows/ci.yml` for the exact build/test process
- **Environment Specification**: See `.github/environment.yml` for full dependency list

This setup provides full access to modify files, run tests, and debug both the Python layer and C/C++ implementation.
