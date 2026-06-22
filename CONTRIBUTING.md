# CCL Development Environment Setup Guide

Based on the project analysis, here's how to set up your shell session for CCL development:

## Project Overview
CCL (Core Cosmology Library) is a Python package with C/C++ extensions built using:
- **Python 3.10+** with setuptools
- **CMake** build system for C/C++ components
- **SWIG** for Python-C bindings
- **GSL** (GNU Scientific Library) and **FFTW** dependencies
- **pytest** for testing
- **flake8** for linting

## Development Setup and Workflow

For detailed instructions on setting up your development environment, testing, and benchmarking, please refer to the project documentation in `readthedocs/source`:

- **Installation**: [`readthedocs/source/developer_installation.rst`](readthedocs/source/developer_installation.rst)
- **Workflow**: [`readthedocs/source/development_workflow.rst`](readthedocs/source/development_workflow.rst)
- **Testing**: [`readthedocs/source/writing_and_running_unit_tests.rst`](readthedocs/source/writing_and_running_unit_tests.rst)
- **Benchmarking**: [`readthedocs/source/writing_and_running_benchmarks.rst`](readthedocs/source/writing_and_running_benchmarks.rst)

### Quick Start (Conda)

```bash
# Create conda environment from the project specification
conda env create -f .github/environment.yml
conda activate test

# Install CCL in development mode
pip install -v -e .
```

### Quick Development Cycle

```bash
# Data cycle
conda activate test
# ... edit files ...
pip install -v -e .  # Rebuild if C files changed
flake8 pyccl/
OMP_NUM_THREADS=2 pytest -vv pyccl
```

## Updating dependencies


To upgrade the project dependencies defined in [`.github/environment.yml`](.github/environment.yml):

```bash
# Update the conda environment from the YAML (reinstalls pip entries too)
conda env update -f .github/environment.yml --prune

# Activate the environment (name is `test` by default in the YAML)
conda activate test
```

- Notes:
   - `--prune` removes packages that are no longer listed in the YAML.
   - Re-running `env update` will also (re)install the `pip:`-listed packages from the YAML, but if you need to explicitly upgrade pip-installed packages afterwards, run:

```bash
# Reinstall/upgrade pip packages listed in the YAML (example)
pip install --upgrade classy isitgr \
   'velocileptors @ git+https://github.com/sfschen/velocileptors' \
   'baccoemu @ git+https://bitbucket.org/rangulo/baccoemu.git@master' \
   MiraTitanHMFemulator dark_emulator colossus
```

- If you prefer to recreate the environment from scratch:

```bash
conda env remove -n test
conda env create -f .github/environment.yml
conda activate test
```

- Reminder: pip-installed packages come from PyPI or the specified VCS URLs and are unaffected by conda channels; the YAML's `channels:` controls only conda package resolution.

