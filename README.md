<div align="center">
  <img src="https://cdn.openai.com/triton/assets/triton-logo.png" alt="Triton logo" width="88" height="100">
</div>

[![Wheels](https://github.com/openai/triton/actions/workflows/wheels.yml/badge.svg?branch=release/2.0.x)](https://github.com/openai/triton/actions/workflows/wheels.yml)

We're hiring! If you are interested in working on Triton at OpenAI, we have roles open for [Compiler Engineers](https://openai.com/careers/software-engineer-triton-compiler) and [Kernel Engineers](https://openai.com/careers/kernel-engineer).

**`Documentation`** |
------------------- |
[![Documentation](https://github.com/openai/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/)


# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.

# Quick Installation

You can install the latest stable release of Triton from pip:

```bash
pip install triton
```
Binary wheels are available for CPython 3.7-3.11 and PyPy 3.8-3.9.

And the latest nightly release:

```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

# Install from source

```
git clone https://github.com/openai/triton.git;
cd triton;

pip install ninja cmake wheel; # build-time dependencies
pip install -e python
```

Or with a virtualenv:

```
git clone https://github.com/openai/triton.git;
cd triton;

python -m venv .venv --prompt triton;
source .venv/bin/activate;

pip install ninja cmake wheel; # build-time dependencies
pip install -e python
```

# Building 

1. build triton; in setup, ensure the ptxas, cuobjdump etc version matches the host version

2. build pytorch; note that pytorch also needs to match the version

3. pip uninstall triton (install by pytorch); and re-build

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- vscode intellisense has some difficulty figuring out how to build Triton's C++
  (probably because, in our build, users don't invoke cmake directly, but
  instead use setup.py).  Teach vscode how to compile Triton as follows.

    - Do a local build.
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find python/build -name 'compile_commands.json | xargs readlink -f'`
    - In vscode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Compatibility

Supported Platforms:
  * Linux

Supported Hardware:
  * NVIDIA GPUs (Compute Capability 7.0+)
  * Under development: AMD GPUs, CPUs
