# latform

Bmad lattice parser/formatter tool.

## Installation

Create a conda environment for the package:

```bash
mamba env create -n latform python=3.12 pip
conda activate latform
```

And then install this package into the environment:

```bash
git clone https://github.com/ken-lauer/latform
cd latform
python -m pip install .
```

To install extras for running the package tests or documentation, use one the
following:

```bash
# Install the base requirements and test suite requirements:
$ python -m pip install .[test]
# Install the base requirements and the documentation requirements:
$ python -m pip install .[doc]
# Install all of the requirements:
$ python -m pip install .[test,doc]
```
