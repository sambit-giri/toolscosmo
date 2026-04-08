# Toolscosmo

[![License](https://img.shields.io/github/license/sambit-giri/toolscosmo.svg)](https://github.com/sambit-giri/toolscosmo/blob/main/LICENSE)
[![GitHub Repository](https://img.shields.io/github/repo-size/sambit-giri/toolscosmo)](https://github.com/sambit-giri/toolscosmo)
![CI Status](https://github.com/sambit-giri/toolscosmo/actions/workflows/ci.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/toolscosmo.svg)](https://badge.fury.io/py/toolscosmo)


A Python package for cosmological calculations required to study large-scale structures. Full documentation (with examples, installation instructions and complete module description) can be found at [readthedocs](https://toolscosmo.readthedocs.io/).

**Note:** Some modules in the package are still under active development. Please contact the authors if you encounter any issues.

## Package details

The package provides tools to model standard cosmology and its extensions. Currently, `Toolscosmo` supports the following calculations:

- **Cosmological calculators:** Various functions for cosmological calculations and conversions.

- **Matter power spectrum:**
  - Interface with Boltzmann solvers (e.g., CLASS and CAMB) to simulate the linear power spectrum.
  - Model the non-linear power spectrum using the halo model.

- **Emulators:** Machine learning-based models for:
  - Fast simulation of the linear power spectrum.

- **Halo mass function:** Probability distribution function of dark matter halo masses.

For detailed documentation and usage instructions, see the [contents page](https://toolscosmo.readthedocs.io/contents.html).

## Under Development

- **Dark matter merger trees:** Analytical merger trees using the extended Press-Schechter formalism.

- **Initial Condition Generator:** Lagrangian Perturbation Theory (LPT) based initial condition generator for cosmological numerical simulation frameworks.


## INSTALLATION

To install the package from source, one should clone this package running the following::

    git clone https://github.com/sambit-giri/toolscosmo.git

To install the package in the standard location, run the following in the root directory::

    pip install .



One can also install the latest version using pip by running the following command::

    pip install git+https://github.com/sambit-giri/toolscosmo.git

The dependencies should be installed automatically during the installation process. The list of required packages can be found in the `pyproject.toml` file present in the root directory.

### Optional Dependencies

Some features require optional dependencies to be installed manually:
- **classy**: Install manually by running `pip install classy`
- **PyTorch (Hardware Specific)**: 
  - For CUDA 11.8: `pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html`
  - For CPU-only: `pip install torch torchvision torchaudio`

### Tests

For testing, one can use [pytest](https://docs.pytest.org/en/stable/). To run all the test script, run the following::

    python -m pytest tests
    
## CONTRIBUTING

If you find any bugs or unexpected behavior in the code, please feel free to open a [Github issue](https://github.com/sambit-giri/toolscosmo/issues). The issue page is also good if you seek help or have suggestions for us. For more details, please see [here](https://toolscosmo.readthedocs.io/contributing.html).

## CREDIT

    This package uses the template provided at https://github.com/sambit-giri/SimplePythonPackageTemplate/ 
    