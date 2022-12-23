# GalaxyTools

A python package for analysing galaxy simulations and observations. More documentation can be found at its [readthedocs](https://GalaxyTools.readthedocs.io/) page.

## Package details

The package provides basic structure of a python package that can be used as a starting point to build your own.

## INSTALLATION

To install the package from source, one should clone this package running the following::

    git clone https://github.com/sambit-giri/GalaxyTools.git

To install the package in the standard location, run the following in the root directory::

    python setup.py install

In order to install it in a separate directory::

    python setup.py install --home=directory

One can also install the latest version using pip by running the following command::

    pip install git+https://github.com/sambit-giri/GalaxyTools.git

The dependencies should be installed automatically during the installation process. The list of required packages can be found in the requirements.txt file present in the root directory.

### Tests

For testing, one can use [pytest](https://docs.pytest.org/en/stable/). To run all the test script, run the either of the following::

    python -m pytest tests
    
## CONTRIBUTING

If you find any bugs or unexpected behavior in the code, please feel free to open a [Github issue](https://github.com/sambit-giri/GalaxyTools/issues). The issue page is also good if you seek help or have suggestions for us. For more details, please see [here](https://GalaxyTools.readthedocs.io/contributing.html).

## CREDIT


    This package uses the template provided at https://github.com/sambit-giri/SimplePythonPackageTemplate/ 
    
