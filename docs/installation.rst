============
Installation
============

We highly recommend the use of a virtual environement. It helps to keep dependencies required by different projects separate. Few example tools to create virtual environments are `anaconda <https://www.anaconda.com/distribution/>`_, `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ and `venv <https://docs.python.org/3/library/venv.html>`_.

The dependencies should be installed automatically during the installation process. If they fail for some reason, you can install them manually before installing the package. The list of required packages can be found in the *pyproject.toml* file present in the root directory.

For a standard non-editable installation use::

    pip install git+https://github.com/sambit-giri/toolscosmo.git [--user]

The --user is optional and only required if you don't have write permission to your main python installation.
If you wants to work on the code, you can download it directly from the `GitHub <https://github.com/sambit-giri/toolscosmo>`_ page or clone the project using::

    git clone git://github.com/sambit-giri/toolscosmo.git

Then, you can just install in place without copying anything using::

    pip install -e /path/to/toolscosmo [--user]

The package should be installed using pip from the root directory. To install in the standard directory, run::

    pip install .

Or if you are developing and want an editable install::

    pip install -e .

Tests
-----
For testing, one can use `pytest <https://docs.pytest.org/en/stable/>`_. To run all the test script, run the either of the following::

    python -m pytest tests 
