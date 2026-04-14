'''
Created on 2022-12-23
@author: Sambit Giri
Setup script
'''

import setuptools
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy as np
import os



# Check if the Cython source file exists
cython_files = ["src/toolscosmo/cython_ParkinsonColeHelly2008.pyx"]

for cython_file in cython_files:
    if not os.path.isfile(cython_file):
        raise FileNotFoundError(f"Required file not found: {cython_file}")

# Define the extension module
extensions = [
    Extension(
        'toolscosmo.cython_ParkinsonColeHelly2008',
        cython_files,
        language="c++",
        include_dirs=[np.get_include()]
    )
]


setup(
    ext_modules=cythonize(extensions, language_level=3) if cython_files else [],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={'toolscosmo': ['input_data/*']},
    include_package_data=True,
    include_dirs=[np.get_include()],
)
