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

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

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
    name='toolscosmo',
    version='0.1.9',
    author='Sambit Giri',
    author_email='sambit.giri@su.se',
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={'toolscosmo': ['input_data/*']},
    install_requires=requirements,
    include_package_data=True,
    # ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
