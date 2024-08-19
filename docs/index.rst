==========
Toolscosmo
==========

.. :noindex::

.. image:: https://github.com/sambit-giri/toolscosmo/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/sambit-giri/toolscosmo/actions/workflows/ci.yml

A python package for cosmological calculations required to study large-scale structures. The source files can be found at its `GitHub page <https://github.com/sambit-giri/toolscosmo>`_.

Note: There are some modules in the package that are still under active development. Therefore please contact the authors if you get erronous results.

.. :noindex::

Package details
===============

The package provides tools to model standard cosmology and extensions. Currently, `Toolscosmo` supports the following calculations:

* Cosmological calculators: various functions for sevaral cosmological calculations and conversions

* Matter power spectrum:
   * interface with Boltzmann solvers (e.g. CLASS and CAMB) to simulate the linear power spectrum 
   * model the non-linear power spectrum using halo model  

* Emulators: machine learning-based models
   * fast simulation of linear power spectrum

* Halo mass function: probability distribution function of dark matter halo masses

For detailed documentation and how to use them, see `contents page <https://toolscosmo.readthedocs.io/contents.html>`_.

.. :noindex::

Under Development
------------------

* Dark matter merger trees: analystical merger trees using the extended Press-Schechter formalism 

Contents
========

.. toctree::
   :maxdepth: 2

   installation
   tutorials
   contents
   contributing
   authors
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
