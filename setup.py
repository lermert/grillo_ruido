#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name = 'ruido',
    version = '0.0.0a0',
    description = 'Package to work on observational data',
    #long_description =
    # url = 
    author = 'L. Ermert, M. Denolle',
    author_email  = 'lermert@fas.harvard.edu',
    # license
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Topic :: Seismology',
        'Programming Language :: Python :: 3',
    ],
    keywords = 'Ambient seismic noise',
    packages = find_packages(),
    package_data={},
    install_requires = [
        "numpy",
        "scipy",
        "pandas",
        "h5py",
        "pytest"],
    entry_points = {
    },
)

