#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'pandas',
    'mbpo @ git+https://github.com/lasgroup/Model-based-policy-optimizers.git',
    'bsm @ git+https://github.com/lasgroup/bayesian_statistical_models.git',
    'wtc @ git+ssh://git@github.com/lenarttreven/when_to_control.git',
    'diff_smoothers @ git+https://github.com/ChristopherBiel/differentiating-smoother.git',
    'ray',
]


extras = {}
setup(
    name='mbrl',
    version='0.0.1',
    license="MIT",
    packages=find_packages(),
    python_requires='>=3.11',
    install_requires=required,
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
