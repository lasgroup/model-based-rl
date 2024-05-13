#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'pandas',
    'numpy>=1.25.2',
    'brax==0.10.0',
    'chex>=0.1.82',
    'flax==0.7.2',
    'jax==0.4.14',
    'jaxlib==0.4.14+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
    'jaxopt==0.8',
    'jaxtyping==0.2.21',
    'jaxutils==0.0.8',
    'ml-collections==0.1.0',
    'numba',
    'scipy==1.11.2',
    'setuptools>=68.1.2',
    'tensorflow>=2.13.0',
    'mbpo @ git+https://github.com/lasgroup/Model-based-policy-optimizers.git',
    'bsm @ git+https://github.com/lasgroup/bayesian_statistical_models.git',
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
