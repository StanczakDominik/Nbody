from setuptools import setup, find_packages
from os import path
from itertools import chain
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

extras_require={
    'test': ['coverage', 'pytest', 'hypothesis'],
    'dev': ['asv', 'perfplot', 'pudb', 'pytest-profiling'],
}

extras_require['all'] = list(set(chain(*extras_require.values())))

setup(
    name="NBody",
    version="0.1.0",
    packages=find_packages(exclude=['benchmarks', 'prof', 'sourcedocs']),
    url="https://github.com/StanczakDominik/Nbody",
    license="CC BY 4.0",
    author="Dominik Stańczak",
    author_email="stanczakdominik@gmail.com",
    description="molecular dynamics in Python on the GPU via numba.cuda",
    install_requires="numba cupy numpy h5py matplotlib scipy click tqdm pandas gitpython".split(), 
    extras_require=extras_require,
    project_urls={
        'Bug Reports': 'https://github.com/StanczakDominik/nbody/issues',
        'Say Thanks!': 'https://saythanks.io/to/StanczakDominik',
        'Source': 'https://github.com/StanczakDominik/nbody/',
    },
)
