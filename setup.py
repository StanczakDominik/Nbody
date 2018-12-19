from setuptools import setup, find_packages

setup(
    name='NBody',
    version='0.1.0',
    packages=find_packages('nbody'),
    url='https://github.com/StanczakDominik/Nbody',
    license='CC BY 4.0',
    author='Dominik Stańczak',
    author_email='stanczakdominik@gmail.com',
    description='n-body/molecular dynamics in Python on the GPU via numba.cuda'
    )
