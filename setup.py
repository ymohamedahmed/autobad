from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='autobad',
    version='1.0.0',
    description='A naive implementation of vectorized, reverse-mode autodiff.',
    author='Yousuf',
    author_email='youmed.tech@gmail.com',
    packages=['autobad'],
    install_requires=requirements,
)
