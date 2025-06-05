from setuptools import setup, find_packages

setup(
    name="hybrid-maximum-principle-ev",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'scipy>=1.10.0',
    ],
    python_requires='>=3.8',
) 