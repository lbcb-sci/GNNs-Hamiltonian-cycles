from setuptools import setup, find_packages

setup(
    name="gnn-hamilton",
    version="0.1",
    packages=find_packages(),
    install_requires=["torch", "torch_geometric", "numpy"]
)