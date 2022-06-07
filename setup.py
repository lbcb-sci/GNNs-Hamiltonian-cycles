from setuptools import setup, find_packages

setup(
    name="gnn-hamilton",
    version="0.1",
    packages=[{"ham_gnn": "src"}],
    install_requires=["torch", "torch_geometric", "json", "numpy"]
)