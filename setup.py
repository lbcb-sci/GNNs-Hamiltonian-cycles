from setuptools import setup, find_packages

setup(
    name="ham-gnn",
    version="0.1",
    packages=["hamgnn"],
    install_requires=["torch", "torch_geometric", "numpy"]
)