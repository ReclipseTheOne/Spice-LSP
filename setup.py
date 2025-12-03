"""
Setup script for Spice Language Server.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="spice-lsp",
    version="0.1.0",
    description="Language Server Protocol implementation for Spice language",
    author="Spice Team",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "spice-lsp=spice_lsp.server:start_server",
        ],
    },
    python_requires=">=3.8",
)
