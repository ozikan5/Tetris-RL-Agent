"""Setup file for Tetris RL project."""

from setuptools import setup, find_packages

setup(
    name="tetris-rl",
    version="0.1.0",
    description="Reinforcement learning agents for Tetris",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "pybind11>=2.10.0",
    ],
)
