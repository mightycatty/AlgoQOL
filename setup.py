from setuptools import setup, find_packages
import os

setup(
    name="algo_qol",
    version="1.0",
    author="shuai.he",
    author_email="shuai.he@gmail.com",

    description="QOL utilities for algo dev",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            'algo-qol = algo_qol.cli:main',
        ],
    },
)
