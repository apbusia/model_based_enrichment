# Build script for setuptools.

import setuptools
import os


# Define required packages.
REQUIRED_PACKAGES = [
        "tensorflow>=2.6.0",
        "numpy>=1.19.5",
        "joblib>=1.0.1",
        "scipy>=1.7.1",
]

# Read in the project description.
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="model_based_enrichment",
    install_requires=REQUIRED_PACKAGES,
    version="0.0.1",
    author="Akosua Busia",
    author_email="akosua@berkeley.edu",
    description="Code for 'Model-based differential sequencing analysis'",
    long_description=long_description,
    long_description_content_type=
    "text/markdown",
    #url="https://github.com/apbusia/model_based_enrichment",
    packages=setuptools.find_packages(
    ),  # automatically finds packages in the current directory.
    classifiers=
    [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
