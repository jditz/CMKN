from setuptools import setup

setup(
    name="cmkn",
    version="0.1",
    description="Implementation of the convolutional motif kernel network (CMKN) architecture.",
    author="Jonas Ditz",
    author_email="jonas.ditz@uni-tuebingen.de",
    packages=["cmkn"],
    python_requires=">=3.7, <3.10",
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.6.1",
        "matplotlib>=3.3.4",
        "pandas>=1.2.3",
        "biopython>=1.78",
        "scikit-learn>=0.24.1",
        "torch>=1.8.1",
        "torchvision>=0.9.1",
        "torchaudio>=0.8.1",
        "sphinx",
        "sphinx_bootstrap_theme"
    ],
    zip_safe=False
)
