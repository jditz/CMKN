# CMKN

Implementation of the Convolutional Motif Kernel Network (CMKN). This method allows to build neural network models that
incorporate learning within the reproducing kernel Hilbert space (RKHS) of the position-aware motif kernel into simple 
end-to-end learning. This approach results in artificial neural networks that can robustly learn on small datasets and 
are inherently interpretable.

## Content
- `cmkn/`: CMKN's source code
- `docs/`: CMKN's documentation
- `scripts/`: Scripts to perform experiments and analysis
- `data/`: Preprocessing scripts for the datasets used in the experiments shown in the corresponding paper  

## Installation

You can perform a user-specific installation by running

    $ python -m pip install .

from the root of the project. We strongly advise an installation in a virtual environment. You can create and activate 
one by executing the following two commands from the root of the project

    $ python -m venv venv
    $ . venv/bin/activate

If you are using anaconda, you can create a separate environment with the following commands

    $ conda create -n venv python=3.9
    $ conda activate venv

and then performing the installation as usual by running

    (venv) $ python -m pip install .

If you plan to extend the code, then you should perform an editable installation with

    (venv) $ python -m pip install -e .

## Testing

You can run the unit-tests by executing

    $ python -m unittest

from the root of the project. The ground truth needed for the tests is stored in the folder `data/gound_truth/`.

## Documentation

The documentation is written with `sphinx`. You can build it by running

    $ cd docs && make html

from the root of the project. The entry point for the documentation will be placed in `doc/_build/html/index.html` which you can open with a browser of your choice.
