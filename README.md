# CMKN

Implementation of the convolutional motif kernel network (CMKN) introduced in Ditz et al., "Convolutional Motif Kernel Network", 2021.

## Testing

You can run the unit-tests by executing

    $ python -m unittest

from the root of the project. The tests include some standard problems of pulse propagation in nonlinear media. During the tests an interactive plotter demonstrating the integration results will be shown. Unfortunately, at the moment it is not possible to disable it, so running tests in a headless setup is not straightforward.

## Documentation

The documentation is written with `sphinx`. You can build it by running

    $ cd doc && make html

from the root of the project. The entry point for the documentation will be placed in `doc/_build/html/index.html` which you can open with a browser of your choice.