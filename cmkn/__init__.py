"""Convolutional Motif Kernel Network

This package implements convolutional motif kernel networks (CMKNs) as well as functions that help interpret trained CMKN
models.

References:
    model: This module contains a basic network that utilizes the convolutional motif kernel.
    layers: This module contains different layers that can be used in creating a CMKN.
    interpretability. This module contains functions that help interpret a trained CMKN model.
    data_utils: This module contains a custom DataSet object that handels data used for training a CMKN model.
    utils: This module contains auxiliary classes and functions used in the implementation of the convolutional motif
        kernel paradigm.

Authors:
    Jonas C. Ditz: jonas.ditz@uni-tuebingen.de
"""

from .data_utils import CMKNDataset
from .interpretability import (
    anchors_to_motifs,
    get_learned_motif,
    get_weight_distribution,
    model_interpretation,
    seq2ppm,
    visualize_kernel_activation,
)
from .layers import CMKNLayer
from .model import CMKN, PhyloCMKN
from .utils import (
    ClassBalanceLoss,
    Hook,
    build_kmer_ref_from_file,
    build_kmer_ref_from_list,
    category_from_output,
    compute_metrics,
    create_consensus,
    find_kmer_positions,
    kmer2dict,
    matrix_inverse_sqrt,
    oli2number,
    plot_grad_flow,
    register_hooks,
)

MODEL = ["CMKN", "CMKNlayer", "CMKNDataset", "ClassBalanceLoss"]
UTIL = [
    "find_kmer_positions",
    "kmer2dict",
    "build_kmer_ref_from_file",
    "build_kmer_ref_from_list",
    "category_from_output",
    "compute_metrics",
    "create_consensus",
    "oli2number",
    "matrix_inverse_sqrt",
]
INTERPRETATION = [
    "model_interpretation",
    "seq2ppm",
    "anchors_to_motifs",
    "get_weight_distribution",
    "get_learned_motif",
    "visualize_kernel_activation",
]
DEBUG = ["register_hooks", "plot_grad_flow", "Hook"]

__all__ = MODEL + UTIL + INTERPRETATION + DEBUG
