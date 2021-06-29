"""Convolutional Motif Kernel Network

This package implements convolutional motif kernel networks (CMKNs) as well as functions that help interpret trained CMKN
models.

Modules:
    model: This module contains a basic network that utilizes the convolutional motif kernel.
    layers: This module contains different layers that can be used in creating a CMKN.
    interpretability. This module contains functions that help interpret a trained CMKN model.
    data_utils: This module contains a custom DataSet object that handels data used for training a CMKN model.
    utils: This module contains auxiliary classes and functions used in the implementation of the convolutional motif
        kernel paradigm.

Authors:
    Jonas C. Ditz: jonas.ditz@uni-tuebingen.de
"""

from .model import CMKN
from .utils import (
    find_kmer_positions, kmer2dict, category_from_output, build_kmer_ref_from_file,
    build_kmer_ref_from_list, compute_metrics, register_hooks, plot_grad_flow, Hook, ClassBalanceLoss,
    create_consensus, oli2number, matrix_inverse_sqrt
)
from .data_utils import CMKNDataset
from .interpretability import seq2pwm, model_interpretation, anchors_to_motifs

MODEL = ['CMKN', 'CMKNDataset', 'ClassBalanceLoss']
UTIL = ['find_kmer_positions', 'kmer2dict', 'build_kmer_ref_from_file', 'build_kmer_ref_from_list',
        'category_from_output', 'compute_metrics', 'create_consensus', 'oli2number', 'matrix_inverse_sqrt']
INTERPRETATION = ['model_interpretation', 'seq2pwm', 'anchors_to_motifs']
DEBUG = ['register_hooks', 'plot_grad_flow', 'Hook']

__all__ = MODEL + UTIL + INTERPRETATION + DEBUG
