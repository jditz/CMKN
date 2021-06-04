################################################################
# Implementation of Convolutional Oligo Kernel Networks (CON)  #
#                                                              #
# This package is based on the implementation of CKN-seq by    #
# Dexiong Chen (Chen et al., 2019). The original               #
# implementation can be found under the following url:         #
# https://gitlab.inria.fr/dchen/CKN-seq/-/tree/master/ckn      #
#                                                              #
# Author: Jonas Ditz                                           #
# Contact: ditz@informatik.uni-tuebingen.de                    #
################################################################

from .model import CON
from .utils import (
    find_kmer_positions, kmer2dict, category_from_output, build_kmer_ref_from_file,
    build_kmer_ref_from_list, compute_metrics, register_hooks, plot_grad_flow, Hook, ClassBalanceLoss,
    create_consensus, oli2number, matrix_inverse_sqrt
)
from .data_utils import CONDataset
from .interpretability import seq2pwm, model_interpretation, anchors_to_motivs

MODEL = ['CON', 'CONDataset', 'ClassBalanceLoss']
UTIL = ['find_kmer_positions', 'kmer2dict', 'build_kmer_ref_from_file', 'build_kmer_ref_from_list',
        'category_from_output', 'compute_metrics', 'create_consensus', 'oli2number', 'matrix_inverse_sqrt']
INTERPRETATION = ['model_interpretation', 'seq2pwm', 'anchors_to_motivs']
DEBUG = ['register_hooks', 'plot_grad_flow', 'Hook']

__all__ = MODEL + UTIL + INTERPRETATION + DEBUG
