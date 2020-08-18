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

from .model import CON, CON2
from .utils import (
    find_kmer_positions, kmer2dict, category_from_output, build_kmer_ref, compute_metrics, register_hooks,
    plot_grad_flow, Hook
)
from .data_utils import CONDataset

MODEL = ['CON', 'CON2', 'CONDataset']
UTIL = ['find_kmer_positions', 'kmer2dict', 'build_kmer_ref', 'category_from_output', 'compute_metrics']
DEBUG = ['register_hooks', 'plot_grad_flow', 'Hook']

__all__ = MODEL + UTIL + DEBUG
