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
from .utils import find_kmer_positions, kmer2dict, category_from_output

__all__ = ['CON', 'find_kmer_positions', 'kmer2dict', 'category_from_output']