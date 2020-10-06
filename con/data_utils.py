################################################################
# Utility functions for handling data in experiments involving #
# the use of CON models                                        #
#                                                              #
# Author: Jonas Ditz                                           #
# Contact: ditz@informatik.uni-tuebingen.de                    #
################################################################

import os
import sys
import numpy as np
from Bio import SeqIO

import torch
from torch.utils import data

from .utils import kmer2dict, find_kmer_positions

# define alphabets
ALPHABETS = {
    'DNA': (
        'ACGT',
        '\x01\x02\x03\x04'
    ),
    'DNA_AMBI': (
        'ACGTN',
        '\x01\x02\x03\x04' + '\x00'
    ),
    'PROTEIN': (
        'ARNDCQEGHILKMFPSTWYV',
        '\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14'
    ),
    'PROTEIN_AMBI': (
        'ARNDCQEGHILKMFPSTWYVXBZJUO',
        '\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14' + '\x00'*6
    )
}


class CONDataset(data.Dataset):
    """ Custom PyTorch Dataset object to handle input for experiments with CON models
    """
    def __init__(self, filepath, ext="fasta", kmer_size=3, alphabet="DNA_AMBI", clean_set=None):
        """ Custom Dataset Constructor

        - **Parameters**::

            :param filepath: Path to the file containing the data
                :type filepath: String
            :param ext: Extension of the file containing the data
                :type ext: String
            :param kmer_size: Size of the kmers considered in the current experiment
            :param clean_set: Specify if the dataset should be cleaned. Only works for fasta-ish files.
                :type clean_set: Tuple where the first entry is an Integer (specifies which column of the label holds
                                 the information needed for cleaning), and the second entry is a string (specifies the
                                 value that indicates an invalid entry)
        """

        # call constructor of parent class
        super(CONDataset, self).__init__()

        # store parameters
        self.filepath = filepath
        self.ext = ext
        self.kmer_size = kmer_size

        # parse input data
        self.data = list(SeqIO.parse(filepath, ext))

        # check if all sequences have the same length
        #   -> store length if true, raise exception otherwise
        if all(len(x.seq) == len(self.data[0].seq) for x in self.data):
            self.seq_len = len(self.data[0].seq)
        else:
            raise ValueError('Sequences are of different length!')

        # clean dataset of invalid samples iff the file is fasta-ish and a cleaning criterion, i.e. the column of the
        # id string that holds the class label information, is provided
        if clean_set is not None and ext == "fasta":
            self.data = [i for i in self.data if i.id.split('|')[clean_set[0]] != clean_set[1]]

        # create dictionary that maps each string of length kmer_size that can be build using alphabet to an integer
        self.kmer_dict = kmer2dict(kmer_size, ALPHABETS[alphabet][0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Implementation of the getitem routine. A sample if provided as a tuple holding a data tensor and a
        label list that contains the ids of all entries in the data tensor.

        - **Parameters**::

            :param idx: Index or set of indices of the sample(s) that will be retrieved

        - **Returns**::

            :return: Sample(s) from the input data
        """

        # if the set of indices is given as a tensor, convert to list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if idx is an integer, convert to list
        if isinstance(idx, int):
            idx = [idx]

        # initialize data tensor
        if len(idx) == 1:
            data_tensor = torch.zeros(len(self.kmer_dict), self.seq_len)
        else:
            data_tensor = torch.zeros(len(idx), len(self.kmer_dict), self.seq_len)

        # fill the data tensor
        for i, sample in enumerate(idx):
            # create list of k-mer positions in the current sequence for each k-mer over the alphabet
            #     -> raise exception if one of the sequences is not created with the specified alphabet
            try:
                positions = find_kmer_positions(self.data[sample].seq, self.kmer_dict, self.kmer_size)
            except ValueError:
                raise

            # iterate over all kmers
            for kmer, j in self.kmer_dict.items():

                # for each kmer, iterate over all occurences and update data tensor, accordingly
                for pos in positions[j]:

                    if len(idx) == 1:
                        data_tensor[j, pos] = 1
                    else:
                        data_tensor[i, j, pos] = 1

        # return data tensor and id string
        sample = (data_tensor, [self.data[i].id for i in idx])
        return sample
