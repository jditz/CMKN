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
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch.utils import data

from .utils import kmer2dict, find_kmer_positions

if sys.version_info < (3,):
    import string
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


####################
# MAKRO DEFINITION #
####################


_RNA = maketrans('U', 'T')


# define alphabets
ALPHABETS = {
    'DNA': (
        'ACGT',
        '\x01\x02\x03\x04'
    ),
    'DNA_FULL': (
        'ACGTN',
        '\x01\x02\x03\x04' + '\x00'
    ),
    'DNA_AMBI': (
        'N',
        '\x00'
    ),
    'PROTEIN': (
        'ARNDCQEGHILKMFPSTWYV',
        '\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14'
    ),
    'PROTEIN_FULL': (
        'ARNDCQEGHILKMFPSTWYVXBZJUO',
        '\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14' + '\x00'*6
    ),
    'PROTEIN_AMBI': (
        'XBZJUO',
        '\x00' * 6
    )
}


########################################
# START AUXILIARY FUNCTIONS DEFINITION #
########################################


def seq_to_graph(seq):
    graph = defaultdict(list)
    for i in range(len(seq) - 1):
        graph[seq[i]].append(seq[i+1])
    return graph


def graph_to_seq(graph, first_vertex, len_seq):
    is_done = False
    new_seq = first_vertex
    while not is_done:
        last_vertex = new_seq[-1]
        new_seq += graph[last_vertex][0]
        graph[last_vertex].pop(0)
        if len(new_seq) >= len_seq:
            is_done = True
    return new_seq


# function from: http://www.python.org/doc/essays/graphs/
def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    for node in graph[start][-1]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath:
                return newpath
        return None


def sample_new_graph(graph, last_vertex):
    new_graph = defaultdict(list)
    for vertex in graph:
        edges = graph[vertex]
        np.random.shuffle(edges)
        new_graph[vertex].extend(edges)
    for vertex in new_graph:
        if not find_path(new_graph, vertex, last_vertex):
            return False, new_graph
    return True, new_graph


def doublet_shuffle(seq):
    seq = seq.upper().translate(_RNA)
    last_vertex = seq[-1]
    graph = seq_to_graph(seq)

    is_eulerian = False
    while not is_eulerian:
        is_eulerian, new_graph = sample_new_graph(graph, last_vertex)
    new_seq = graph_to_seq(new_graph, seq[0], len(seq))

    return new_seq


def pad_sequences(sequences, pre_padding=0, maxlen=None, padding='pre', alphabet='DNA_AMBI'):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths) + 2*pre_padding

    # select the correct letter used for padding the sequences
    pad_letter = ALPHABETS[alphabet][0][0]

    corrected_seqs = []
    for s in sequences:

        # if the sequence has the correct length, do nothing
        if len(s) == maxlen:
            corrected_seqs.append(s)
            continue

        if padding == 'post':
            corrected_seq = s + (pad_letter * (maxlen - len(s)))
        elif padding == 'pre':
            corrected_seq = (pad_letter * (maxlen - len(s))) + s
        else:
            raise ValueError('Padding type "%s" not understood' % padding)

        corrected_seqs.append(corrected_seq)

    return corrected_seqs


######################################
# END AUXILIARY FUNCTIONS DEFINITION #
######################################


class CONDataset(data.Dataset):
    """ Custom PyTorch Dataset object to handle input for experiments with CON models.

    This class extension is only tested with fasta files. Extensions for further file formats might be implemented in
    the future.
    """
    def __init__(self, filepath, ext="fasta", kmer_size=3, alphabet="DNA_FULL", clean_set=None, tfid=None):
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
            :param tfid: ID of the used transcription factor.
                :type tfid: String
        """

        # call constructor of parent class
        super(CONDataset, self).__init__()

        # store parameters
        self.filepath = filepath
        self.alphabet = alphabet
        self.ext = ext
        self.kmer_size = kmer_size

        if self.ext == "fasta":
            # parse input data
            self.data = list(SeqIO.parse(filepath, ext))

            # check if all sequences have the same length
            #   -> store length if true, raise exception otherwise
            if all(len(x.seq) == len(self.data[0].seq) for x in self.data):
                self.seq_len = len(self.data[0].seq)
            else:
                raise ValueError('Sequences are of different length!')

            # clean dataset of invalid samples iff the file is fasta-ish and a cleaning criterion, i.e. the column of
            # the id string that holds the class label information, is provided
            if clean_set is not None and ext == "fasta":
                self.data = [i for i in self.data if i.id.split('|')[clean_set[0]] != clean_set[1]]

        elif self.ext == "seq.gz":
            # make sure that the alphabet is given in the correct format
            if alphabet.__contains__('_'):
                alphabet = alphabet.split('_')[0]
            self.data_loader = EncodeLoader(filepath, alphabet=alphabet)
            self.tfids = self.data_loader.get_ids()

            # if no transcription factor is selected, use the first one in the datasets
            if tfid is None:
                tfid = self.tfids[0]

            # load sequences and labels
            self.data, self.labels = self.data_loader.get_dataset(tfid, val_split=0)

            # raise an error if data and labels does not have the same length
            if len(self.data) != len(self.labels):
                raise ValueError('data and labels have to be of the same length\n' +
                                 '    Found instead: len(data)={}, len(labels)={}'.format(len(self.data),
                                                                                          len(self.labels)))

            # check if all sequences have the same length
            #   -> store length if true, raise exception otherwise
            if all(len(x) == len(self.data[0]) for x in self.data):
                self.seq_len = len(self.data[0])
            else:
                raise ValueError('Sequences are of different length!')

        else:
            raise ValueError('Unknown file extension: {}'.format(ext))

        # create dictionary that maps each string of length kmer_size that can be build using alphabet to an integer
        self.kmer_dict = kmer2dict(kmer_size, ALPHABETS[self.alphabet][0])

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
                if self.ext == 'fasta':
                    positions = find_kmer_positions(self.data[sample].seq, self.kmer_dict, self.kmer_size)
                elif self.ext =='seq.gz':
                    positions = find_kmer_positions(self.data[sample], self.kmer_dict, self.kmer_size)
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
        if self.ext == 'fasta':
            sample = (data_tensor, [self.data[i].id for i in idx])
        elif self.ext == 'seq.gz':
            sample = (data_tensor, [self.labels[i] for i in idx])
        else:
            sample = None
        return sample

    def update_dataset(self, tfid=None, split='train'):
        """This function updates the DataSet object to use the sequences for the specified tf id
        """
        # check if the DataSet object is set up for ENCODE datasets
        if self.ext != 'seq.gz':
            raise ValueError('DataSet object is not set up for ENCODE datasets')

        # if no transcription factor is selected, use the first one in the datasets
        if tfid is None:
            tfid = self.tfids[0]

        # check if several tfids were given. If yes, combine corresponding data
        if isinstance(tfid, str):
            # load sequences and labels
            self.data, self.labels = self.data_loader.get_dataset(tfid, val_split=0, split=split)
        elif hasattr(tfid, "__iter__"):
            self.data = self.labels = []
            # iterate over all provided ids and combine sequences and labels
            for s in tfid:
                aux_data, aux_labels = self.data_loader.get_dataset(s, val_split=0, split=split)
                self.data = [*self.data, *aux_data]
                self.labels = [*self.labels, *aux_labels]
        else:
            raise ValueError('Expected String or [String] for tfid; received {}'.format(tfid))

        # raise an error if data and labels does not have the same length
        if len(self.data) != len(self.labels):
            raise ValueError('data and labels have to be of the same length\n' +
                             '    Found instead: len(data)={}, len(labels)={}'.format(len(self.data),
                                                                                      len(self.labels)))


class EncodeLoader(object):
    """ Custom loader handling the ENCODE dataset used in 'B. Alipanahi, A. Delong, M. T. Weirauch, and B. J. Frey.
    Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. Nature Biotechnology,
    doi:10.1038/nbt.3300'
    """
    def __init__(self, datadir, ext='.seq.gz', maxlen=None, pre_padding=0, alphabet='DNA'):
        self.alphabet, self.code = ALPHABETS[alphabet]
        alpha_ambi, code_ambi = ALPHABETS[alphabet + '_AMBI']
        self.translator = maketrans(
            alpha_ambi + self.alphabet, code_ambi + self.code)
        self.alpha_nb = len(self.alphabet)
        self.datadir = datadir
        self.ext = ext
        self.pre_padding = pre_padding
        self.maxlen = maxlen

    def seq2index(self, seq):
        seq = seq.translate(self.translator)
        seq = np.fromstring(seq, dtype='uint8')
        return seq.astype('int64')

    def pad_seq(self, seq):
        seq = pad_sequences(seq, pre_padding=self.pre_padding, maxlen=self.maxlen, padding='post')
        self.maxlen = len(seq[0])
        return seq

    def get_ids(self, ids=None):
        targetnames = sorted(
            [filename.replace("_AC" + self.ext, "") for filename in os.listdir(
                self.datadir) if filename.endswith("_AC" + self.ext)])
        if ids is not None and ids != []:
            targetnames = [
                targetnames[tfid] for tfid in ids if tfid < len(targetnames)
            ]
        return targetnames

    def aug_neg(self, df):
        df2 = df.copy()
        df2['Seq'] = df['Seq'].apply(doublet_shuffle)
        df2['y'] = 0
        df2['EventID'] += '_neg'
        df2.index += df2.shape[0]
        df2['seq_index'] = df2['Seq'].apply(self.seq2index)
        df2 = pd.concat([df, df2])
        return df2

    def load_data(self, tfid, split='train', toindex=False):
        if split == 'train':
            filename = "%s/%s_AC%s" % (self.datadir, tfid, self.ext)
        else:
            filename = "%s/%s_B%s" % (self.datadir, tfid, self.ext)
        df = pd.read_csv(filename, compression='gzip', delimiter='\t')
        df.rename(columns={'Bound': 'y', 'seq': 'Seq'}, inplace=True)
        if toindex:
            df['seq_index'] = df['Seq'].apply(self.seq2index)
        return df

    def get_dataset(self, dataid, split='train', val_split=0.25, top=False, generate_neg=True):
        df = self.load_data(dataid, split)
        if split == 'train' and generate_neg and hasattr(self, "aug_neg"):
            df = self.aug_neg(df)
        X, y = df['Seq'].values, df['y'].values
        X = self.pad_seq(X)
        if top:
            X, _, y, _ = train_test_split(
                X, y, stratify=y, train_size=500)
        if split == 'train' and val_split > 0:
            X, X_val, y, y_val = train_test_split(X, y, test_size=val_split, stratify=y)
            return X, y, X_val, y_val

        return X, y

    def get_sequences(self, tfids):
        # create a list of all files that store sequences
        all_files = []
        for tfid in tfids:
            all_files.append('%s/%s_AC%s' % (self.datadir, tfid, self.ext))
            all_files.append('%s/%s_B%s' % (self.datadir, tfid, self.ext))

        # convert all csv files into pandas data frames
        li = []
        for file in all_files:
            df = pd.read_csv(file, compression='gzip', delimiter='\t')
            li.append(df)

        # concatenate all data frames into one frame
        frame = pd.concat(li, axis=0, ignore_index=True)

        # return the sequences as a list
        return self.pad_seq(frame['seq'].values)