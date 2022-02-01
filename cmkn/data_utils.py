"""Module that contains classes and functions to handle input data for CMKN experiments.

Attributes:
    ALPHABETS: A dictionary containing different biological alphabets.

Authors:
    Jonas C. Ditz: jonas.ditz@uni-tuebingen.de
"""

import os
import sys
import numpy as np
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch.utils import data

from .utils import kmer2dict, oli2number

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
    """Dinucleotide shuffling implementation

    This function creates a new sequence based on the input sequence using a dinucleatide shuffling to make sure that
    each dinucleotide has the same frequence in the old and new sequence.

    Args:
        seq (:obj:`str`): Input sequence on which the dinucleotide frequencies of the new sequence will be based.

    Returns:
        A new sequence with the same dinucleotide frequencies as the input sequence.
    """
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


class CMKNDataset(data.Dataset):
    """ Custom PyTorch Dataset object to handle input for experiments with CMKN models.

    The CONDataset object can handle dataset given as fasta files, directly. To succesfully read ENCODE datasets, the
    EncodeLoader object will be used. Sequences can be accessed as one-hot encoding matrices or as a two-row matrix,
    where each column holds the k-mer starting at that position mapped onto the upper half of the unit circle. To
    guarantee flexibility, CMKNDataset does not process id strings of fasta files. Therefore, one has to implement a
    child class that inherits from CMKNDataset and process the id strings within this child's __getitem__ function in
    order to create a fully functional Dataset object.

    Attributes:
        data (:obj:`list` of :obj:`str`): List containing all sequences of the dataset
        labels (:obj:`list` of :obj:`str`): List containing the label of each sequence in the dataset
    """

    def __init__(self, filepath, ext="fasta", kmer_size=3, alphabet="DNA_FULL", clean_set=None, tfid=None,
                 seq_encode='onehot'):
        """ Custom Dataset Constructor

        Args:
            filepath (:obj:`str`): Path to the file containing the data.
            ext (:obj:`str`): Extension of the file containing the data. Defaults to 'fasta'.
            kmer_size (:obj:`int`): Size of the kmers considered in the current experiment. This attribute has no
                functionality if the one-hot encoding was chosen. Defaults to 3.
            alphabet (:obj:`str`): Alphabet that was used to create the sequences in the dataset. Defaults to
                'DNA_FULL'.
            clean_set (:obj:`tuple`): Specify if the dataset should be cleaned. Only works for fasta-ish files.
                The first entry of the tuple is an Integer (specifies which column of the label holds the information
                needed for cleaning), and the second entry is a string (specifies the value that indicates an invalid
                entry). Defaults to None
            tfid (:obj:`str`): ID of the used transcription factor. Only used for ENCODE datasets. Defaults to None.
            seq_encode (:obj:`str`): Specifies how the sequences will be made accessible to PyTorch's DataLoader object.
                Currently, only 'onehot' and 'continuous' are supported. Defaults to 'onehot'.

        Raises:
            ValueError: If the dataset contains sequences of different length
            ValueError: If `data` and `labels` have different length
            ValueError: If `seq_encode` is set to an unknown encoding type.
        """

        # call constructor of parent class
        super(CMKNDataset, self).__init__()

        # store parameters
        self.filepath = filepath
        self.alphabet = alphabet
        self.ext = ext
        self.kmer_size = kmer_size

        if self.ext == "fasta":
            # parse input data and check if all sequences have the same length
            aux = []
            get_length = True
            with open(filepath) as handle:
                for record in SimpleFastaParser(handle):
                    if get_length:
                        self.seq_len = len(record[1])
                        get_length = False
                    else:
                        if len(record[1]) != self.seq_len:
                            raise ValueError('Sequences are of different length!')
                    aux.append(record)

            # clean dataset of invalid samples iff the file is fasta-ish and a cleaning criterion, i.e. the column of
            # the id string that holds the class label information, is provided.
            if clean_set is not None:
                self.data = [i[1] for i in aux if i[0].split('|')[clean_set[0]] != clean_set[1]]
                self.labels = [i[0] for i in aux if i[0].split('|')[clean_set[0]] != clean_set[1]]
            else:
                self.data = [i[1] for i in aux]
                self.labels = [i[0] for i in aux]

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

        # create easy-to-use handle for the sequence encoding
        if seq_encode == 'continuous':
            # create dictionary that maps each string of length kmer_size that can be build using alphabet to an integer
            self.kmer_dict = kmer2dict(kmer_size, ALPHABETS[self.alphabet][0])
            self.encode_function = lambda x: self._encode_continuous(x)
        elif seq_encode == 'onehot':
            self.encode_function = lambda x: self._encode_onehot(x)
        else:
            raise ValueError('Unknown encoding option: {}'.format(seq_encode))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Implementation of the getitem routine.

        A sample if provided as a tuple holding a data tensor and a label list that contains the ids of all entries in
        the data tensor.

        Args:
            idx: Index or set of indices of the sample(s) that will be retrieved

        Returns:
            Sample(s) from the input data where the sequences are encoded as specified in the initialization of this
            object.
        """
        # if the set of indices is given as a tensor, convert to list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.encode_function(idx)

    def _encode_onehot(self, idx):
        """This function encodes sequences as one-hot encoding matrices.

        Each column of such a matrix is a one-hot encoding vector of the letter starting at that position in the
        corresponding sequence.

        Args:
            param idx: Index or set of indices of the sample(s) that will be retrieved

        Returns:
            Sample(s) from the input data where the sequences are one-hot encoded.
        """
        # perform different sample extraction depending on whether a list of indices is given or not
        try:
            # initialize data tensor
            data_tensor = torch.zeros(len(idx), len(ALPHABETS[self.alphabet][0]), self.seq_len)

            # fill the data tensor
            for i, sample in enumerate(idx):
                # create one-hot encoding matrix for the requested sequence
                data_tensor[i, :, :] = [[0 if char != letter else 1 for char in self.data[sample]] for letter in
                                        ALPHABETS[self.alphabet][0]]

            # return data tensor and id string
            sample = (data_tensor, [self.labels[i] for i in idx])

        except TypeError:
            # create one-hot encoding of requested sequence
            data_tensor = torch.tensor([[0 if char != letter else 1 for char in self.data[idx]] for letter in
                                        ALPHABETS[self.alphabet][0]], dtype=torch.float)

            # return data tensor and id string
            sample = (data_tensor, [self.labels[idx]])

        return sample

    def _encode_continuous(self, idx):
        """ Return sequences as continuously encoded matrices.

        This function encodes oligomer starting information of each extracted sample as a continuous number mapped
        onto the upper half of the unti circle.

        Args:
            idx: Index or set of indices of the sample(s) that will be retrieved

        Returns:
            Sample(s) from the input data where the sequences are continuously encoded
        """
        # perform different sample extraction depending on whether a list of indices is given or not
        try:
            # initialize data tensor
            data_tensor = torch.zeros(len(idx), 2, self.seq_len)

            # fill the data tensor
            for i, sample in enumerate(idx):
                # calculate the continuous encoding of the current samples
                try:
                    data_tensor[i, :, :] = oli2number(self.data[sample], self.kmer_dict, self.kmer_size,
                                                      ambi=self.alphabet.split('_')[0])
                except ValueError:
                    raise

            # return data tensor and id string
            sample = (data_tensor, [self.labels[i] for i in idx])

        except TypeError:
            # calculate the continuous encoding of the current sample
            try:
                data_tensor = oli2number(self.data[idx], self.kmer_dict, self.kmer_size,
                                         ambi=self.alphabet.split('_')[0])

            except ValueError:
                raise

            # return data tensor and id string
            sample = (data_tensor, [self.labels[idx]])

        return sample

    def update_dataset(self, tfid=None, split='train'):
        """Updates `data` and `labels` attributes

        This function updates the attributes `data` and `labels` to reflect the dataset of a specified transcription
        factor within the encode dataset. Only usable if the Dataset object was initialized to handle ENCODE data.

        Args:
            tfid (:obj:`str`): The transcription factor that will be loaded. If multiple transcription factors should
                be used, provide the IDs as a list. Defaults to None.
            split (:obj:`str`): Specifies if training or test data should be loaded. Defaults to 'train'.

        Raises:
            TypeError: If the object was not initialized to handle ENCODE data.
            ValueError: If tfid is neither :obj:`str` or :obj:`list` of :obj:`str`.
            ValueError: If `data` and `labels` have different sizes after updateing both.
        """
        # check if the DataSet object is set up for ENCODE datasets
        if self.ext != 'seq.gz':
            raise TypeError('DataSet object is not set up for ENCODE datasets')

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
    """ Custom loader handling the ENCODE datasets.

    ENCODE dataset were published in 'B. Alipanahi, A. Delong, M. T. Weirauch, and B. J. Frey.
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