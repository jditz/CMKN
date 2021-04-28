#############################################################
# Utility functions for convolutional oligo kernel networks #
#                                                           #
# Author: Jonas Ditz                                        #
# Contact: ditz@informatik.uni-tuebingen.de                 #
#############################################################

import math
import numpy as np
from itertools import combinations, product
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from Bio import SeqIO

import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score, precision_recall_curve, average_precision_score,
                             f1_score)

from timeit import default_timer as timer


# definition of macros
EPS = 1e-4


def find_kmer_positions(sequence, kmer_dict, kmer_size):
    """Utility to localize k-meres

    This function takes a sequences and returns a list of positions for each k-mer of
    the alphabet.

    - **Parameters**::

        :param sequence: Input sequences which is used to generate list of k-mer positions.
            :type sequence: string
        :param kmer_dict: Dictionary mapping each possible k-mer to an integer.
            :type kmer_dict: dictionary
        :param kmer_size: Size of the k-meres.
            :type kmer_size: integer

    - **Returns**::

        :return positions: List containing a list of positions for each k-mer.
            :rtype positions: list

    - **Raises**::

        :raise ValueError: If one of the sequences was not build using the specified alphabet.
    """

    # initialize list of lists, where each list will hold the starting positions of one specific k-mer
    positions = [[] for i in range(len(kmer_dict))]

    # iterate through all k-meres in the input sequence
    for x, y in combinations(range(len(sequence)+1), r=2):
        if len(sequence[x:y]) == kmer_size:

            # check if current k-mere is part of the dictionary and raise exception if  not
            #   -> if k-mere is not part of the dictionary, the sequence is not created using the specified
            #      alphabet
            if sequence[x:y] not in kmer_dict:
                raise ValueError('Substring \'' + sequence[x:y] + '\' not found in k-mere dictionary!\n' +
                                 '            The sequence ' +
                                 '\'{}\' was NOT build using the specified alphabet.'.format(sequence))

            # append start position of the current k-mer to the corresponding list of positions
            positions[kmer_dict.get(sequence[x:y])].append(x)

    return positions


def kmer2dict(kmer_size, alphabet):
    """Utility to create k-mere mapping

    This function takes an alphabet and a integer k and returns a dictionary that maps each k-mer
    (that can be created using the specified alphabet) to an integer. This dictionary can than be
    used to create input features for CON.

    - **Parameters**::

        :param kmer_size: Size of the k-meres.
            :type kmer_size: integer
        :param alphabet: Alphabet used to generate the k-meres.
            :type alphabet: list (of chars)

    - **Returns**::

        :return kmer_dict: Dictionary mapping each k-mer to an integer.
            :rtype kmer_dict: dictionary
    """

    # initialize an empty dictionary
    kmer_dict = {}

    # iterate through all possible k-meres of the alphabet and update the dictionary
    for kmer in product(alphabet, repeat=kmer_size):
        kmer_dict.update({''.join(kmer): len(kmer_dict)})

    return kmer_dict


def oli2number(seq, kmer_dict, kmer_size, ambi='DNA'):
    """Utility to translate a string into a tensor

    This function takes a sequence represented by a string and returns an tensor, where each oligomer starting in the
    sequence is encoded by a 2-dimensional vector at the corresponding position

    - **Parameters**::

        :param seq: Input sequences which is used to generate list of k-mer positions.
            :type seq: string
        :param kmer_dict: Dictionary mapping each possible k-mer to an integer.
            :type kmer_dict: dictionary
        :param kmer_size: Size of the k-meres.
            :type kmer_size: integer
        :param ambi: String indicating the alphabet from which the ambiguous character should be chosen
            :type ambi: string

    - **Returns**::

        :return positions: Tensor encoding oligomers starting at each position of seq.
            :rtype positions: Tensor (2 x len(seq))

    - **Raises**::

        :raise ValueError: If one of the sequences was not build using the specified alphabet.
    """

    # select the correct ambiguous character
    if ambi == 'DNA':
        ambi = 'N'
    elif ambi == 'PROTEIN':
        ambi = 'X'
    else:
        raise ValueError('Error in create_consensus: Please set ambi to either DNA or PROTEIN.')

    # store the length of the sequence
    seq_len = len(seq)

    # initialize tensor to hold the oligomer encoding
    oli_tensor = torch.zeros((2, seq_len))

    # add blank character to the end of the sequence
    seq = seq + ambi * kmer_size

    # iterate through all oligomers of length kmer_size
    for i in range(seq_len):

        # check if current k-mere is part of the dictionary and raise exception if  not
        #   -> if k-mere is not part of the dictionary, the sequence is not created using the specified
        #      alphabet
        if seq[i:i+kmer_size] not in kmer_dict:
            raise ValueError('Substring \'' + seq[i:i+kmer_size] + '\' not found in k-mere dictionary!\n' +
                             '            The sequence ' +
                             '\'{}\' was NOT build using the specified alphabet.'.format(seq))

        # update tensor with the number of the starting oligomer
        oli_tensor[0, i] = np.cos((kmer_dict.get(seq[i:i+kmer_size]) / len(kmer_dict)) * np.pi)
        oli_tensor[1, i] = np.sin((kmer_dict.get(seq[i:i+kmer_size]) / len(kmer_dict)) * np.pi)

    return oli_tensor


def create_consensus(sequences, extension=None, ambi='DNA'):
    """Build consensus sequence

    This utility function takes sequences, either stored in a file readable by Biopython or in a list, and creates a
    consensus sequence.
    """

    # select the correct ambiguous character
    if ambi == 'DNA':
        ambi = 'N'
    elif ambi == 'PROTEIN':
        ambi = 'X'
    else:
        raise ValueError('Error in create_consensus: Please set ambi to either DNA or PROTEIN.')

    # if sequences is a string, we have to deal with a file
    if isinstance(sequences, str):

        # check if extension is set properly
        if extension is None:
            raise ValueError('Error in create_consensus: Please specify a valid file format. ' +
                             'Currently, only fasta is tested.')

        # import the Biopython routines to create a consensus sequence
        from Bio import AlignIO
        from Bio.Align import AlignInfo

        # read in all sequences and create consensus sequence
        alignment = AlignIO.read(sequences, extension)
        summary_align = AlignInfo.SummaryInfo(alignment)
        consensus = summary_align.dumb_consensus(threshold=0.5, ambiguous=ambi)

    else:

        # import the needed Biopython routines
        from Bio.Align import MultipleSeqAlignment, AlignInfo

        # create multiple alignment object
        alignment = MultipleSeqAlignment([])

        # iterate over list of sequences and add each sequence to the alignment
        for seq in sequences:
            alignment.add_sequence('aux', seq)

        # create consensus sequence
        summary_align = AlignInfo.SummaryInfo(alignment)
        consensus = summary_align.dumb_consensus(threshold=0.25, ambiguous=ambi)

    return consensus


def build_kmer_ref_from_file(filepath, extension, kmer_dict, kmer_size):
    """Utility to create reference k-mere positions

    This function uses a specified training set of sequences and creates a tensor that stores for each position the
    frequency of k-mers (k = kmer_size) starting at that position.

    - **Parameters**::

        :param filepath: Path to the file that contains the dataset
            :type filepath: String
        :param extension: Extension of the dataset file (needed for Biopython's SeqIO routine)
            :type extension: String
        :param kmer_dict: Dictionary mapping each k-mer to an integer.
            :type kmer_dict: dictionary
        :param kmer_size: Size of the k-mers (k = kmer_size).
            :type kmer_size: Integer
    """
    # get the length of the sequences in the dataset by first reading only the first entry
    first_record = next(SeqIO.parse(filepath, extension))

    # initialize the tensor holding the kmer reference positions
    ref_pos = torch.zeros(len(kmer_dict), len(first_record.seq), requires_grad=False)

    # keep track of the number of sequences in the dataset
    data_size = 0

    with open(filepath, 'rU') as handle:
        # iterate through file
        for record in SeqIO.parse(handle, extension):

            # update number of sequences in the dataset
            data_size += 1

            # get kmer positions in the current sequence
            positions = find_kmer_positions(record.seq.upper(), kmer_dict, kmer_size)

            # update reference tensor
            for i, pos in enumerate(positions):
                for p in pos:
                    ref_pos[i, p] += 1

    # divide every entry in the ref position tensor by the size of the dataset
    return ref_pos / data_size


def build_kmer_ref_from_list(seq_list, kmer_dict, kmer_size):
    """Utility to create reference k-mere positions

    This function uses a specified training set of sequences and creates a tensor that stores for each position the
    frequency of k-mers (k = kmer_size) starting at that position.

    - **Parameters**::

        :param seq_list: List containing all sequences of the dataset
            :type seq_list: List of String
        :param kmer_dict: Dictionary mapping each k-mer to an integer.
            :type kmer_dict: dictionary
        :param kmer_size: Size of the k-mers (k = kmer_size).
            :type kmer_size: Integer
    """

    # check whether all sequences have the same length
    if not all(len(x) == len(seq_list[0]) for x in seq_list):
        raise ValueError('Sequences are of different length!')

    # initialize the tensor holding the kmer reference positions
    ref_pos = torch.zeros(len(kmer_dict), len(seq_list[0]), requires_grad=False)

    # keep track of the number of sequences in the dataset
    data_size = 0

    c = 0
    tic = timer()
    for seq in seq_list:
        # update number of sequences in the dataset
        data_size += 1

        # get kmer positions in the current sequence
        positions = find_kmer_positions(seq, kmer_dict, kmer_size)

        # update reference tensor
        for i, pos in enumerate(positions):
            for p in pos:
                ref_pos[i, p] += 1

        if (c % 10000) == 0:
            toc = timer()
            print("Finished {} samples, elapsed time: {:.2f}min".format(c, (toc - tic) / 60))

        c += 1

    # divide every entry in the ref position tensor by the size of the dataset
    return ref_pos / data_size


def anchors_to_motivs(anchor_points, kmer_ref, kmer_dict, kmer_size, type="amino", outdir=""):
    """Motiv creation

    This function takes a list of anchor points, learned by the oligo kernel layer of a CON, and returns the motivs that
    are represented by these anchor points.

    ATTENTION: Motifs generated by this function are not meant for scientific publications!

    - **Parameters**::

        :param anchor_points: List of anchor points, learned by an oligo kernel layer
            :type anchor_points: List of floats
        :param kmer_ref: Tensor storing the frequency with which every k-mer starts at each position in the dataset
            :type kmer_ref: Tensor
        :param kmer_dict: Dictionary mapping each k-mer to an index
            :type kmer_dict: Dictionary
    """
    # import needed libraries
    import matplotlib as mpl
    from matplotlib.text import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt

    # initialize needed functionality
    fp = FontProperties(family="Arial", weight="bold")
    globscale = 1.35
    if type == 'nucleotides':
        LETTERS = {"T": TextPath((-0.305, 0), "T", size=1, prop=fp),
                   "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
                   "U": TextPath((-0.384, 0), "U", size=1, prop=fp),
                   "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
                   "C": TextPath((-0.366, 0), "C", size=1, prop=fp)}
        COLOR_SCHEME = {'G': 'orange',
                        'A': 'darkgreen',
                        'C': 'blue',
                        'T': 'red',
                        'U': 'red'}
    elif type == 'amino':
        LETTERS = {"G": TextPath((-0.384, 0), "G", size=1, prop=fp),
                   "S": TextPath((-0.366, 0), "S", size=1, prop=fp),
                   "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
                   "Y": TextPath((-0.305, 0), "Y", size=1, prop=fp),
                   "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
                   "Q": TextPath((-0.375, 0), "Q", size=1, prop=fp),
                   "N": TextPath((-0.35, 0), "N", size=1, prop=fp),
                   "K": TextPath((-0.39, 0), "K", size=1, prop=fp),
                   "R": TextPath((-0.35, 0), "R", size=1, prop=fp),
                   "H": TextPath((-0.35, 0), "H", size=1, prop=fp),
                   "D": TextPath((-0.366, 0), "D", size=1, prop=fp),
                   "E": TextPath((-0.33, 0), "E", size=1, prop=fp),
                   "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
                   "V": TextPath((-0.335, 0), "V", size=1, prop=fp),
                   "L": TextPath((-0.33, 0), "L", size=1, prop=fp),
                   "I": TextPath((-0.14, 0), "I", size=1, prop=fp),
                   "P": TextPath((-0.305, 0), "P", size=1, prop=fp),
                   "W": TextPath((-0.48, 0), "W", size=1, prop=fp),
                   "F": TextPath((-0.305, 0), "F", size=1, prop=fp),
                   "M": TextPath((-0.415, 0), "M", size=1, prop=fp),
                   "X": TextPath((-0.35, 0), "N", size=1, prop=fp)}
        COLOR_SCHEME = {'G': 'darkgreen',
                        'S': 'darkgreen',
                        'T': 'darkgreen',
                        'Y': 'darkgreen',
                        'C': 'darkgreen',
                        'Q': 'darkgreen',
                        'N': 'darkgreen',
                        'K': 'blue',
                        'R': 'blue',
                        'H': 'blue',
                        'D': 'red',
                        'E': 'red',
                        'A': 'black',
                        'V': 'black',
                        'L': 'black',
                        'I': 'black',
                        'P': 'black',
                        'W': 'black',
                        'F': 'black',
                        'M': 'black',
                        'X': 'brown'}
    else:
        raise ValueError('Unknown alphabet type!')

    def letterAt(letter, x, y, yscale=1, ax=None):
        text = LETTERS[letter]

        t = mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale) + \
            mpl.transforms.Affine2D().translate(x, y) + ax.transData
        p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter], transform=t)
        if ax != None:
            ax.add_artist(p)
        return p

    # iterate through the anchor points
    for anchor in anchor_points:
        # initialize list that will store tuples of (sequence, scale)
        #   -> these are all sequences involved in the motiv with their corresponding scale variable
        all_scores = []

        # get ref positions involved in the motiv
        weight1, pos1 = math.modf(anchor)
        pos1 = int(pos1)
        pos2 = pos1 + 1
        weight2 = 1 - weight1

        # iterate over all kmer positions
        for i in range(kmer_size):
            # add empty list to seqs; this will store all letters and weights for the ith position of the motif
            all_scores.append([])

            # iterate over all kmers starting at pos1
            for idx_kmer in torch.nonzero(kmer_ref[:, pos1], as_tuple=False):
                # get the sequence corresponding to the index
                kmer_seq = list(kmer_dict.keys())[list(kmer_dict.values()).index(idx_kmer)]

                # get the weight of this specific kmer
                kmer_weight = kmer_ref[idx_kmer, pos1].item() * weight1

                # store sequence and weight as tuple
                does_exist = False
                for j in range(len(all_scores[i])):
                    if all_scores[i][j][0] == list(kmer_seq)[i]:
                        all_scores[i][j] = (list(kmer_seq)[i], all_scores[i][j][1] + kmer_weight)
                        does_exist = True

                if not does_exist:
                    all_scores[i].append((list(kmer_seq)[i], kmer_weight))

            # iterate over all kmers starting at pos2
            for idx_kmer in torch.nonzero(kmer_ref[:, pos2], as_tuple=False):
                # get the sequence corresponding to the index
                kmer_seq = list(kmer_dict.keys())[list(kmer_dict.values()).index(idx_kmer)]

                # get the weight of this specific kmer
                kmer_weight = kmer_ref[idx_kmer, pos2].item() * weight2

                # store sequence and weight as tuple but make sure that letters are not duplicated
                does_exist = False
                for j in range(len(all_scores[i])):
                    if all_scores[i][j][0] == list(kmer_seq)[i]:
                        all_scores[i][j] = (list(kmer_seq)[i], all_scores[i][j][1] + kmer_weight)
                        does_exist = True

                if not does_exist:
                    all_scores[i].append((list(kmer_seq)[i], kmer_weight))

        fig, ax = plt.subplots(figsize=(10, 3))

        x = 1
        maxi = 0
        for scores in all_scores:
            y = 0
            for base, score in scores:
                letterAt(base, x, y, score, ax)
                y += score
            x += 1
            maxi = max(maxi, y)

        plt.xticks(range(1, x))
        plt.xlim((0, x))
        plt.ylim((0, maxi))
        plt.tight_layout()
        plt.savefig(outdir + "/motif_anchor_" + str(anchor) + ".png")
        plt.close(fig)


def anchor_weight_matrix(anchors, kmer_ref, kmer_dict, sigma, num_best, viz=True):
    """Construction of the anchor weight matrix

    This function converts the learned anchors into a 2d matrix where each row represents one k-mer and each column
    represents a sequence position. This matrix can be visualized to get an image comparable to Figure 2 in
    Meinicke et al., 2004.

    - **Parameters**::

        :param anchors: List of learned anchor positions
            :type anchors: List of Floats
        :param kmer_ref: Reference sequence of the trained model
            :type kmer_ref: Tensor (num_k-mers x len_seq)
        :param kmer_dict: Dictionary mapping each k-mer to an index
            :type kmer_dict: Dictionary
        :param sigma: sigma value used for the trained model
            type sigma: Float
        :param num_best: Indicates how many of the most informative oligomers and positions should be showed
            :type num_best: Integer
        :param viz: Indicates whether the matrix should be visualized using matplotlib
            :type viz: Boolean

    - **Returns**::

        :returns: Matrix
            :rtype: 2d NumPy array
    """
    # import section
    from scipy.ndimage import gaussian_filter1d

    # initialize output and anchor weight matrix
    image_matrix = np.zeros((len(kmer_dict), kmer_ref.shape[1]))
    anchor_weight = np.zeros(kmer_ref.shape[1])

    # calculate the weights imposed by the anchors at each position
    for anchor in anchors:
        # get positions that are affected by the current anchor
        weight1, pos1 = math.modf(anchor)
        pos1 = int(pos1)
        pos2 = pos1 + 1
        weight2 = 1 - weight1

        # update anchor weights
        anchor_weight[pos1] += weight1
        anchor_weight[pos2] += weight2

    # iterate over all oligomers to calculate the weighted oligo functions
    for i in range(image_matrix.shape[0]):
        # get all position where the current oligomer is present in the reference sequence
        oligo_pos = torch.nonzero(kmer_ref[i, :]).numpy()
        oligo_pos_weights = kmer_ref[i, oligo_pos].numpy()

        # iterate over all sequence positions to calculate oligo function at each position
        for j in range(image_matrix.shape[1]):
            image_matrix[i, j] = anchor_weight[j] * np.sum(np.exp(-(1 / (2 * sigma ** 2)) * (oligo_pos - j) ** 2) *
                                                           oligo_pos_weights)

    # apply a Gaussian filter to incorporate the oligo kernel information
    image_matrix = gaussian_filter1d(image_matrix, sigma=sigma, mode='constant')

    # scale matrix between 0 and 1
    image_matrix /= image_matrix.max()

    # reduce noise in the matrix
    below_threshold = image_matrix < 0.1
    image_matrix[below_threshold] = 0

    # get row and column norm for ranking of oligomers and positions respectivly
    norm_oligomers = np.linalg.norm(image_matrix, axis=1)
    norm_positions = np.linalg.norm(image_matrix, axis=0)

    # sort oligomers and positions according to their norm
    idx_oliSort = np.flip(np.argsort(norm_oligomers)[-num_best:])
    idx_posSort = np.flip(np.argsort(norm_positions)[-num_best:])

    if viz:

        # import matplotlib to plot the matrix
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3)
        im = axs[0].imshow(image_matrix, interpolation=None, aspect='auto')
        fig.colorbar(im, ax=axs[0])
        axs[0].set_xlabel('Position')
        axs[0].set_xticks(np.arange(0, kmer_ref.size(1), 5))
        axs[0].set_yticks(np.arange(len(list(kmer_dict.keys()))))
        axs[0].set_yticklabels(list(kmer_dict.keys()))
        axs[0].set_title('k-mer weight functions')
        axs[1].bar(np.arange(num_best), norm_oligomers[idx_oliSort], 0.35)
        axs[1].set_xlabel('Oligomers')
        axs[1].set_ylabel('l2-Norm')
        axs[1].set_xticks(np.arange(num_best))
        axs[1].set_xticklabels([list(kmer_dict.keys())[i] for i in idx_oliSort])
        axs[1].set_title('Oligomer Ranking')
        axs[2].bar(np.arange(num_best), norm_positions[idx_posSort], 0.35)
        axs[2].set_xlabel('Positions')
        axs[2].set_ylabel('l2-Norm')
        axs[2].set_xticks(np.arange(num_best))
        axs[2].set_xticklabels([str(i) for i in idx_posSort])
        axs[2].set_title('Position Ranking')
        plt.show()

    return image_matrix, (norm_oligomers, idx_oliSort), (norm_positions, idx_posSort)


def model_interpretation(anchors, kmer_ref, kmer_dict, sigma, num_best, viz=True):
    """Visual interpretation of the learned model

    This function converts the learned anchors into a 2d matrix where each row represents one k-mer and each column
    represents a sequence position. This matrix can be visualized to get an image comparable to Figure 2 in
    Meinicke et al., 2004. Furthermore, oligomers and positions will be ranked based on their importance

    - **Parameters**::

        :param anchors: List of learned anchor positions
            :type anchors: List of Floats
        :param kmer_ref: Reference sequence of the trained model
            :type kmer_ref: Tensor (num_k-mers x len_seq)
        :param kmer_dict: Dictionary mapping each k-mer to an index
            :type kmer_dict: Dictionary
        :param sigma: sigma value used for the trained model
            type sigma: Float
        :param num_best: Indicates how many of the most informative oligomers and positions should be showed
            :type num_best: Integer
        :param viz: Indicates whether the matrix should be visualized using matplotlib
            :type viz: Boolean

    - **Returns**::

        :returns: Matrix
            :rtype: 2d NumPy array
    """
    # import section
    from scipy.ndimage import gaussian_filter1d

    # initialize output and anchor weight matrix
    image_matrix = np.zeros((len(kmer_dict), kmer_ref.shape[1]))
    anchor_weight = np.zeros(kmer_ref.shape[1])

    # calculate the weights imposed by the anchors at each position
    for anchor in anchors:
        # get positions that are affected by the current anchor
        weight1, pos1 = math.modf(anchor)
        pos1 = int(pos1)
        pos2 = pos1 + 1
        weight2 = 1 - weight1

        # update anchor weights
        anchor_weight[pos1] += weight1
        anchor_weight[pos2] += weight2

    # iterate over all oligomers to calculate the weighted oligo functions
    for i in range(image_matrix.shape[0]):
        # get all position where the current oligomer is present in the reference sequence
        aux_oli = np.cos((i / len(kmer_dict)) * np.pi)
        oligo_pos = [pos for pos in range(kmer_ref.shape[1]) if math.isclose(aux_oli, kmer_ref[0, pos].item(),
                                                                             rel_tol=1e-6)]
        oligo_pos = np.array(oligo_pos)

        # iterate over all sequence positions to calculate oligo function at each position
        for j in range(image_matrix.shape[1]):
            image_matrix[i, j] = anchor_weight[j] * np.sum(np.exp(-(1 / (2 * sigma**2)) * (oligo_pos - j)**2))

    # apply a Gaussian filter to incorporate the oligo kernel width information
    image_matrix = gaussian_filter1d(image_matrix, sigma=sigma, mode='constant')

    # scale matrix between 0 and 1
    image_matrix /= image_matrix.max()

    # reduce noise in the matrix
    below_threshold = image_matrix < 0.1
    image_matrix[below_threshold] = 0

    # get row and column norm for ranking of oligomers and positions respectivly
    norm_oligomers = np.linalg.norm(image_matrix, axis=1)
    norm_positions = np.linalg.norm(image_matrix, axis=0)

    # sort oligomers and positions according to their norm
    idx_oliSort = np.flip(np.argsort(norm_oligomers)[-num_best:])
    idx_posSort = np.flip(np.argsort(norm_positions)[-num_best:])

    if viz:

        # import matplotlib to plot the matrix
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3)
        im = axs[0].imshow(image_matrix, interpolation=None, aspect='auto')
        fig.colorbar(im, ax=axs[0])
        axs[0].set_xlabel('Position')
        axs[0].set_xticks(np.arange(0, kmer_ref.size(1), 5))
        axs[0].set_yticks(np.arange(len(list(kmer_dict.keys()))))
        axs[0].set_yticklabels(list(kmer_dict.keys()))
        axs[0].set_title('k-mer weight functions')
        axs[1].bar(np.arange(num_best), norm_oligomers[idx_oliSort], 0.35)
        axs[1].set_xlabel('Oligomers')
        axs[1].set_ylabel('l2-Norm')
        axs[1].set_xticks(np.arange(num_best))
        axs[1].set_xticklabels([list(kmer_dict.keys())[i] for i in idx_oliSort])
        axs[1].set_title('Oligomer Ranking')
        axs[2].bar(np.arange(num_best), norm_positions[idx_posSort], 0.35)
        axs[2].set_xlabel('Positions')
        axs[2].set_ylabel('l2-Norm')
        axs[2].set_xticks(np.arange(num_best))
        axs[2].set_xticklabels([str(i) for i in idx_posSort])
        axs[2].set_title('Position Ranking')
        plt.show()

    return image_matrix, (norm_oligomers, idx_oliSort), (norm_positions, idx_posSort)


def category_from_output(output):
    """Output selector

    This helper function returns the class with highest probability from the CON output.

    - **Parameters**::

        :param output: Output of a CON network.
            :type output: tensor

    - **Returns**::

        :return category_i: Index of the category with the highest probability.
            :rtype category_i: integer
    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i


def gaussian_filter_1d(size, sigma=None):
    """1D Gaussian filter

    This function creates a 1D Gaussian filter mask used for pooling in a CKN layer.

    - **Parameters**::

        :param size: Size of the pooling filter
            :type size: Integer
        :param sigma: Parameter of the Gaussian function
            :type sigma: Float

    - **Returns**::

        :return Mask for a pooling layer based on a Gaussian function
            :rtype Tensor
    """
    # if the filter is of size 1, no Gaussian is needed
    if size == 1:
        return torch.ones(1)

    # if sigma is not specified, initialize sigma dependent on the filter size
    if sigma is None:
        sigma = (size - 1.)/(2.*math.sqrt(2))

    # build the filter mask
    m = float((size - 1) // 2)
    filt = torch.arange(-m, m+1)
    filt = torch.exp(-filt.pow(2)/(2.*sigma*sigma))
    return filt/torch.sum(filt)


def normalize_(x, p=2, dim=-1):
    """Matrix Normalization

    Auxiliary function implementing numerically stable matrix normalization.

    :param x: Input tensor
    :param p: Indicates the order of the norm
    :param dim: If it is an int, vector norm will be calculated, if it is 2-tuple of ints, matrix norm will be
                calculated. If the value is None, matrix norm will be calculated when the input tensor only has two
                dimensions, vector norm will be calculated when the input tensor only has one dimension. If the input
                tensor has more than two dimensions, the vector norm will be applied to last dimension.
    :return: Normalized input tensor
    """
    norm = x.norm(p=p, dim=dim, keepdim=True)
    x.div_(norm.clamp(min=EPS))
    return x


#############################################################################
# KERNEL FUNCTIONS
#############################################################################


def exp_func(x, sigma=1, scale=1):
    """ Exponential part of the oligo kernel

    This helper function implements the exponential part of the oligo kernel function. The scaling parameter can be
    set to accommodate for the difference between the oligo kernel and the oligo kernel network.

    - **Parameters**::

        :param x: Input to the oligo kernel function.
            :type x: Float
        :param sigma: Degree of positional uncertainty.
            :type sigma: Float
        :param scale: Scaling parameter to accommodate for the oligo kernel network formulation.
            :type scale: Float

    - **Returns**::

        :returns same shape tensor as x
    """
    return torch.exp(scale/(2*sigma**2) * (x-1.))


def exp_func2(x, alpha):
    """Element wise non-linearity
    kernel_exp is defined as k(x)=exp(alpha * (x-1))
    return:
        same shape tensor as x
    """
    return torch.exp(alpha*(x - 1.))


def add_exp(x, alpha):
    return 0.5 * (exp_func2(x, alpha) + x)


def exp_oli(x, y, length=1, sigma=1, scale=1, alpha=10000):
    """Exponential activation function for the oligo layer

    This helper function implements the exponential activation function used in the oligo layer kernel function
    formulation.

    - **Parameters**::

        :param x: Input (convolution of input position with anchor points) to the oligo kernel function.
            :type x: Tensor
        :param y: Input (convolution of oligomer encoding tensors) to the oligo kernel function
            :type y: Tensor
        :param length: Length of the oligomers.
            :type length: Int
        :param sigma: Degree of positional uncertainty.
            :type sigma: Float
        :param scale: Scaling parameter to accommodate for the oligo kernel network formulation.
            :type scale: Float
        :param alpha: Parameter to switch off unwanted position pairings
            :type alpha: Float

    - **Returns**::

        :returns same shape tensor as x
    """
    return torch.exp((alpha**2 * (y-length)) + (scale/(2*sigma**2) * (x-1.)))


# dictionary for easy mapping of kernel functions
#   - if one wants to implement new kernel functions write the function definition directly above this dictionary
#     and add an entry to the dictionary
kernels = {
    "exp": exp_func,
    "exp_chen": exp_func2,
    "add_exp_chen": add_exp,
    "exp_oli": exp_oli
}


#############################################################################
# PYTORCH MODULS (LOSS, AUTOGRAD EXTENSION, ETC)
#############################################################################


class ClassBalanceLoss(nn.Module):
    """Implementation of the Class-Balance Loss

    Reference: Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie; Proceedings of the IEEE/CVF Conference on
               Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268-9277
    """
    def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma, reduction='mean'):
        """Constructor of the class-balance loss class

        - **Parameters**::

            :param samples_per_cls: List containing the number of samples per class in the dataset
                :type samples_per_cls: List of Integers
            :param no_of_classes: Number of classes
                :type no_of_classes: Integer
            :param loss_type: Loss function used for the class-balance loss
                :type loss_type: String
            :param beta: Hyperparameter for Class balanced loss.
                :type beta: Float
            :param gamma: Hyperparameter for Focal loss
                :type gamma: Float
        """
        # call constructor of parent class
        super(ClassBalanceLoss, self).__init__()

        # check whether the parameters are valid
        if len(samples_per_cls) != no_of_classes:
            raise ValueError('Dimensionality of first argument expected to be {}. Found {} instead!'.format(
                no_of_classes, len(samples_per_cls)))

        # store user-specified parameters
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def focal_loss(self, labels, logits, alpha):
        """Compute the focal loss between `logits` and the ground truth `labels`.

        Focal loss = -alpha_t * (1-pt)^gamma * log(pt), where pt is the probability of being classified to the true
        class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

        - **Parameters**::

            :param labels: Tensor containing the true labels
                :type labels: float tensor of size [batch, num_classes].
            :param logits: Tensor containing the output of the network
                :type logits: float tensor of size [batch, num_classes].
            :param alpha: Tensor specifying per-example weight for balanced cross entropy.
                :type alpha: float tensor of size [batch_size]

        - **Returns**::

            :returns a float32 scalar representing normalized total loss.
        """
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

    def forward(self, logits, labels):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        - **Parameters**::

            :param logits: Tensor containing the output of the network
                :type logits: float tensor of size [batch, num_classes].
            :param labels: Tensor containing the true labels
                :type labels: float tensor of size [batch].

        - **Returns**::

            :returns a float tensor representing class balanced loss
        """
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        # we need to adapt the dimensionality of logits if the batch size is 1
        #   -> otherwise logits and labels_one_hot have mismatching dimensionality
        if labels_one_hot.shape[0] == 1:
            logits = logits.view_as(labels_one_hot)

        weights_tensor = labels_one_hot.new_tensor(weights)
        weights_tensor = weights_tensor.unsqueeze(0)
        weights_tensor = weights_tensor.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights_tensor = weights_tensor.sum(1)
        weights_tensor = weights_tensor.unsqueeze(1)
        weights_tensor = weights_tensor.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights_tensor)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights_tensor,
                                                         reduction=self.reduction)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights_tensor,
                                             reduction=self.reduction)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels, weight=torch.tensor(weights).float(),
                                      reduction=self.reduction)
        else:
            raise ValueError("Undefined loss function: {}.".format(self.loss_type) +
                             "\n            Valid values are 'focal', 'sigmoid', 'softmax', and 'cross_entropy'.")
        return cb_loss


"""
class MatrixInverseSqrt(torch.autograd.Function):
    Matrix inverse square root for a symmetric definite positive matrix
    
    @staticmethod
    def forward(ctx, input, eps=1e-2):
        use_cuda = input.is_cuda
        input = input.cpu()
        e, v = torch.symeig(input, eigenvectors=True)
        if use_cuda:
            e = e.cuda()
            v = v.cuda()
        e = e.clamp(min=0)
        e_sqrt = e.sqrt_().add_(eps)
        ctx.e_sqrt = e_sqrt
        ctx.v = v
        e_rsqrt = e_sqrt.reciprocal()

        output = v.mm(torch.diag(e_rsqrt).mm(v.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        e_sqrt, v = Variable(ctx.e_sqrt), Variable(ctx.v)
        ei = e_sqrt.expand_as(v)
        ej = e_sqrt.view([-1, 1]).expand_as(v)
        f = torch.reciprocal((ei + ej) * ei * ej)
        grad_input = -v.mm((f*(v.t().mm(grad_output.mm(v)))).mm(v.t()))
        return grad_input, None
"""

class MatrixInverseSqrt(torch.autograd.Function):
    """Matrix inverse square root for a symmetric definite positive matrix
    """
    @staticmethod
    def forward(ctx, input, eps=1e-2):
        dim = input.dim()
        ctx.dim = dim
        use_cuda = input.is_cuda
        if input.size(0) < 300:
            input = input.cpu()
        e, v = torch.symeig(input, eigenvectors=True)
        if use_cuda and input.size(0) < 300:
            e = e.cuda()
            v = v.cuda()
        e = e.clamp(min=0)
        e_sqrt = e.sqrt_().add_(eps)
        ctx.save_for_backward(e_sqrt, v)
        e_rsqrt = e_sqrt.reciprocal()

        if dim > 2:
            output = v.bmm(v.permute(0, 2, 1) * e_rsqrt.unsqueeze(-1))
        else:
            output = v.mm(v.t() * e_rsqrt.view(-1, 1))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        e_sqrt, v = ctx.saved_variables
        if ctx.dim > 2:
            ei = e_sqrt.unsqueeze(1).expand_as(v)
            ej = e_sqrt.unsqueeze(-1).expand_as(v)
        else:
            ei = e_sqrt.expand_as(v)
            ej = e_sqrt.view(-1, 1).expand_as(v)
        f = torch.reciprocal((ei + ej) * ei * ej)
        if ctx.dim > 2:
            vt = v.permute(0, 2, 1)
            grad_input = -v.bmm((f*(vt.bmm(grad_output.bmm(v)))).bmm(vt))
        else:
            grad_input = -v.mm((f*(v.t().mm(grad_output.mm(v)))).mm(v.t()))
        return grad_input, None


def matrix_inverse_sqrt(input, eps=1e-2):
    """Wrapper for MatrixInverseSqrt"""
    return MatrixInverseSqrt.apply(input, eps)


#############################################################################
# KMEANS+***
#############################################################################


def init_kmeans(x, n_clusters, n_local_trials=None, use_cuda=False):
    n_samples, n_features = x.size()
    clusters = torch.Tensor(n_clusters, n_features)
    if use_cuda:
        clusters = clusters.cuda()

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))
    clusters[0] = x[np.random.randint(n_samples)]

    closest_dist_sq = 1 - clusters[[0]].mm(x.t())
    closest_dist_sq = closest_dist_sq.view(-1)
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(-1), rand_vals)
        distance_to_candidates = 1 - x[candidate_ids].mm(x.t())

        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = torch.min(closest_dist_sq,
                                    distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        clusters[c] = x[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return clusters


def spherical_kmeans(x, n_clusters, max_iters=100, verbose=True,
                     init=None, eps=1e-4):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x n_features): data points
        n_clusters (int): number of clusters
    """
    use_cuda = x.is_cuda
    n_samples, n_features = x.size()
    if init == "kmeans++":
        print(init)
        clusters = init_kmeans(x, n_clusters, use_cuda=use_cuda)
    else:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]

    prev_sim = np.inf

    for n_iter in range(max_iters):
        # assign data points to clusters
        cos_sim = x.mm(clusters.t())
        tmp, assign = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                # clusters[j] = x[random.randrange(n_samples)]
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm()

        if np.abs(prev_sim - sim)/(np.abs(sim)+1e-20) < 1e-6:
            break
        prev_sim = sim
    return clusters


#############################################################################
# METRICES
#############################################################################


def bootstrap_auc(y_true, y_pred, ntrial=10):
    n = len(y_true)
    aucs = []
    for t in range(ntrial):
        sample = np.random.randint(0, n, n)
        try:
            auc = roc_auc_score(y_true[sample], y_pred[sample])
        except:
            return np.nan, np.nan  # If any of the samples returned NaN, the whole bootstrap result is NaN
        aucs.append(auc)
    return np.mean(aucs), np.std(aucs)


def recall_at_fdr(y_true, y_score, fdr_cutoff=0.05):
    # convert y_true and y_score into desired format
    #   -> both have to be lists of shape [nb_samples]
    y_true_new = y_true.argmax(axis=1)
    y_score_new = [y_score[j][i] for j, i in enumerate(y_true_new)]
    precision, recall, thresholds = precision_recall_curve(y_true_new, y_score_new)
    fdr = 1 - precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]


def xroc(res, cutoff):
    """
    :type res: List[List[label, score]]
    :type cutoff: all or 50
    """
    area, height, fp, tp = 0.0, 0.0, 0.0, 0.0
    for x in res:
        label = x
        if cutoff > fp:
            if label == 1:
                height += 1
                tp += 1
            else:
                area += height
                fp += 1
        else:
            if label == 1:
                tp += 1
    lroc = 0
    if fp != 0 and tp != 0:
        lroc = area / (fp * tp)
    elif fp == 0 and tp != 0:
        lroc = 1
    elif fp != 0 and tp == 0:
        lroc = 0
    return lroc


def get_roc(y_true, y_pred, cutoff):
    score = []
    label = []

    for i in range(y_pred.shape[0]):
        label.append(y_true[i])
        score.append(y_pred[i])

    index = np.argsort(score)
    index = index[::-1]
    t_score = []
    t_label = []
    for i in index:
        t_score.append(score[i])
        t_label.append(label[i])

    score = xroc(t_label, cutoff)
    return score


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metric = {}
    metric['log.loss'] = log_loss(y_true, y_pred)
    metric['accuracy'] = accuracy_score(y_true, y_pred > 0.5)
    metric['F_score'] = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    metric['auROC'] = roc_auc_score(y_true, y_pred)
    metric['auROC50'] = roc_auc_score(y_true, y_pred, max_fpr=0.5)
    metric['auPRC'] = average_precision_score(y_true, y_pred)
    metric['recall_at_10_fdr'] = recall_at_fdr(y_true, y_pred, 0.10)
    metric['recall_at_5_fdr'] = recall_at_fdr(y_true, y_pred, 0.05)
    metric["pearson.r"], metric["pearson.p"] = stats.pearsonr(y_true.ravel(), y_pred.ravel())
    metric["spearman.r"], metric["spearman.p"] = stats.spearmanr(y_true, y_pred, axis=None)
    df = pd.DataFrame.from_dict(metric, orient='index')
    df.columns = ['value']
    df.sort_index(inplace=True)
    return df


#############################################################################
# DEBUG UTILITIES
#############################################################################


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        print(queue)
        callback(fn)

def register_hooks(var):
    # import Digraph
    from graphviz import Digraph

    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return torch.isnan(grad_output).any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, "fn = {}\nkeys in fn_dict = {}".format(fn, fn_dict.keys())
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook:
    def __init__(self, module, backward=False):
        # initialize parameters
        self.input = None
        self.output = None

        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def plot_grad_flow(named_parameters, zoom=False):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    # import needed packages
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    if zoom:
        # zoom in on the lower gradient regions
        plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
