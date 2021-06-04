################################################
# Python file containing functions and classes #
# that are needed for the interpretability of  #
# a trained CON model                          #
#                                              #
# Author: Jonas Ditz                           #
# Contact: ditz@informatik.uni-tuebingen.de    #
################################################

import math

import torch
from torch.autograd import Variable
import numpy as np
from scipy import optimize
from scipy.stats import chi2

from .data_utils import ALPHABETS


def seq2pwm(seq, alphabet='DNA'):
    """This function takes sequences and an alphabet and returns the position weight matrix (PWM) of the set of
    sequences. If only one sequence is provided, the one-hot encoding matrix of the sequence will be returned.

    - **Parameters**::

        :param seq: Input sequences
            :type seq: List of Strings or
                       m x n ndarray (m sequences of length n), each letter is represented by their position in the
                       alphabet (i.e. a number between 0 and |alphabet|-1)
        : param alphabet: The alphabet that was used to build the input sequences
            :type alphabet: String
    """

    # get the alphabet
    alphabet = ALPHABETS[alphabet][0]

    # if the sequences are given as a list of strings (or it is a single string), convert them into an ndarray
    if isinstance(seq, str):
        x = np.zeros((1, len(seq)), dtype=np.int8)
        for i, letter in enumerate(seq):
            x[0, i] = alphabet.index(letter)

    elif all(isinstance(item, str) for item in seq):
        # all sequences have to have the same length
        if not all(len(item) == len(seq[0]) for item in seq):
            raise ValueError('All sequences must be of same length')

        x = np.zeros((len(seq), len(seq[0])), dtype=np.int8)
        for i, s in enumerate(seq):
            for j, letter in enumerate(s):
                x[i, j] = alphabet.index(letter)

    elif isinstance(seq, np.ndarray):
        x = seq

    else:
        raise ValueError('Unknown type of input seq: {}'.format(type(seq)))

    # create the position frequency matrix (PFM)
    #   -> this will be used to calculate the PWM
    pfm = np.zeros((len(alphabet), x.shape[1]))

    # iterate over the length of the sequence
    for i in range(x.shape[1]):
        # count the frequency of each character at the current position
        values, counts = np.unique(x[:, i], return_counts=True)

        # insert frequency of each character into the pfm
        for j, v in enumerate(values):
            pfm[v, i] = counts[j]

    # calculate the PWM
    pwm = pfm / np.sum(pfm, axis=0)
    return pwm


def optimize_anchor(anchor_idx, con_layer, max_iter):
    """This function uses a gradient based optimization algorithm to find the PWM-position pair that is closest to
    the specified anchor under the geometry induced by the CON layer's kernel

    - **Parameter**::

        :param anchor_idx: Index of the anchor point (oligomer-position pair) used for the optimization
            :type anchor_idx: Integer
        :param con_layer: CON layer of a trained CON model
            :type con_layer: con.CONLayer
        :param max_iter: Maximal number of iterations during the optimization
            :type max_iter: Integer

    - **Returns**::

        :returns: The optimized PWM-position pair together with the loss
    """
    pass


def compute_pwm_position(model, max_iter=2000):
    """This function takes a trained CON model and returns the PWM-position pairs that are closest to each anchor point
    under the geometry induced by the convolutional oligo kernel.

    - **Parameter**::

        :param model: Trained CON model
            :type model: con.CON
        :param max_iter: Maximal number of iterations during the optimization
    """


def model_interpretation(seq, anchors_oli, anchors_pos, alphabet, sigma, alpha, num_best=-1, viz=True):
    """Visual interpretation of the learned model

    This function converts the learned anchors into a 2d matrix where each row represents one motif and each column
    represents a sequence position. This matrix can be visualized to get an image comparable to Figure 2 in
    Meinicke et al., 2004. Furthermore, oligomers and positions will be ranked based on their importance

    - **Parameters**::

        :param seq: Input sequence for which the trained model should be visualized
            :type seq: String or One-Hot Encoding matrix
        :param anchors_oli: Learned anchor motifs
            :type anchors_oli: ndarray (nb_anchors x |alphabet| x k)
        :param anchors_pos: Learned anchor positions
            :type anchors_pos: ndarray (nb_anchors x 2 x 1)
        :param alphabet: Alphabet of the sequences used for training the model
        :param sigma: sigma value used for the trained model
            type sigma: Float
        :param alpha: Scaling parameter for the oligomer/motif comparison kernel
            :type alpha: Float
        :param num_best: Indicates how many of the most informative anchors should be showed. All anchors are included,
                         if set to -1.
            :type num_best: Integer
        :param viz: Indicates whether the matrix should be visualized using matplotlib
            :type viz: Boolean

    - **Returns**::

        :returns: Matrix
            :rtype: 2d NumPy array
    """
    # if the sequence is given as a string, convert it into a one-hot encoding matrix
    if isinstance(seq, str):
        seq = seq2pwm(seq, alphabet)

    # perform zero padding to the end of the sequence to prevent out-of-bound errors
    kmer_size = anchors_oli.shape[-1]
    seq_len = seq.shape[-1]

    # perform padding if necessary
    if kmer_size > 1:
        seq = np.concatenate((seq, np.zeros((seq.shape[0], kmer_size - 1))), axis=1)

    # convert anchor position vectors into sequence positions
    positions = (np.arccos(np.reshape(anchors_pos, (-1, 2))[:, 0]) / np.pi) * (seq_len - 1)

    # initialize the vizualisation matrix
    image_matrix = np.zeros((anchors_oli.shape[0], seq_len))

    # iterate over each anchor point
    for i in range(anchors_oli.shape[0]):

        # convert anchor position into discrete sequence positions with corresponding weights
        weight1, pos1 = math.modf(positions[i])
        pos1 = int(pos1)
        pos2 = pos1 + 1
        weight2 = 1 - weight1

        # create the sequence motif of the input sequence at the current anchor position
        seq_motif = weight1 * seq[:, pos1:pos1+kmer_size] + weight2 * seq[:, pos2:pos2+kmer_size]

        # get the current anchor motif
        anchor_motif = np.reshape(anchors_oli[i, :, :], (anchors_oli.shape[1], kmer_size))

        # iterate over each sequence position
        for j in range(seq_len):

            # calculate the oligo function for the current anchor point at the current sequence position
            image_matrix[i, j] = np.exp(alpha * (np.vdot(anchor_motif, seq_motif) - kmer_size) -
                                        (1/(np.pi * sigma**2)) * ((j - positions[i])**2))

    # scale matrix between 0 and 1
    image_matrix /= image_matrix.max()

    # reduce noise in the matrix
    below_threshold = image_matrix < 0.1
    image_matrix[below_threshold] = 0

    if num_best == -1:
        num_best = anchors_oli.shape[0]

    # get row and column norm for ranking of oligomers and positions respectivly
    norm_oligomers = np.linalg.norm(image_matrix, axis=1)
    norm_positions = np.linalg.norm(image_matrix, axis=0)

    # sort oligomers and positions according to their norm
    idx_oliSort = np.flip(np.argsort(norm_oligomers)[-num_best:])
    idx_posSort = np.flip(np.argsort(norm_positions)[-num_best:])

    if viz:

        # import matplotlib to plot the matrix
        import matplotlib.pyplot as plt

        if num_best != anchors_oli.shape[0]:
            image_matrix = image_matrix[idx_oliSort, :]

        plt.figure()
        plt.imshow(image_matrix, interpolation=None, aspect='auto')
        plt.colorbar()
        plt.xlabel('Sequence Position')
        plt.ylabel('Anchor')
        plt.yticks(range(num_best), idx_oliSort)
        plt.show()

    return image_matrix, (norm_oligomers, idx_oliSort), (norm_positions, idx_posSort)


def anchors_to_motivs(anchor_points, positions, type="DNA_FULL", outdir=""):
    """Motiv creation

    This function takes a list of anchor points, learned by the oligo kernel layer of a CON, and returns the motivs that
    are represented by these anchor points.

    ATTENTION: Motifs generated by this function are not meant for scientific publications!

    - **Parameters**::

        :param anchor_points: Tensor containing the oligomers corresponding to each anchor points, learned by an oligo
                              kernel layer (i.e. the tensor that can be accessed by model.oligo.weight)
            :type anchor_points: Tensor
        :param positions: List of positions of each oligomer
            :type positions: List
    """
    # import needed libraries
    import matplotlib as mpl
    from matplotlib.text import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt

    from .data_utils import ALPHABETS

    # initialize needed functionality
    fp = FontProperties(family="Arial", weight="bold")
    globscale = 1.35
    if type.split('_')[0] == 'DNA':
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
    elif type.split('_')[0] == 'PROTEIN':
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
    for anchor in range(anchor_points.shape[0]):
        # initialize list that will store tuples of (sequence, scale)
        #   -> these are all sequences involved in the motiv with their corresponding scale variable
        all_scores = []

        # iterate over the length of the oligomer
        for pos in range(anchor_points.shape[2]):
            # all characters contributing to the current oligomer position will be stored as a list of
            # (character, scale)-tupels
            scores = []

            # iterate over all characters in the alphabet and check if they contribute to the current position of the
            # anchor oligomer
            for char in range(anchor_points.shape[1]):
                if anchor_points[anchor, char, pos] != 0:
                    scores.append((ALPHABETS[type][0][char], anchor_points[anchor, char, pos].cpu().numpy()))

            all_scores.append(scores)

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
        plt.savefig(outdir + "/motif_anchor_" + str(positions[anchor]).zfill(3) + ".png")
        plt.close(fig)
