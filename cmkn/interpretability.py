"""Module that contains functions used to interpret a trained CMKN model.

Authors:
    Jonas C. Ditz: jonas.ditz@uni-tuebingen.de
"""

import os
import math

import numpy as np

from .data_utils import ALPHABETS


def seq2pwm(seq, alphabet='DNA_FULL'):
    """Converts sequences into a position weight matrix (PWM).

    This function takes sequences and an alphabet and returns the position weight matrix (PWM) of the set of
    sequences. If only one sequence is provided, the one-hot encoding matrix of the sequence will be returned.

    Args:
        seq: Input to the function which can be given as a single sequence provided as a :obj:`str`, a set of sequences
            provided as a :obj:`list` of :obj:`str`, or a NumPy ndarray of dimension m x n, where m is the number of
            sequences and n is the length of each sequence. If the input is a ndarray, each letter must be represented
            by theír position in the alphabet (i.e. a number between 0 and len(alphabet)-1)
        alphabet (:obj:`str`): The alphabet that was used to build the input sequences. Defaults to 'DNA_FULL'.

    Returns:
        The position weight matrix of the input sequences. If a single sequence was given, this will be the one-hot
        encoding matrix of that sequence.

    Raises:
        ValueError: If sequences are of different length
        ValueError: If the input was neither a :obj:`str`, a :obj:`list` of :obj:`str`, nor a ndarray
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


def model_interpretation(seq, anchors_oli, anchors_pos, alphabet, sigma, alpha, beta, num_best=-1, viz=True, norm=True,
                         threshold=0.1):
    """Visual interpretation of the learned model

    This function converts the learned anchors into a 2d matrix where each row represents one motif and each column
    represents a sequence position. Each row is the discretized motif function of the corresponding anchor point over
    the whole input sequence. This matrix can be visualized to get an image comparable to Figure 2 in
    Meinicke et al., 2004. Furthermore, oligomers and positions will be ranked based on their importance. Importance
    will be assessed using the l2-norm of corresponding columns or rows, respectively.

    Args:
        seq (:obj:`str`): Input sequence for which the trained model should be visualized
        anchors_oli (ndarray): Learned anchor motifs. Alternatively, this can be a numpy array representing a list of
            motifs.
        anchors_pos (ndarray): Learned anchor positions. Alternatively this can be a numpy array storing a sequence
            position (given as a float) for each motif given with the input for anchors_oli.
        alphabet (:obj:`str`): Alphabet of the sequences used for training the model
        sigma (:obj:`float`): sigma value used for the trained model
        alpha (:obj:`float`): Scaling parameter for the oligomer/motif comparison kernel
        beta (:obj:`float`): Scaling parameter for position comparision term
        num_best (:obj:`int`): Indicates how many of the most informative anchors should be showed. All anchors
            are included, if set to -1.
        viz (:obj:`bool`): Indicates whether the matrix should be visualized using matplotlib
        norm (:obj:`bool`): Indicates whether the matrix should be normalized between zero and one. Defaults to True.
        threshold (:obj:`float`): All values of the matrix that are smaller than this value will be set to zero in
            order to denoise the image. If no denoising should be performed, set threshold to zero. Defaults to 0.1.

    Returns:
        NumPy ndarray which will be the discretized motif function for each anchor point over the whole input sequence.
    """
    # if the sequence is given as a string, convert it into a one-hot encoding matrix
    aux = seq
    if isinstance(seq, str):
        seq = seq2pwm(seq, alphabet)

    # perform zero padding to the end of the sequence to prevent out-of-bound errors
    kmer_size = anchors_oli.shape[-1]
    seq_len = seq.shape[-1]

    # set beta to the value dependend on the sequence length of non is specified
    if beta == -1 or beta == None:
        beta = seq_len**2 / 10

    # perform padding if necessary
    if kmer_size > 1:
        seq = np.concatenate((seq, np.zeros((seq.shape[0], kmer_size - 1))), axis=1)

    # convert anchor position vectors into sequence positions
    #if len(anchors_pos.shape) == 1:
    #    positions = anchors_pos
    #elif len(anchors_pos.shape) == 2:
    #    positions = (np.arccos(np.reshape(anchors_pos, (-1, 2))[:, 0]) / np.pi) * (seq_len - 1)
    #else:
    #    raise ValueError("Unexpected dimensionality of anchors_pos. Expected 1 or 2 dimensions, "
    #                     "got {} dimensions".format(len(anchors_pos.shape)))
    # if positions are given as plain sequence position, map them onto the upper half of the uni circle
    if len(anchors_pos.shape) == 1:
        positions = np.zeros((2, len(anchors_pos)))
        for i in range(len(anchors_pos)):
            positions[0, i] = np.cos((anchors_pos[i] / seq_len) * np.pi)
            positions[1, i] = np.sin((anchors_pos[i] / seq_len) * np.pi)
    elif len(anchors_pos.shape) == 2:
        positions = anchors_pos
        anchors_pos = (np.arccos(np.reshape(anchors_pos, (-1, 2))[:, 0]) / np.pi) * (seq_len - 1)
    else:
        raise ValueError("Unexpected dimensionality of anchors_pos. Expected 1 or 2 dimensions, "
                         "got {} dimensions".format(len(anchors_pos.shape)))

    # initialize the vizualisation matrix
    image_matrix = np.zeros((anchors_oli.shape[0], seq_len))

    # iterate over each anchor point
    for i in range(anchors_oli.shape[0]):

        # convert anchor position into discrete sequence positions with corresponding weights
        weight2, pos1 = math.modf(anchors_pos[i])
        pos1 = int(pos1)
        pos2 = pos1 + 1
        weight1 = 1 - weight2

        print(i, anchors_pos[i])

        # create the sequence motif of the input sequence at the current anchor position
        seq_motif = weight1 * seq[:, pos1:pos1+kmer_size] + weight2 * seq[:, pos2:pos2+kmer_size]

        # get the current anchor motif
        anchor_motif = np.reshape(anchors_oli[i, :, :], (anchors_oli.shape[1], kmer_size))

        # iterate over each sequence position
        for j in range(seq_len):

            #seq_motif = seq[:, j:j+kmer_size]
            aux_pos = np.array([np.cos(((j + 1) / seq_len) * np.pi), np.sin(((j + 1) / seq_len) * np.pi)])

            # calculate the motif function for the current anchor point at the current sequence position
            image_matrix[i, j] = np.exp(alpha/2 * (np.vdot(anchor_motif, seq_motif) - kmer_size) -
                                        (beta/(2 * sigma**2)) * (np.linalg.norm(aux_pos - positions[:, i])**2))

    # scale matrix between 0 and 1
    if norm:
        image_matrix /= image_matrix.max()

    # reduce noise in the matrix
    below_threshold = image_matrix < threshold
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


def anchors_to_motifs(anchor_points, type="DNA_FULL", outdir="", eps=1e-4):
    """Motif image creation

    This function takes a list of anchor points, learned by the oligo kernel layer of a CON, and returns the motifs that
    are represented by these anchor points.

    ATTENTION: Motifs generated by this function are not meant for scientific publications!

    Args:
        anchor_points (Tensor): Tensor containing the oligomers corresponding to each anchor points, learned by an motif
            kernel layer (i.e. the tensor that can be accessed by model.oligo.weight)
        type (:obj:`str`): Specifies whether the result will be a DNA or a Protein motif. Defaults to 'DNA'
        outdir (:obj:`str`): Destination where the motifs will be stored. If an empty string is given, motifs will not
            be stored but displayed instead. Defaults to an empty string.
        eps (:obj:`float` or :obj:`int`): If a floating point number is given, values below this threshold will be set
            to 0. If an integer is given, only the top k values will be used in the motif (k is the integer provided).
            This argument can be used to denoise motifs or display only the most informative parts.

    Raises:
        ValueError: If an unknown alphabet was chosen.
    """
    # import needed libraries
    import matplotlib as mpl
    from matplotlib.text import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt

    from .data_utils import ALPHABETS

    # if the outdir does not exist, create it
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            pass

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
                   "X": TextPath((-0.35, 0), "X", size=1, prop=fp)}
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
        raise ValueError('Unknown alphabet type: {}!'.format(type.split('_')[0]))

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
        print('creating motif for anchor {}...'.format(anchor))

        # store current anchor as a numpy array
        cur_anchor = anchor_points[anchor, :, :].cpu().numpy()

        # make sure that each column of the anchor has unit l1 norm
        aux_norm = np.linalg.norm(cur_anchor, ord=1, axis=0)
        aux_norm[aux_norm == 0] = 1e-6
        cur_anchor = cur_anchor / aux_norm

        # utelize the eps argument
        if isinstance(eps, int):
            # set each value to zero except the k biggest values
            cur_anchor = cur_anchor * (cur_anchor >= np.sort(cur_anchor, axis=0)[[-eps], :]).astype(int)
        else:
            # remove all values that are below the threshold
            cur_anchor[cur_anchor < eps] = 0

        # recalculate the norm of each column and make sure that each column of  the anchor is normalized to have unit
        # l1 norm. Otherwise, it would not be a valid PWM.
        aux_norm = np.linalg.norm(cur_anchor, ord=1, axis=0)
        aux_norm[aux_norm == 0] = 1e-6
        cur_anchor = cur_anchor / aux_norm

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
                if cur_anchor[char, pos] != 0:
                    scores.append((ALPHABETS[type][0][char], cur_anchor[char, pos]))

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
        plt.axis('off')
        if outdir != "":
            plt.savefig(outdir + str(anchor).zfill(3) + "-anchor.png", bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
