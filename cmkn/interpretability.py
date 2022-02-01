"""Module that contains functions used to interpret a trained CMKN model.

Authors:
    Jonas C. Ditz: jonas.ditz@uni-tuebingen.de
"""

import os
import math

import numpy as np
import torch

from .data_utils import ALPHABETS


def seq2ppm(seq, alphabet='DNA_FULL'):
    """Converts sequences into a position probability matrix (PPM).

    This function takes sequences and an alphabet and returns the position probability matrix (PPM) of the set of
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
    ppm = pfm / np.sum(pfm, axis=0)
    return ppm


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
        seq = seq2ppm(seq, alphabet)

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

            # seq_motif = seq[:, j:j+kmer_size]
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
                   "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
                   "N": TextPath((-0.35, 0), "N", size=1, prop=fp)}
        COLOR_SCHEME = {'G': 'orange',
                        'A': 'darkgreen',
                        'C': 'blue',
                        'T': 'red',
                        'U': 'red',
                        'N': 'black'}
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

        # utilize the eps argument
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


def get_weight_distribution(model_path, num_classes, seq_len, layers, viz_peaks=False, win_len=10):
    """Analysis of weight distribution

    This function loads a trained CMKN model and calculates the mean positive weight associated with each class for all
    sequence positions. Optionally, a sliding window approach is provided for peak detection.

    Args:
        model_path (:obj:`str`): Path to the trained CMKN model. The trained network has to be stored in a dictionary
            where the learned parameters are stored with the id 'state_dict'.
        num_classes (:obj:`int`): Number of classes in the classification problem.
        seq_len (:obj:`int`): Length of the sequences that were used to train the CMKN model.
        layers (:obj:`list` of :obj:`str`): List of the layer names that connect the output of the kernel layer to each
            class prediction. The layers has to be in reverse order as they are in the network, i.e. the first name
            should be that of the classification layer, etc.
        viz_peaks (:obj:`bool`): Indicates whether peaks computed with a sliding window approach should be displayed. If
            set to yes, a window of the specified size will be slided over the whole sequence length and the mean and
            std of the mean weights in that window will be calculated. If a position has a mean weight that is two times
            the std higher than the mean of the corresponding window, it will be marked as a peak.
        win_len (:obj:`int`): Length of the window used for the sliding window approach.

    Returns:
        A dictionary containing the indices of weights at each layer that positively contributes to a specific class.
        This return value is mostly interesting for the package function get_learned_motif and can be safely ignored by
        most users.
    """

    # load the trained CMKN model
    model = torch.load(model_path)
    model = model['state_dict']

    # auxiliary variables used for propagating class association through the network
    aux_weights = {}
    aux_values = [{} for _ in range(num_classes)]
    aux_mean = np.zeros([num_classes, seq_len])
    prev_layer = ''

    # iterate over each layer
    for i, layer in enumerate(layers):

        # if the current layer is the classification layer, get each input that is positively associated with each
        # class (i.e. that are connected with that class by a positive weight)
        if i == 0:
            # get the weights of the classification layer
            aux_weights[layer] = model[layer + '.weight'].cpu()

            # get positive indices of positive weights for each class
            for j in range(num_classes):
                aux_indices = aux_weights[layer][j, :] > 0
                aux_indices = aux_indices.nonzero()
                aux_values[j][layer] = aux_indices

            # store this layer's name to access the right indices at the next layer
            prev_layer = layer

        # do the same for each neurons that belong to layers that are neither the classification layer nor the last
        # layer before the kernel layer
        elif i != len(layers) - 1:
            raise NotImplementedError('Currently, the function get_weight_distribution only supports two layers.')
            # TODO: implement propagation of class association through each layer if there are more than two

            # store this layer's name to access the right indices at the next layer
            prev_layer = layer

        # if the current layer is the first after the kernel layer, calculate the mean positive weight
        else:
            # calculate mean weight for each class
            for j in range(num_classes):

                # go over each positions positively connected with the current class
                for idx in aux_values[j][prev_layer]:

                    # get the weights for the current index
                    current_weights = model[layer + '.weight'][idx.item(), :].view(-1, seq_len).cpu().numpy()

                    # iterate over sequence length and calculate mean weight at each position
                    for k in range(seq_len):
                        idx_positives = current_weights[:, k] > 0
                        aux_mean[j, k] += (np.mean(current_weights[idx_positives, k]) *
                                           aux_weights[prev_layer][j, idx.item()])

                # divide sum of weights by number of weights
                aux_mean[j, :] /= len(aux_values[j][prev_layer])

            # normalize by the number of layers to ensure comparability
            aux_mean *= len(layers)

    try:
        import matplotlib.pyplot as plt

        # get positions where the mean weight is at least two standard derivations higher than the mean within the
        # window position +- win_len
        peaks = [[]] * num_classes
        for i in range(num_classes):
            for j in range(seq_len):

                # skip position if the selected window size does not fit
                if j < win_len/2 or j > seq_len - (win_len/2 + 1):
                    continue

                # calculate mean and std of the current window
                window = aux_mean[i, j - int(win_len/2):j + int(win_len/2)]
                mean_window = np.mean(window)
                std_window = np.std(window)

                # the current position is a peak if the value is higher than the window's mean plus two times the std
                # of the window
                if aux_mean[i, j] > mean_window + 2 * std_window:
                    peaks[i].append(j)

        # If there are only two classes, plot weight distributions for negative class in blue and weight distribution
        # for positive class in red. A single plot will be used and the weight distribution for the negative class is
        # mirrored on the x-axis.
        if num_classes == 2:
            plt.figure()
            plt.plot(aux_mean[1, :], 'r')
            plt.plot(aux_mean[0, :] * -1, 'b')
            if viz_peaks:
                plt.scatter(peaks[1], [aux_mean[1, p] for p in peaks[1]], c='r', s=60, marker=(5, 2))
                plt.scatter(peaks[0], [aux_mean[0, p] * -1 for p in peaks[0]], c='b', s=60, marker=(5, 2))
            plt.title('Mean Weight Distribution')
            plt.show()

        # if there are more than two classes, plot weight distribution for each class in a separate plot
        else:
            fig, axs = plt.subplots(num_classes)
            for i in range(num_classes):
                axs[i].plot(aux_mean[i, :])
                if viz_peaks:
                    axs[i].scatter(peaks[i], [aux_mean[i, p] for p in peaks[i]], c='r', s=60, marker=(5, 2))
                axs[i].set_title('Class: {}'.format(i))
            plt.title('Mean Weight Distribution')
            plt.show()

    except ImportError:
        print("Cannot import matplotlib.pyplot")

    except Exception as e:
        print("Unexpected error while trying to plot mean weight distribution:")
        print(e)

    # return the weights of each motif at each sequence position
    return aux_values


def get_learned_motif(model_path, positions, num_classes, seq_len, layers, thld=None, viz=True, eps=1e-4):
    """Compute learned motif at specified sequence positions

    This function takes a trained CMKN model and a set of sequence positions and calculates the learned motif at the s
    specified positions. The learned motif is calculated by computing a weighted mean motif out of all the anchors with
    a positive contribution at the position. A positive contribution is defined by either having a weight that exceeds
    the specified threshold or having a weight that is 2 std higher than the mean weight at the position.

    Args:
        model_path (:obj:`str`): Path to the trained CMKN model. The trained network has to be stored in a dictionary
            where the learned parameters are stored with the id 'state_dict'.
        positions (:obj:`list` of :obj:`int`): Set of position for which the learned motif should be computed. The
            positions have to be 0-based (i.e. sequence position 1 will be given as 0).
        num_classes (:obj:`int`): Number of classes in the classification problem.
        seq_len (:obj:`int`): Length of the sequences that were used to train the CMKN model.
        layers (:obj:`list` of :obj:`str`): List of the layer names that connect the output of the kernel layer to each
            class prediction. The layers has to be in reverse order as they are in the network, i.e. the first name
            should be that of the classification layer, etc. The last layer in the list has to be the kernel layer.
        thld (:obj:`float`): Threshold used to decide which anchors should be included in the computation of the learned
            motif. Defaults to None.
        viz (:obj:`bool`): Indicates whether the learned motifs should be visualized with matplotlib or not. Defaults to
            True.
        eps (:obj:`float` or :obj:`int`): Parameter of the motif visualization function, will be ignored if viz is set
            to False. If a floating point number is given, values below this threshold will be set to 0. If an integer
            is given, only the top k values will be used in the motif (k is the integer provided). This argument can be
            used to denoise motifs or display only the most informative parts.

    Returns:
        A dictionary containing a Tensor of shape (number of positions, alphabet size, motif length) for each of the
        classes. The tensors are holding the learned motifs.
    """

    # if an integer/float is given as the threshold this will be used as a hard threshold
    if isinstance(thld, int) or isinstance(thld, float):
        threshold = thld
    else:
        thld = None

    # load the trained CMKN model
    model = torch.load(model_path)
    args = model['args']
    model = model['state_dict']

    # get the indices indicating which weights belong to which class
    aux_indices = get_weight_distribution(model_path, num_classes, seq_len, layers[:-1], False)

    # initiate tensor to hold the mean motif at each position
    aux_learned_motifs = {}
    for i in range(num_classes):
        aux_learned_motifs[i] = torch.zeros(len(positions), len(args.alphabet), args.kmer_size)

    # iterate over all specified positions
    for i, pos in enumerate(positions):

        # iterate over all classes to get the motifs for each class
        for j in range(num_classes):

            # initialize needed auxiliary variables
            aux_anchors = torch.empty(0)
            aux_anchors_weights = torch.empty(0)

            # iterate over all entries that belong to the current class
            for idx in aux_indices[j][layers[-3]]:

                # get the weights of the last layer before the kernel layer
                aux_weights = model[layers[-2] + '.weight'][idx.item(), :].view(-1, seq_len).cpu()

                # if no threshold was set by the user, include all anchors with a weight 2x STD over the mean value
                if thld is None:
                    aux_pos_weights = aux_weights[:, pos] > 0
                    aux_std_mean = torch.std_mean(aux_weights[aux_pos_weights, pos], dim=0, unbiased=True)
                    threshold = aux_std_mean[1].item() + 2 * aux_std_mean[0].item()

                # store each contributing anchor and the corresponding weight of that anchor
                valid_anchors = aux_weights[:, pos] >= threshold
                aux_anchors = torch.cat((aux_anchors, valid_anchors.nonzero().flatten()), 0)
                aux_anchors_weights = torch.cat((aux_anchors_weights, aux_weights[valid_anchors, pos].flatten() *
                                                 model[layers[-3] + '.weight'][j, idx.item()].cpu().item()), 0)

            # get the learned motif at the i-th position for the j-th class
            if len(aux_anchors) > 0:
                aux_motif = torch.mean(model[layers[-1] + '.weight'][aux_anchors.type(torch.long), :, :].cpu() *
                                       aux_anchors_weights.reshape(-1, 1, 1), dim=0)
                aux_learned_motifs[j][i, :, :] = aux_motif
            else:
                print('No learned motif at position {} for class {}'.format(i+1, j))

    if viz:
        # select the correct type of alphabet
        if len(args.alphabet) > 5:
            alphabet = 'PROTEIN_FULL'
        else:
            alphabet = 'DNA_FULL'

        # display learned motif for each position and each class
        for i, pos in enumerate(positions):
            for j in range(num_classes):
                # convert motifs into position weight matrices using a simple background model that assumes equal
                # propability of each symbol
                pwm_motifs = torch.log2(aux_learned_motifs[j][i, :, :].view(1, len(args.alphabet), args.kmer_size) /
                                        (1/len(args.alphabet))) * -1

                print('displaying learned motif for class {} at position {}...'.format(j, pos + 1))
                anchors_to_motifs(pwm_motifs, alphabet, eps=eps)

    return aux_learned_motifs


def visualize_kernel_activation(model, kernel_layer, sequence):
    """Visualization of the kernel layer activation.

    This function takes a trained CMKN model, the name of the kernel layer, and an input sequence as a one-hot encoding
    vector and visualizes the activation of the kernel layer for inspection.

    Args:
        model (trained CMKN model): A trained CMKN model whose kernel layer activation will be visualized.
        kernel_layer ('obj':`str`): The kernel layer's name that will be visualized. It is possible ot visualize several
        different kernel layers by providing the names as a list of string.
        sequence (Tensor): The sequence for which the activation of the kernel layer will be visualized. This sequence
        has to be presented in form of a one-hot encoded Tensor. It is possible to visualize the activation of several
        sequences at once by providing the Tensors in batch form.
    """
    # initialize the dict that holds the activations
    activations = {}

    # determine the number of sequences given
    nb_seq = sequence.shape[0]

    # define the hook function for the forward hooks
    def get_activation(l_name):
        def hook(module, input, output):
            activations[l_name] = output.detach()
        return hook

    # make sure the kernel layer's name is a list
    if isinstance(kernel_layer, list):
        layers = kernel_layer
    elif isinstance(kernel_layer, str):
        layers = [kernel_layer]
    else:
        raise ValueError('Argument kernel_layer has unsupported type: {}'.format(type(kernel_layer)))

    # iterate through the layers in the model and register hook for each specified layer
    for name, layer in model.named_modules():
        if name in layers:
            layer.register_forward_hook(get_activation(name))

    # perform the forward pass with the given input and capture the activations
    _ = model(sequence)

    # visualize the activation
    try:
        import matplotlib.pyplot as plt

        # create the activation plots for each layer and each sequence
        for i in range(nb_seq):
            for j in layers:
                plt.figure()
                plt.title('Activation of {} for sequence {}'.format(j, i))
                plt.imshow(activations[j].numpy()[i, :, :])

        # show the activation plots
        plt.show()

    except ImportError:
        print("Cannot import matplotlib.pyplot")

    except Exception as e:
        print("Unexpected error while trying to plot mean weight distribution:")
        print(e)
