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
from torch.autograd import Variable
from Bio import SeqIO


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
                raise ValueError('Substring not found in k-mere dictionary! ' +
                                 'The sequence \'{}\' was NOT build using the specified alphabet.'.format(sequence))

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


def build_kmer_ref(filepath, extension, kmer_dict, kmer_size):
    """

    """
    # get the length of the sequences in the dataset by first reading only the first entry
    first_record = next(SeqIO.parse(filepath, extension))

    # initialize the tensor holding the kmer reference positions
    ref_pos = torch.zeros(len(kmer_dict), len(first_record.seq))

    # keep track of the number of sequences in the dataset
    data_size = 0

    with open(filepath, 'rU') as handle:
        # iterate through file
        for record in SeqIO.parse(handle, extension):

            # update number of sequences in the dataset
            data_size += 1

            # get kmer positions in the current sequence
            positions = find_kmer_positions(record.seq, kmer_dict, kmer_size)

            # update reference tensor
            for i, pos in enumerate(positions):
                for p in pos:
                    ref_pos[i, p] += 1

    # devide every entry in the ref position tensor by the size of the dataset
    return ref_pos / data_size


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

    This function creates a 1D Gaussian filter used for pooling in a CON layer.

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

    :param x: Input matrix
    :param p:
    :param dim:
    :return:
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


# dictionary for easy mapping of kernel functions
#   - if one wants to implement new kernel functions write the function definition directly above this dictionary
#     and add an entry to the dictionary
kernels = {
    "exp": exp_func,
    "exp_chen": exp_func2,
    "add_exp_chen": add_exp
}


#############################################################################
# AUTOGRAD EXTENSIONS
#############################################################################


class MatrixInverseSqrt(torch.autograd.Function):
    """Matrix inverse square root for a symmetric definite positive matrix
    """
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



