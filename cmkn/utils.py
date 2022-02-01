"""Module containing auxiliary functions used in different parts of the CMKN implementation.

The functions and classes in this file provide functionalities that include accessing k-mer information within
biological sequences (e.g. find_kmer_position, kmer2dict, oli2number, etc.), utilities to train a model (e.g.
ClassBalanceLoss, exp_motif, k_mean, etc.), and utilities to evaluate trained models (e.g. compute_matrices, etc.).

Attributes:
    EPS: A macro for relative tolerance.
    kernels: Dictionary to simplify access to different kernel methods

Authors:
    Jonas C. Ditz: jonas.ditz@uni-tuebingen.de
"""

import numpy as np
from itertools import combinations, product
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from Bio import SeqIO

import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score, precision_recall_curve, average_precision_score,
                             f1_score, matthews_corrcoef)

from timeit import default_timer as timer


# definition of macros
EPS = 1e-4


def find_kmer_positions(sequence, kmer_dict, kmer_size):
    """Utility to localize k-mers

    This function takes a sequences and returns a list of positions for each k-mer of the alphabet.

    Args:
        sequence (:obj:`str`): Input sequence which is used to generate list of k-mer positions.
        kmer_dict (:obj:`dict`): Dictionary mapping each possible k-mer to an integer.
        kmer_size (:obj:`int`): Size of the k-meres.

    Returns:
        List containing a list of positions for each k-mer.

    Raises:
        ValueError: If the input sequence contains a k-mer that is not part of the k-mer dictionary, i.e. the sequence
            and the dictionary were build with different alphabets.
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
                raise ValueError('Substring \'' + sequence[x:y] + '\' not found in k-mer dictionary!\n' +
                                 '            The sequence ' +
                                 '\'{}\' and kmer_dict were build with different alphabets.'.format(sequence))

            # append start position of the current k-mer to the corresponding list of positions
            positions[kmer_dict.get(sequence[x:y])].append(x)

    return positions


def kmer2dict(kmer_size, alphabet):
    """Utility to create k-mer mapping

    This function takes an alphabet and an integer k and returns a dictionary that maps each k-mer (that can be created
    using the specified alphabet) to an integer. This dictionary can than be used to create input features for CON.

    Args:
        kmer_size (:obj:`int`): Size of the k-meres.
        alphabet (:obj:`str`): Alphabet used to generate the k-meres.

    Returns:
        Dictionary mapping each k-mer to an integer.
    """

    # initialize an empty dictionary
    kmer_dict = {}

    # iterate through all possible k-meres of the alphabet and update the dictionary
    for kmer in product(alphabet, repeat=kmer_size):
        kmer_dict.update({''.join(kmer): len(kmer_dict)})

    return kmer_dict


def oli2number(seq, kmer_dict, kmer_size, ambi='DNA'):
    """Utility to translate a string into a tensor

    This function takes a sequence represented by a string and returns an tensor, where each k-mer starting in the
    sequence is encoded by a 2-dimensional vector at the corresponding position.

    Args:
        seq (:obj:`str`): Input sequence which is used to generate list of k-mer positions.
        kmer_dict (:obj:`dict`): Dictionary mapping each possible k-mer to an integer.
        kmer_size (:obj:`int`): Size of the k-meres.
        ambi (:obj:`str`): String indicating the alphabet from which the ambiguous character should be chosen. Defaults
            to 'DNA'.

    Returns:
        Tensor encoding k-mers starting at each position of seq.

    Raises:
        ValueError: If an unknown alphabet was selected for the ambi argument. Currently only 'DNA' and 'PROTEIN' are
            supported.
        ValueError: If the input sequence contains a k-mer that is not part of the k-mer dictionary, i.e. the sequence
            and the dictionary were build with different alphabets.
    """

    # select the correct ambiguous character
    if ambi == 'DNA':
        ambi = 'N'
    elif ambi == 'PROTEIN':
        ambi = 'X'
    elif ambi != 'N' and ambi != 'X':
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
            raise ValueError('Substring \'' + seq[i:i+kmer_size] + '\' not found in k-mer dictionary!\n' +
                             '            The sequence ' +
                             '\'{}\' and kmer_dict were build with different alphabets.'.format(seq))

        # update tensor with the number of the starting oligomer
        oli_tensor[0, i] = np.cos((kmer_dict.get(seq[i:i+kmer_size]) / len(kmer_dict)) * np.pi)
        oli_tensor[1, i] = np.sin((kmer_dict.get(seq[i:i+kmer_size]) / len(kmer_dict)) * np.pi)

    return oli_tensor


def create_consensus(sequences, extension=None, ambi='DNA'):
    """Build consensus sequence

    This utility function takes sequences, either stored in a file readable by Biopython or in a list, and creates a
    consensus sequence.

    Args:
        sequences (:obj:`str` or :obj:`list` of :obj:`str`): Path to Biopython-readable file containing the sequences
            or a list of sequences.
        extension (:obj:`str`): If sequences are provided in a file, this argument has to be set to the file extension.
            Defaults to None.
        ambi (:obj:`str`): String indicating the alphabet from which the ambiguous character should be chosen. Defaults
            to 'DNA'.

    Returns:
        The consensus sequence of the set of sequences provided as input.

    Raises:
        ValueError: If an unknown alphabet was selected for the ambi argument. Currently only 'DNA' and 'PROTEIN' are
            supported.
        ValueError: If extension argument is not set even though the sequences are provided in a file.
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

    Args:
        filepath (:obj:`str`): Path to the file that contains the dataset.
        extension (:obj:`str`): Extension of the dataset file (needed for Biopython's SeqIO routine).
        kmer_dict (:obj:`dict`): Dictionary mapping each k-mer to an integer.
        kmer_size (:obj:`int`): Size of the k-mers (k = kmer_size).

    Returns:
        Tensor containing the starting frequency for each k-mer at each sequence position
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

    Args:
        seq_list (:obj:`list` of :obj:`str`): List containing each sequences as a string.
        kmer_dict (:obj:`dict`): Dictionary mapping each k-mer to an integer.
        kmer_size (:obj:`int`): Size of the k-mers (k = kmer_size).

    Returns:
        Tensor containing the starting frequency for each k-mer at each sequence position

    Raises:
        ValueError: If the sequences are not of the same length.
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


def category_from_output(output):
    """Output selector

    This auxiliary function returns the class with highest probability from the CON output.

    Args:
        output (Tensor): Output of a PyTorch model.

    Returns:
        Index of the category with the highest probability.
    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i


def normalize_(x, p=2, dim=-1):
    """Matrix Normalization

    Auxiliary function implementing numerically stable matrix normalization.

    Args:
        x (Tensor): Input tensor..
        p (:obj:`int`): Indicates the order of the norm
        dim (:obj:`int`): If it is an int, vector norm will be calculated, if it is 2-tuple of ints, matrix norm will be
            calculated. If the value is None, matrix norm will be calculated when the input tensor only has two
            dimensions, vector norm will be calculated when the input tensor only has one dimension. If the input
            tensor has more than two dimensions, the vector norm will be applied to last dimension.
    Returns:
        Normalized input tensor.
    """
    norm = x.norm(p=p, dim=dim, keepdim=True)
    x.div_(norm.clamp(min=EPS))
    return x


#############################################################################
# KERNEL FUNCTIONS
#############################################################################


def exp_func(x, sigma=1, scale=1):
    """ Exponential part of the oligo kernel

    This auxiliary function implements the exponential part of the oligo kernel function. The scaling parameter can be
    set to accommodate for differences in absolute position distances due to different encoding stategies for positional
    information.

    Args:
        x (:obj:`float`): Input to the oligo kernel function.
        sigma (:obj:`float`): Degree of positional uncertainty.
        scale (:obj:`float`): Scaling parameter to accommodate for different positional information encodings.

    Returns:
        Result of the exponential function as a tensor with the same shape of the input tensor.
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


def exp_motif(x, y, length=1, sigma=1, scale=1, alpha=1):
    """Exponential activation function for the oligo layer

    This auxiliary function implements the exponential activation function used in the motif layer kernel function.

    Args:
        x (Tensor): Positional input for the motif kernel function.
        y (Tensor): Motif comparison input for the motif kernel function.
        length (:obj:`int`): Length of the motif.
        sigma (:obj:`float`): Degree of positional uncertainty.
        scale (:obj:`float`): Scaling parameter to accommodate for different positional information encodings.
        alpha (:obj:`float`): Degree of impact of inexact motif matching.

    Returns:
        Result of the motif kernel function as a tensor with the same shape as the input tensors.
    """
    return torch.exp((alpha**2 * (y-length)) + (scale/(2*sigma**2) * (x-1.)))


# dictionary for easy mapping of kernel functions
#   - if one wants to implement new kernel functions write the function definition directly above this dictionary
#     and add an entry to the dictionary
kernels = {
    "exp": exp_func,
    "exp_chen": exp_func2,
    "add_exp_chen": add_exp,
    "exp_oli": exp_motif
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

        Args:
            samples_per_cls (:obj:`list` of :obj:`int`): List containing the number of samples per class in the dataset.
            no_of_classes (:obj:`int`): Number of classes in the classification problem.
            loss_type (:obj:`str`): Loss function used for the class-balance loss.
            beta (:obj:`float`): Hyperparameter for class-balanced loss.
            gamma (:obj:`float`): Hyperparameter for Focal loss

        Raises:
            ValueError: If len(samples_per_cls) != no_of_classes
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

        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes
        print(weights)

    def one_hot(self, labels, num_classes, device, dtype=None, eps=1e-6):
        """Convert an integer label x-D tensor to a one-hot (x+1)-D tensor. Implementation by Kornia
        (https://github.com/kornia).

        Args:
            labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
              Each value is an integer representing correct classification.
            num_classes: number of classes in labels.
            device: the desired device of returned tensor.
            dtype: the desired data type of returned tensor.

        Returns:
            the labels in one hot tensor of shape :math:`(N, C, *)`,

        Examples:
            >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
            >>> one_hot(labels, num_classes=3)
            tensor([[[[1.0000e+00, 1.0000e-06],
                      [1.0000e-06, 1.0000e+00]],
            <BLANKLINE>
                     [[1.0000e-06, 1.0000e+00],
                      [1.0000e-06, 1.0000e-06]],
            <BLANKLINE>
                     [[1.0000e-06, 1.0000e-06],
                      [1.0000e+00, 1.0000e-06]]]])
        """
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

        if not labels.dtype == torch.int64:
            raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

        if num_classes < 1:
            raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

        shape = labels.shape
        one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

    def focal_loss(self, input, target, alpha, gamma=2.0, reduction='none', eps=None):
        """Criterion that computes Focal loss. Implementation by Kornia (https://github.com/kornia).

        According to :cite:`lin2018focal`, the Focal loss is computed as follows:

        .. math::

            \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

        Where:

           - :math:`p_t` is the model's estimated probability for each class.

        Args:
            input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
            target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
            alpha: Weighting factor :math:`\alpha \in [0, 1]`.
            gamma: Focusing parameter :math:`\gamma >= 0`.
            reduction: Specifies the reduction to apply to the
              output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
              will be applied, ``'mean'``: the sum of the output will be divided by
              the number of elements in the output, ``'sum'``: the output will be
              summed.
            eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.

        Return:
            the computed loss.

        Example:
            >>> N = 5  # num_classes
            >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
            >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
            >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
            >>> output.backward()
        """
        if eps is not None and not torch.jit.is_scripting():
            warnings.warn(
                "`focal_loss` has been reworked for improved numerical stability "
                "and the `eps` argument is no longer necessary",
                DeprecationWarning,
                stacklevel=2,
            )

        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        if not len(input.shape) >= 2:
            raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

        if input.size(0) != target.size(0):
            raise ValueError(
                f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

        n = input.size(0)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError(f'Expected target size {out_size}, got {target.size()}')

        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

        # compute softmax over the classes axis
        input_soft: torch.Tensor = F.softmax(input, dim=1)
        log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot: torch.Tensor = self.one_hot(target, num_classes=input.shape[1], device=input.device,
                                                    dtype=input.dtype)

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, gamma)

        focal = -alpha * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

        if reduction == 'none':
            loss = loss_tmp
        elif reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        return loss

    def forward(self, logits, labels):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits) where Loss is one of the standard losses used
        for Neural Networks.

        Args:
            logits (Tensor): Output of the network given as a tensor of shape (batch_size x num_classes).
            labels (Tensor): True label of each sample given as a tensor of shape (batch_size x num_classes).

        Returns:
            A float tensor representing class balanced loss.

        Raises:
            ValueError: If an unknown loss function was specified during initialization of the ClassBalanceLoss object.
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
        e, v = torch.linalg.eigh(input, UPLO='U')
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
# KMEANS/KMEANS++
#############################################################################

def _init_kmeans(x, n_clusters, n_local_trials=None, use_cuda=False, distance='euclidean'):
    """Initialization method for K-Means (k-Means++)

    Args:
        x (Tensor): Data that will be used for clustering provided by a tensor of shape (n_samples x n_dimensions).
        n_clusters (:obj:`int`): Number of clusters that will be computed.
        n_local_trials (:obj:`int`): Number of local seeding trails. Defaults to None.
        use_cuda (:obj:`bool`): Flag that determines whether computations should be performed on the GPU. Defaults to
            False.
        distance (:obj:`str`): Distance measure used for clustering. Defaults to 'euclidean'.

    Returns:
        Initial centers for each cluster.
    """
    n_samples, n_features = x.size()

    # initialize tensor that will hold the cluster centers and send it to GPU if needed
    clusters = torch.Tensor(n_clusters, n_features)
    if use_cuda:
        clusters = clusters.cuda()

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # pick first cluster center randomly
    clusters[0] = x[np.random.randint(n_samples)]

    # initialize list of distances to the selected centroid and calculate current potential
    if distance == 'cosine':
        # calculate distance of each point to the selected centroid using the distance measure of the spherical k-Means
        closest_dist_sq = 1 - clusters[[0]].mm(x.t())
        closest_dist_sq = closest_dist_sq.view(-1)
    elif distance == 'euclidean':
        # calculate distance of each point to the selected centroid using the Euclidean distance measure
        closest_dist_sq = torch.cdist(clusters[[0]], x, p=2)
        closest_dist_sq = closest_dist_sq.view(-1)
    else:
        raise ValueError('Unknown value for parameter mode: {}'.format(distance))
    current_pot = closest_dist_sq.sum().item()

    # pick the remaining n_clusters-1 cluster centers
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional to the squared distance to the closest
        # existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(-1).cpu(), rand_vals)

        # calculate distance of each data point to the candidates
        if distance == 'cosine':
            distance_to_candidates = 1 - x[candidate_ids].mm(x.t())
        elif distance == 'euclidean':
            distance_to_candidates = torch.cdist(x[candidate_ids], x, p=2)
        else:
            raise ValueError('Unknown value for parameter mode: {}'.format(distance))

        # iterate over the candidates for the new cluster center and select the most suitable
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = torch.min(closest_dist_sq,
                                    distance_to_candidates[trial])
            new_pot = new_dist_sq.sum().item()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        clusters[c] = x[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return clusters


def kmeans_gpu(x, n_clusters, distance='euclidian', max_iters=100, verbose=True, init=None, tol=1e-4):
    """Performing k-Means clustering (Lloyd's algorithm) with Tensors utilizing GPU resources.

    Args:
        x (Tensor): Data that will be used for clustering provided as a tensor of shape (n_samples x n_dimensions).
        n_clusters (:obj:`int`): Number of clusters that will be computed.
        distance (:obj:`str`): Distance measure used for clustering. Defaults to 'euclidean'.
        max_iters (:obj:`int`): Maximal number of iterations used in the K-Means clustering. Defaults to 100.
        verbose (:obj:`bool`): Flag to activate verbose output. Defaults to True.
        init (:obj:`str`): Initialization process for the K-Means algorithm. Defaults to None.
        tol (:obj:`float`): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
            of two consecutive iterations to declare convergence. It's not advised to set `tol=0` since convergence
            might never be declared due to rounding errors. Use a very small number instead. Defaults to 1e-4.

    Returns:
        Cluster centers calculated by the K-Means algorithm provided as a tensor of shape (n_clusters x n_dimensions).
    """
    # make sure there are more samples than requested clusters
    if x.shape[0] < n_clusters:
        raise ValueError(f"n_samples={x.shape[0]} should be >= n_clusters={n_clusters}.")

    # check whether the input tensor is on the GPU
    use_cuda = x.is_cuda

    # store number of data points and dimensionality of each data point
    n_samples, n_features = x.size()

    # determine initialization procedure for this run of the k-Means algorithm
    if init == "k-means++":
        print("        Initialization method for k-Means: k-Means++")
        clusters = _init_kmeans(x, n_clusters, use_cuda=use_cuda, distance=distance)
    elif init is None:
        print("        Initialization method for k-Means: random")
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]
    else:
        raise ValueError("Unknown initialization procedure: {}".format(init))

    # perform Lloyd's algorithm iteratively until convergence or the number of iterations exceeds max_iters
    prev_sim = np.inf
    for n_iter in range(max_iters):
        # calculate the distance of data points to clusters using the selected distance measure. Use the calculated
        # distances to assign each data point to a cluster
        if distance == 'cosine':
            sim = x.mm(clusters.t())
            tmp, assign = sim.max(dim=-1)
        elif distance == 'euclidean':
            sim = torch.cdist(x, clusters, p=2)
            tmp, assign = sim.min(dim=-1)
        else:
            raise ValueError('Unknown distance measure: {}'.format(distance))

        # get the mean distance to the cluster centers
        sim_mean = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("        k-Means iter: {}, distance: {}, objective value: {}".format(n_iter + 1, distance, sim_mean))

        # update clusters
        for j in range(n_clusters):
            # get all data points that were assigned to the current cluster
            index = assign == j

            # if no data point was assigned to the current cluster, use the data point furthest away from every cluster
            # as new cluster center
            if index.sum() == 0:
                if distance == 'cosine':
                    idx = tmp.argmin()
                elif distance == 'euclidean':
                    idx = tmp.argmax()
                clusters[j] = x[idx]
                tmp[idx] = 1

            # otherwise, update the center of the current cluster based on all data points assigned to this cluster
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm()

        # stop k-Means if the difference in the cluster center is below the tolerance (i.e. the algorithm converged)
        if torch.abs(prev_sim - sim_mean) / (torch.abs(sim_mean) + 1e-20) < tol:
            break
        prev_sim = sim_mean

    return clusters


def kmeans(x, n_clusters, distance='euclidian', max_iters=100, verbose=True, init=None, tol=1e-4, use_cuda=False):
    """Wrapper for the k-Means clustering algorithm to utilize either GPU or CPU resources.

    Args:
        x (Tensor): Data that will be used for clustering provided as a tensor of shape (n_samples x n_dimensions).
        n_clusters (:obj:`int`): Number of clusters that will be computed.
        distance (:obj:`str`): Distance measure used for clustering. Defaults to 'euclidean'.
        max_iters (:obj:`int`): Maximal number of iterations used in the K-Means clustering. Defaults to 100.
        verbose (:obj:`bool`): Flag to activate verbose output. Defaults to True.
        init (:obj:`str`): Initialization process for the K-Means algorithm. Defaults to None.
        tol (:obj:`float`): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
            of two consecutive iterations to declare convergence. It's not advised to set `tol=0` since convergence
            might never be declared due to rounding errors. Use a very small number instead. Defaults to 1e-4.
        use_cuda (:obj:`bool`): Determine whether to utilize GPU resources or compute kmeans on CPU resources. If set to
            False, scikit-learn's implementation of kmeans will be used. Defaults to False.

    Returns:
        Cluster centers calculated by the K-Means algorithm provided as a tensor of shape (n_clusters x n_dimensions).
    """
    # use GPU implementation if use_cuda was set to true
    if use_cuda:
        clusters = kmeans_gpu(x, n_clusters, distance, max_iters, verbose, init, tol)

    # otherwise, cast Tensors to numpy arrays and use scikit-learn's implementation of kmeans
    else:
        aux_x = x.cpu().numpy()
        sklearn_kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iters, tol=tol, verbose=int(verbose),
                                algorithm='full').fit(aux_x)

        clusters = torch.Tensor(sklearn_kmeans.cluster_centers_)

        # make sure that the cluster centers are on the GPU if the input is on the GPU
        if x.is_cuda:
            clusters = clusters.cuda()

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
    """Compute standard performance metrics for predictions of a trained model.

    Args:
        y_true (Tensor): True label for each sample provided as a tensor of shape (n_sample x n_classes).
        y_pred (Tensor): Predicted label for each sample provided as a tensor of shape (n_samples x n_classes).
        binary (:obj:`int`): Indicates whether the classification task is binary

    Returns:
        Different performance metrics for the provided predictions as a Pandas DataFrame.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metric = {}
    metric['log.loss'] = log_loss(y_true, y_pred)
    metric['accuracy'] = accuracy_score(y_true, y_pred > 0.5)
    metric['F_score'] = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    metric['MCC'] = matthews_corrcoef(y_true.argmax(axis=1), y_pred.argmax(axis=1))
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
