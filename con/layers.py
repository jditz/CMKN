################################################
# Python class containing the object holding   #
# network layers for CON models.               #
#                                              #
# Author: Jonas Ditz                           #
# Contact: ditz@informatik.uni-tuebingen.de    #
################################################

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from scipy import optimize
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin

from .utils import kernels, matrix_inverse_sqrt, kmeans, EPS, normalize_, ClassBalanceLoss


class CONLayer(nn.Conv1d):
    """ Convolutional Oligo Kernel Network Layer

    This class implements one layer of a convolutional Oligo Kernel Network (CON).
    """
    def __init__(self, in_channels, out_channels, kmer_length, padding='SAME', dilation=1, groups=1, subsampling=1,
                 kernel_func="exp_oli", kernel_args=(1, 1, 1), kernel_args_trainable=False):
        """Constructor of a CON layer

                - **Parameters**::

                    :param in_channels: Number of input channels (aka size of the used alphabet)
                        :type in_channels: Integer
                    :param out_channels: Number of output channels of the layer (aka number of anchor points)
                        :type out_channels: Integer
                    :param kmer_length: Length of the oligomers investigated (aka the kernel size of the convolutional layer)
                        :type kmer_length: Integer
                    :param padding: Used padding type
                        :type padding: String or Integer
                    :param dilation: Controls the spacing between the kernel points; also known as the à trous algorithm
                        :type dilation: Integer (Default: 1)
                    :param groups: Controls the connections between inputs and outputs
                        :type groups: Integer (Default: 1)
                    :param subsampling: Controls the amount of subsampling in the current layer
                        :type subsampling: Integer (Default: 1)
                    :param kernel_func: Specified the kernel function that is used in the current layer
                        :type kernel_func: String (Default: "exp_oli")
                    :param kernel_args: All parameters of the kernel function. The parameters have to be given in the order
                                        [sigma, scale, alpha].
                        :type kernel_args: Tuple (Default: (1, 1, 1))
                    :param kernel_args_trainable: Specifies if the kernel arguments are trainable
                        :type kernel_args_trainable: Boolean
        """
        # set padding parameter dependent on the selected padding type
        if padding == "SAME":
            self.padding_length = kmer_length - 1
        else:
            self.padding_length = None

        # initialize the parent class
        super(CONLayer, self).__init__(in_channels, out_channels, kernel_size=kmer_length, padding=0,
                                       dilation=dilation, groups=groups, bias=False)

        # set parameters
        self.subsampling = subsampling
        self.filter_size = kmer_length
        self.patch_dim = self.in_channels * self.filter_size

        # add a parameter for the position comparison term
        self.register_parameter('pos_anchors', nn.Parameter(torch.Tensor(out_channels, 2, 1)))

        # specify if the linear transformation factor will be calculated and initialize a buffer for the linear
        # transformation term
        self._need_lintrans_computed = True
        self.register_buffer("lintrans", torch.Tensor(out_channels, out_channels))

        # initialize the kernel function kappa
        self.kernel_args_trainable = kernel_args_trainable
        # make sure that the kernel arguments are given as a list
        #   -> this is important for the call to the kernel function later
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        self.kernel_args = kernel_args
        # if the kernel parameters are trainable, initialize training of parameters here
        if kernel_args_trainable:
            self.kernel_args = nn.ParameterList([nn.Parameter(torch.Tensor([kernel_arg]))
                                                 for kernel_arg in kernel_args])

        # select the chosen kernel function from the dictionary that maps to all available functions
        kernel_func = kernels[kernel_func]
        # for convenient's sake, initialize a simple-to-use handler for the kernel function
        #   -> x = position tensor
        #      y = oligomer tensor
        self.kappa = lambda x, y: kernel_func(x, y, kmer_length, *self.kernel_args)

        # set the kernel function used for computing the linear transformation factor
        #kernel_func_lintrans = kernels["exp"]
        #self.kappa_lintrans = lambda x: kernel_func_lintrans(x, *self.kernel_args[0:2])

    def sample_oligomers(self, x_in, n_sampling_oligomers=1000, include_pos=False):
        """Sample oligomers from the given Tensor. These oligomers will be used as input to the spherical k-Means
        algorithm that is used during initialization of the network.

        - **Parameters**::

            :param x_in: One-hot encoding representation of a sequence
                :type x_in: Tensor (batch_size x self.in_channels x seq_len)
            :param n_sampling_oligomers: Number of patches to sample
                :type n_sampling_oligomers: Integer
            :param include_pos: Determine whether sampled oligomers should include positional information
                :type include_pos: Boolean

        - **Returns**::

            oligomers: (batch_size x (H - filter_size + 1)) x (in_channels x filter_size)
        """
        # unfold the input tensor to create oligomers of the desired length
        oligomers = x_in.unfold(-1, self.filter_size, 1).transpose(1, 2)

        # flatten the one-hot encodings of each oligomer and combine batch size and sequence length into a single
        # dimension
        #   -> this allows to easily sample oligomers of the desired length across the batch and the whole length of
        #      the sequence
        oligomers = oligomers.contiguous().view(-1, self.patch_dim)

        # add positional information to each oligomer if requested
        #   -> since the contiguous() function will put the sorted oligomers of each input sequence next to each other,
        #      we can still add the positional information to each oligomer by simply replicating the list
        #      [0 : sequence_length] a number of times equal to the batch size and concatenate this tensor with the
        #      oligomer tensor at dimension 1
        if include_pos:
            pos_info = oligomers.new_tensor(list(range(x_in.shape[-1] - (self.filter_size - 1))) * x_in.shape[0])
            pos_info = torch.cos((pos_info / x_in.shape[-1]) * np.pi)
            oligomers = torch.cat([oligomers, pos_info.view(-1, 1)], dim=1)

        # make sure to only sample at most the number of oligomers that are presented in the current batch
        n_sampling_oligomers = min(oligomers.size(0), n_sampling_oligomers)

        # sample random indices of the oligomer tensor (number of sampled indices is either the maximum number of
        # oligomers or n_sampling_oligomers, whatever is smaller)
        indices = torch.randperm(oligomers.size(0))[:n_sampling_oligomers]
        oligomers = oligomers[indices]

        # normalize the oligomers only if no positional information was added
        if not include_pos:
            normalize_(oligomers)
        return oligomers

    def initialize_weights(self, distance, oligomers, seq_len, init=None, max_iters=100):
        """Initialization of CONLayer's weights and alphanet's weights

        The anchor points of the CON layer will be equidistantly distributed over the whole length of the sequence.
        Afterwards, the buffer alphanet will be set to the corresponding oligomer encodings stored in self.kmer_ref.

        - **Parameters**::

            :param distance: Distance measure used in the k-Means algorithm
                :type distance: String
            :param oligomers: Oligomers that will be used to initialize the oligomer anchor points using a spherical
                              k-Means algorithm
                :type oligomers: Tensor (n_sampling_oligomers x self.patch_dim)
            :param seq_len: Length of the sequences used as input to the initialization process
                :type seq_len: Integer
            :param init: Initialization parameter for the spherical k-Means algorithm
                :type init: String
            :param max_iters: Maximal number of iterations used in the K-Means clustering
                :type max_iters: Integer

        - **Updates**::

            self.weight (out_channels x in_channels): These represent the oligomer anchor points
            self.pos_anchors (out_channels x 2): These represent the encoded position anchor points
        """
        # perform spherical kmeans algorithm on the given oligomer-position tensors
        oli_tensor = kmeans(oligomers, self.out_channels, distance=distance, init=init, max_iters=max_iters)

        # seperate oligomers and positions
        if distance == 'euclidean':
            pos_tensor = oli_tensor[:, -1]
            oli_tensor = oli_tensor[:, :-1]

            # convert position tensor into sequence positions
            pos_tensor = (torch.acos(pos_tensor) / np.pi) * (seq_len - 1)
        else:
            dist = seq_len / self.out_channels
            pos_tensor = oli_tensor.new_tensor([dist / 2 + i * dist for i in range(self.out_channels)])

        # update the oligomer weights with the computed cluster centers
        oli_tensor = oli_tensor.view_as(self.weight)
        self.weight.data = oli_tensor.data

        # convert sequence position into 2-dimensional vectors with unit l-2 norm
        pos_anchors = self.pos_anchors.new_zeros(self.out_channels, 2)
        for i in range(self.out_channels):
            pos_anchors[i, 0] = torch.cos((pos_tensor[i] / seq_len) * np.pi)
            pos_anchors[i, 1] = torch.sin((pos_tensor[i] / seq_len) * np.pi)

        # update the position anchor points
        pos_anchors = pos_anchors.view_as(self.pos_anchors)
        self.pos_anchors.data = pos_anchors.data
        self._need_lintrans_computed = True

    def train(self, mode=True):
        """ Toggle train mode

        This function extends the train mode toggle functionality to CON layers.

        - **Paramters**::

            :param mode: Input that indicates whether the train mode should be enabled or disabled
                :type mode: Boolean
        """
        super(CONLayer, self).train(mode)
        if self.training is True:
            self._need_lintrans_computed = True

    def _compute_lintrans(self):
        """Compute the linear transformation factor kappa(ZtZ)^(-1/2)

        - **Returns**::

            :return lintrans: Linear transformation factor
                :rtype lintrans: tensor (out_channels x out_channels)
        """
        # return the current linear transformation factor, if no new factor needs to be computed
        if not self._need_lintrans_computed:
            return self.lintrans

        # compute the linear transformation matrix for the positions of each anchor point
        lintrans_pos = self.pos_anchors.view(self.out_channels, -1)
        lintrans_pos = lintrans_pos.mm(lintrans_pos.t())

        # compute the linear transformation matrix for the oligomers of each anchor point
        lintrans_oli = self.weight.view(self.out_channels, -1)
        lintrans_oli = lintrans_oli.mm(lintrans_oli.t())

        # resolve the kernel function using the linear transformation matrices
        lintrans = self.kappa(lintrans_pos, lintrans_oli)
        lintrans = matrix_inverse_sqrt(lintrans)
        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data = lintrans.data

        return lintrans

    def _conv_layer(self, x_in):
        """Convolution layer

        This layer computes the convolution: x_out = <phi(p), phi(Z)> * kappa(Zt p)

        - **Parameters**::

            :param x_in: Oligomer encoding of the input sequence
                :type x_in: Tensor (batch_size x in_channels x |S|)

        - **Returns**::

            :return x_out: Result of the convolution
                :rtype x_out: Tensor (batch_size x out_channels x |S|)
        """
        # build the position tensor
        #   -> this tensor is only dependent on the length of the input sequence, therefore it can be created in the
        #      forward call and doesn't have to be given as an argument
        pos_in = x_in.new_zeros(x_in.size(0), 2, x_in.size(-1))
        for i in range(x_in.size(-1)):
            # project current position on the upper half of the unit circle
            x_circle = np.cos(((i + 1) / x_in.size(-1)) * np.pi)
            y_circle = np.sin(((i + 1) / x_in.size(-1)) * np.pi)

            # fill the input tensor
            pos_in[:, 0, i] = x_circle
            pos_in[:, 1, i] = y_circle

        # add padding to one side of the sequence input to ensure that input and output have the same dimensionality
        #   -> we cannot use the build-in padding provided by nn.Conv1d because padding on both sides of the input
        #      sequence would destroy the oligo kernel formulation
        if self.padding_length > 0:
            x_in = torch.cat([x_in, x_in.new_zeros(x_in.shape[0], x_in.shape[1], self.padding_length)], dim=-1)

        # calculate the convolution between oligomer encoding of the input and oligomer encoding of the anchor points
        x_out = super(CONLayer, self).forward(x_in)

        # calculate convolution between positions of the input sequence and anchor point positions
        pos_out = F.conv1d(pos_in, self.pos_anchors, padding=self.padding, dilation=self.dilation, groups=self.groups)

        # evaluate kernel function with the result
        x_out = self.kappa(pos_out, x_out)

        return x_out

    def _mult_layer(self, x_in, lintrans):
        """Multiplication layer

        This layer multiplies the convolution output with the linear transformation factor:
        x_out = kappa(ZtZ)^(-1/2) x x_in

        - **Parameters**::

            :param x_in: Result of the convolution
                :type x_in: Tensor (batch_size x out_channels x |S|)
            :param lintrans: Linear transformation factor
                :type lintrans: Tensor (out_channels x out_channels)

        - **Returns**:

            :return x_out: Result of the multiplication
                :rtype x_out: Tensor (batch_size x out_channels x |S|)
        """
        batch_size, out_c, _ = x_in.size()

        # calculate normal matrix multiplication or batch matrix multiplication depending on whether input data is
        # presented in batch mode
        if x_in.dim() == 2:
            return torch.mm(x_in, lintrans)
        return torch.bmm(lintrans.expand(batch_size, out_c, out_c), x_in)

    def forward(self, x_in):
        """Encode function for a CON layer

        - **Parameters**::

            :param x_in: one-hot encoding of the input sequence
                :type x_in: Tensor (batch_size x in_channels x |S|)
        """
        # perform the convolution
        x_out = self._conv_layer(x_in)

        # calculate the linear transformation factor (if needed)
        lintrans = self._compute_lintrans()

        # multiply the convolution result by the linear transformation factor
        x_out = self._mult_layer(x_out, lintrans)
        return x_out

    def normalize_(self):
        """ Function to enforce the constraints on the anchor points. The kernel function is valid iff all anchor point
        positions have unit l2-norm and each anchor point motif is a valid PWM.
        """
        # make sure the motifs of all oligomers are valid PWMs by
        #   1. make sure each number is positive
        #   2. make sure each column of the motif sums to one
        self.weight.data.clamp_(min=0)
        norm = self.weight.data.norm(p=1, dim=1).view(-1, 1, self.filter_size)
        norm.clamp_(min=EPS)
        self.weight.data.div_(norm)

        # make sure all position anchor points have unit l2-norm
        norm = self.pos_anchors.data.view(self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1)
        norm.clamp_(min=EPS)
        self.pos_anchors.data.div_(norm)


class GlobalAvg1D(nn.Module):
    """Global average pooling class

    This class implements a global average pooling layer for neural networks.
    """
    def __init__(self):
        # initialize parent class
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=-1)
        mask = mask.float().unsqueeze(1)
        x = x * mask
        return x.sum(dim=-1)/mask.sum(dim=-1)


class GlobalMax1D(nn.Module):
    """Global max pooling class

    This class implements a global max pooling layer for neural networks.
    """
    def __init__(self):
        # initialize parent class
        super(GlobalMax1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(x)
            mask = mask.data
            x[~mask] = -float("inf")
        return x.max(dim=-1)[0]


class GlobalSum1D(nn.Module):
    """Global sum pooling class

    This class implements a global sum pooling layer for neural networks.
    """
    def __init__(self):
        # initialize parent class
        super(GlobalSum1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.sum(dim=-1)
        mask = mask.float().unsqueeze(1)
        x = x * mask
        return x.sum(dim=-1)/mask.sum(dim=-1)


# define directory macro for simple access of pooling options
POOLINGS = {'mean': GlobalAvg1D, 'max': GlobalMax1D, 'sum': GlobalSum1D}


class LinearMax(nn.Linear, LinearModel, LinearClassifierMixin):
    """ Linear Classification Layer
    """
    def __init__(self, in_features, out_features, alpha=0.0, fit_bias=True, penalty="l2"):
        super(LinearMax, self).__init__(in_features, out_features, fit_bias)
        self.alpha = alpha
        self.fit_bias = fit_bias
        self.penalty = penalty
        self.num_classes = out_features

    def forward(self, input, proba=False):
        out = super(LinearMax, self).forward(input)
        if proba:
            # activate with sigmoid function only for binary classification
            if self.num_classes == 2:
                return out.sigmoid()

            # activate with log softmax function for multi-class classification
            else:
                return F.log_softmax(out, dim=1)
        return out

    def fit(self, x, y, criterion=None):
        # determine if computations take place on GPU
        use_cuda = self.weight.data.is_cuda

        # initialize the loss function
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        reduction = criterion.reduction
        criterion.reduction = 'sum'

        # make sure that input is given as Tensors
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

        # transfer input to GPU, if necessary
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        def eval_loss(w):
            """ Custom loss evaluation function for the optimization routine.
            """
            # make sure that w has the correct first dimension
            w = w.reshape((self.out_features, -1))

            # reset weight gradient, if necessary
            if self.weight.grad is not None:
                self.weight.grad = None

            # check, whether a bias is used by the layer
            #   -> if no bias is used, w can be copied to the weight tensor, directly
            #   -> if a bias is used, w has to be split appropriately before being copied to the weight and bias tensors
            if self.bias is None:
                self.weight.data.copy_(torch.from_numpy(w))
            else:
                # reset bias gradient, if necessary
                if self.bias.grad is not None:
                    self.bias.grad = None
                self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
                self.bias.data.copy_(torch.from_numpy(w[:, -1]))

            # perform a classification with the given input
            y_pred = self(x)#.view(-1)

            # calculate the loss
            #   -> differs between binary and multiclass
            #   -> ClassBalanceLoss needs to be handled the same way as multiclass even if only binary classification
            #      is performed
            if self.num_classes > 2 or isinstance(criterion, ClassBalanceLoss):
                loss = criterion(y_pred, y.argmax(1))
            else:
                loss = criterion(y_pred, y)

            # perform back propagation with the calculated loss
            loss.backward()

            # check if regularization is used
            #   -> regularization will be performed if alpha is greater than zero
            if self.alpha != 0.0:

                # calculate the l2 regularization term, if this type of penalty was selected
                if self.penalty == "l2":
                    penalty = 0.5 * self.alpha * torch.norm(self.weight)**2

                # calculate the l1 regularization term, if this type of penalty was selected
                elif self.penalty == "l1":
                    penalty = self.alpha * torch.norm(self.weight, p=1)
                    penalty.backward()

                # catch the case were users insert an invalid argument for the penalty parameter
                else:
                    penalty = 0

                # incorporate the regularization term into the loss
                loss = loss + penalty

            # return the calculated loss
            return loss.item()

        def eval_grad(w):
            """ Custom gradient evaluation function for the optimization routine
            """
            # get the current gradient
            dw = self.weight.grad.data

            # extra step for l2 regularization, if "l2" was selected for the penalty option and alpha is set to a value
            # greater than zero
            if self.alpha != 0.0:
                if self.penalty == "l2":
                    dw.add_(self.alpha, self.weight.data)

            # if a bias was specified, combine weight and bias gradient to optimize both
            if self.bias is not None:
                db = self.bias.grad.data
                dw = torch.cat((dw, db.view(-1, 1)), dim=1)

            # return the computed gradient as a flatten numpy array
            return dw.cpu().numpy().ravel().astype("float64")

        # get initial tensor for the optimization
        #   -> if there is a bias, combine the bias and weight tensors
        w_init = self.weight.data
        if self.bias is not None:
            w_init = torch.cat((w_init, self.bias.data.view(-1, 1)), dim=1)
        w_init = w_init.cpu().numpy().astype("float64")

        # perform the optimization routine
        w = optimize.fmin_l_bfgs_b(eval_loss, w_init, fprime=eval_grad, maxiter=100, disp=0)
        if isinstance(w, tuple):
            w = w[0]

        # make sure that tensor w and the weight tensor have the same shape
        w = w.reshape((self.out_features, -1))

        # reset the weight gradient
        self.weight.grad.data.zero_()

        # store result of the optimization routine
        #   -> if no bias was specified, the resulting tensor can be directly copied to the weight tensor
        #   -> if a bias was specified, the resulting tensor must be split appropriately before copying the content to
        #      the weight and bias vector
        if self.bias is None:
            self.weight.data.copy_(torch.from_numpy(w))
        else:
            self.bias.grad.data.zero_()
            self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
            self.bias.data.copy_(torch.from_numpy(w[:, -1]))
        criterion.reduction = reduction

    def decision_function(self, x):
        x = torch.from_numpy(x)
        return self(x).data.numpy().ravel()

    def predict(self, x):
        return self.decision_function(x)

    def predict_proba(self, x):
        return self._predict_proba_lr(x)

    @property
    def coef_(self):
        return self.weight.data.numpy()

    @property
    def intercept_(self):
        return self.bias.data.numpy()
