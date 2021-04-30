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
from decimal import Decimal, ROUND_HALF_DOWN

from scipy import optimize
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin

from .utils import kernels, gaussian_filter_1d, matrix_inverse_sqrt, spherical_kmeans, EPS, normalize_, ClassBalanceLoss

import matplotlib.pyplot as plt


class CONLayer(nn.Conv1d):
    """ Convolutional Oligo Kernel Network Layer

    This class implements one layer of a convolutional Oligo Kernel Network (CON).
    """
    def __init__(self, in_channels, out_channels, kmer_length, padding=0, dilation=1, groups=1, subsampling=1,
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
            padding = (kmer_length - 1) // 2
        else:
            padding = 0

        # initialize the parent class
        super(CONLayer, self).__init__(in_channels, out_channels, kernel_size=kmer_length, padding=padding,
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
        self.kappa = lambda x, y: kernel_func(x, y, kmer_length, *self.kernel_args)

        # set the kernel function used for computing the linear transformation factor
        kernel_func_lintrans = kernels["exp"]
        self.kappa_lintrans = lambda x: kernel_func_lintrans(x, *self.kernel_args[0:2])

    def sample_oligomers(self, x_in, n_sampling_patches=1000):
        """Sample oligomers from the given Tensor. These oligomers will be used as input to the spherical k-Means
        algorithm that is used during initialization of the network.

        - **Parameters**::

            :param x_in: One-hot encoding representation of a sequence
                :type x_in: Tensor (batch_size x self.in_channels x seq_len)
            :param n_sampling_patches: Number of patches to sample
                :type n_sampling_patches: Integer

        - **Returns**::

            oligomers: (batch_size x (H - filter_size + 1)) x (in_channels x filter_size)
        """
        oligomers = x_in.unfold(-1, self.filter_size, 1).transpose(1, 2)
        oligomers = oligomers.contiguous().view(-1, self.patch_dim)

        n_sampling_patches = min(oligomers.size(0), n_sampling_patches)

        indices = torch.randperm(oligomers.size(0))[:n_sampling_patches]
        oligomers = oligomers[indices]
        normalize_(oligomers)
        return oligomers

    def initialize_weights(self, seq_len, oligomers, init=None):
        """Initialization of CONLayer's weights and alphanet's weights

        The anchor points of the CON layer will be equidistantly distributed over the whole length of the sequence.
        Afterwards, the buffer alphanet will be set to the corresponding oligomer encodings stored in self.kmer_ref.

        - **Parameters**::

            :param seq_len: Length of the sequence. This parameter will be used to equidistantly distribute the position
                            anchor points along the sequence.
                :type seq_len: Integer
            :param oligomers: Oligomers that will be used to initialize the oligomer anchor points using a spherical
                              k-Means algorithm
                :type oligomers: Tensor (n_sampling_oligomers x self.patch_dim)
            :param init: Initialization parameter for the spherical k-Means algorithm
                :type init: String

        - **Updates**::

            self.weight (out_channels x in_channels): These represent the oligomer anchor points
            self.pos_anchors (out_channels x 2): These represent the encoded position anchor points
        """
        # get position anchor points that are equidistantly distributed over the whole sequence
        dist = seq_len / self.out_channels
        positions = [dist / 2 + i * dist for i in range(self.out_channels)]

        # initialize weight tensor
        pos_tensor = self.pos_anchors.new_zeros([self.out_channels, 2])

        # fill the tensor by projecting anchor point positions onto the upper half of the unit circle
        for i in range(self.out_channels):
            # calculate the x coordinate
            pos_tensor[i, 0] = np.cos((positions[i] / seq_len) * np.pi)
            pos_tensor[i, 1] = np.sin((positions[i] / seq_len) * np.pi)

        # make sure that the layer weights and the auxiliary weight variable have the same shape
        pos_tensor = pos_tensor.view_as(self.pos_anchors)

        # update the layer weight
        self.pos_anchors.data = pos_tensor.data
        self._need_lintrans_computed = True

        # initialize the oligomer anchor points unsing a spherical k-Means algorithm
        oli_tensor = spherical_kmeans(oligomers, self.out_channels, init=init)
        oli_tensor = oli_tensor.view_as(self.weight)
        self.weight.data = oli_tensor.data

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

        lintrans = self.pos_anchors.view(self.out_channels, -1)
        lintrans = lintrans.mm(lintrans.t())
        lintrans = self.kappa_lintrans(lintrans)
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

        # calculate the convolution between oligomer encoding of the input and oligomer encoding of the anchor points
        x_out = super(CONLayer, self).forward(x_in)
        #aux1 = super(CONLayer, self).forward(x_in)

        # calculate convolution between positions of the input sequence and anchor point positions
        pos_out = F.conv1d(pos_in, self.pos_anchors, padding=self.padding, dilation=self.dilation, groups=self.groups)

        # evaluate kernel function with the result
        x_out = self.kappa(pos_out, x_out)

        #bsize = x_in.shape[0]
        #torch.save({'posConv{}'.format(bsize): aux, 'oliConv{}'.format(bsize): oli_out, 'kappa{}'.format(bsize): x_out},
        #           'data/debug/convLayer_tensors_batchsize{}.pkl'.format(bsize))
        #fig, axs = plt.subplots(3)
        #im1 = axs[0].imshow(aux1[0, :, :].detach().numpy(), interpolation=None, aspect='auto')
        #im2 = axs[1].imshow(pos_out[0, :, :].detach().numpy(), interpolation=None, aspect='auto')
        #im3 = axs[2].imshow(x_out[0, :, :].detach().numpy(), interpolation=None, aspect='auto')
        #fig.colorbar(im1, ax=axs[0])
        #fig.colorbar(im2, ax=axs[1])
        #fig.colorbar(im3, ax=axs[2])
        #plt.show()

        #plt.figure()
        #plt.imshow(self.weight.detach().numpy())
        #plt.show()

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
        """ Function to enforce the constraints on the anchor points and alphanet's weights. The kernel function is
        valid iff all weights are normalized (, point to discrete sequence positions,) and the corresponding oligomers
        are encoded by alphanet.weight.
        """
        # make sure all oligomer anchor points are positive and have unit l2-norm
        self.weight.data.clamp_(0)
        norm = self.weight.data.view(self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1)
        norm.clamp_(min=EPS)
        self.weight.data.div_(norm)

        # make sure all position anchor points have unit l2-norm
        norm = self.pos_anchors.data.view(self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1)
        norm.clamp_(min=EPS)
        self.pos_anchors.data.div_(norm)


class CONLayerBernhard(nn.Conv1d):
    """ Convolutional Oligo Kernel Network Layer

    This class implements one layer of a convolutional Oligo Kernel Network (CON) using the differentiable ansatz by
    Bernhard.
    """
    def __init__(self, out_channels, kmer_ref, cutoff, dilation=1, groups=1, subsampling=1,
                 kernel_func="exp_oli", kernel_args=[1, 1, 10000], kernel_args_trainable=False):
        """Constructor of a CON layer

        - **Parameters**::

            :param out_channels: Number of output channels of the layer (this is equal to the number of anchor points)
                :type out_channels: Integer
            :param kmer_ref: This tensor holds the encoded reference sequence, i.e. the oligomer starting at each
                             position of the reference sequence is encoded by a 2-dimensional vector
                :type kmer_ref: Tensor (2 x |S|)
            :param cutoff: All values of the alphanet convolution that exceed the cutoff need to be set to one
                :type cutoff: Float
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
                :type kernel_args: List (Default: [1, 1, 10000])
            :param kernel_args_trainable: Specifies if the kernel arguments are trainable
                :type kernel_args_trainable: Boolean
        """
        # set the number of input channels
        #   -> this value is always 2 for the oligo kernel layer, therefore it is not a parameter that can be assigned
        #      by the user
        self.in_channels = 2

        # set filter size and padding parameter for the underlying convolutional network architecture.
        # For a convolutional oligo kernel layer, the filter size is always 1 and the padding parameter is always set to
        # 0.
        filter_size = 1
        padding = 0

        # initialize parent class
        super(CONLayerBernhard, self).__init__(self.in_channels, out_channels, kernel_size=filter_size, stride=1,
                                                padding=padding, dilation=dilation, groups=groups, bias=False)

        # set parameters
        self.subsampling = subsampling
        self.filter_size = filter_size
        self.patch_dim = self.in_channels * self.filter_size

        # add the oligomer position encoding matrix for the reference sequence as a buffer to the model
        self.kmer_ref = kmer_ref
        self.cutoff = cutoff

        # initialize buffer for the oligomer comparison term
        self.register_buffer("alphanet", torch.Tensor(out_channels, self.in_channels, self.filter_size))

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
        self.kappa = lambda x, y: kernel_func(x, y, *self.kernel_args)

        # set the kernel function used for computing the linear transformation factor
        kernel_func_lintrans = kernels["exp"]
        self.kappa_lintrans = lambda x: kernel_func_lintrans(x, *self.kernel_args[0:2])

    def initialize_weights(self):
        """Initialization of CONLayer's weights and alphanet's weights

        The anchor points of the CON layer will be equidistantly distributed over the whole length of the sequence.
        Afterwards, the buffer alphanet will be set to the corresponding oligomer encodings stored in self.kmer_ref.

        - **Updates**::

            self.weight (out_channels x in_channels): These represent the anchor points
            self.alphanet (out_channels x in_channels): These represent the encoded oligomer starting at the
                                                        anchor points in the reference sequence
        """
        # determine the sequence length
        seq_len = self.kmer_ref.shape[1]

        # get anchor points that are equidistantly distributed over the whole sequence
        dist = seq_len / self.out_channels
        anchors = [dist / 2 + i * dist for i in range(self.out_channels)]

        # initialize weight tensor
        weight = self.weight.new_zeros([self.out_channels, self.in_channels])

        # fill the tensor by projecting anchor point positions onto the upper half of the unit circle
        for i in range(self.out_channels):
            # calculate the x coordinate
            weight[i, 0] = np.cos((anchors[i] / seq_len) * np.pi)
            weight[i, 1] = np.sin((anchors[i] / seq_len) * np.pi)

        # make sure that the layer weights and the auxiliary weight variable have the same shape
        weight = weight.view_as(self.weight)

        # update the layer weight
        self.weight.data = weight.data
        self._need_lintrans_computed = True

        # convert anchor points into discrete seqeuence positions
        anchors_int = [int(Decimal(anchor).quantize(Decimal('1.'), rounding=ROUND_HALF_DOWN)) for anchor in anchors]

        # initialize alphanet weight tensor
        weight_alphanet = self.alphanet.new_zeros([self.out_channels, self.in_channels])

        # fill alphanet weight tensor with the oligomer encodings that correspond to the initialized anchor points
        for i in range(self.out_channels):
            weight_alphanet[i, :] = self.kmer_ref[:, anchors_int[i]]

        # make sure that the alphanet weights and the auxiliary weight variable have the same shape
        weight_alphanet = weight_alphanet.view_as(self.alphanet)

        # update alphanet's weights
        self.alphanet.data = weight_alphanet.data

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

        lintrans = self.weight.view(self.out_channels, -1)
        lintrans = lintrans.mm(lintrans.t())
        lintrans = self.kappa_lintrans(lintrans)
        lintrans = matrix_inverse_sqrt(lintrans)
        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data = lintrans.data

        return lintrans

    def _conv_layer(self, x_in, oli_in):
        """Convolution layer

        This layer computes the convolution: x_out = <phi(p), phi(Z)> * kappa(Zt p)

        - **Parameters**::

            :param x_in: 2-dimensional encoding of the input positions
                :type x_in: Tensor (batch_size x in_channels x |S|)
            :param oli_in: Oligomer encoding of the input sequence
                :type oli_in: Tensor (batch_size x in_channels x |S|)

        - **Returns**::

            :return x_out: Result of the convolution
                :rtype x_out: Tensor (batch_size x out_channels x |S|)
        """
        # calculate the convolution between input and anchor points
        x_out = super(CONLayer, self).forward(x_in)
        #aux1 = super(CONLayer, self).forward(x_in)

        # calculate convolution between oligomer encoding of the input and oligomer encoding of the anchor points
        oli_out = F.conv1d(oli_in, self.alphanet, padding=self.padding, dilation=self.dilation, groups=self.groups)

        # make sure that all valid entries in oli_out are equal to one
        aux = oli_out.new_ones(oli_out.shape)
        oli_out = torch.where(oli_out < self.cutoff, oli_out, aux)

        # evaluate kernel function with the result
        #x_out = self.kappa(x_out, oli_out.type(torch.bfloat16))
        x_out = self.kappa(x_out, oli_out)

        #bsize = x_in.shape[0]
        #torch.save({'posConv{}'.format(bsize): aux, 'oliConv{}'.format(bsize): oli_out, 'kappa{}'.format(bsize): x_out},
        #           'data/debug/convLayer_tensors_batchsize{}.pkl'.format(bsize))
        #fig, axs = plt.subplots(4)
        #im1 = axs[0].imshow(aux1[0, :, :].detach().numpy(), interpolation=None, aspect='auto')
        #im2 = axs[1].imshow(oli_out[0, :, :].detach().numpy(), interpolation=None, aspect='auto')
        #axs[2].imshow(oli_out[0, :, :].detach().numpy() == 1, interpolation=None, aspect='auto')
        #im3 = axs[3].imshow(x_out[0, :, :].detach().numpy(), interpolation=None, aspect='auto')
        #fig.colorbar(im1, ax=axs[0])
        #fig.colorbar(im2, ax=axs[1])
        #fig.colorbar(im3, ax=axs[3])

        #anchors = [int(Decimal((np.arccos(anchor[0].item()) / np.pi) * (oli_out.shape[-1] - 1)).quantize(Decimal('1.'),
        #                                                                                       rounding=ROUND_HALF_DOWN)
        #               )
        #           for anchor in self.weight]
        #print('')
        #print(anchors)
        #print('')
        #plt.figure()
        #plt.imshow(oli_out[0, :, :].detach().numpy() == 1, interpolation=None, aspect='auto')
        #plt.title('[]'.format(anchors))
        #plt.show()

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

    def forward(self, x_in, oli_in):
        """Encode function for a CON layer

        - **Parameters**::

            :param x_in: 2-dimensional encoding of the input positions
                :type x_in: Tensor (batch_size x in_channels x |S|)
            :param oli_in: Oligomer encoding of the input sequence
                :type oli_in: Tensor (batch_size x in_channels x |S|)
        """
        # perform the convolution
        x_out = self._conv_layer(x_in, oli_in)

        # calculate the linear transformation factor (if needed)
        lintrans = self._compute_lintrans()

        # multiply the convolution result by the linear transformation factor
        x_out = self._mult_layer(x_out, lintrans)
        return x_out

    def normalize_(self):
        """ Function to enforce the constraints on the anchor points and alphanet's weights. The kernel function is
        valid iff all weights are normalized (, point to discrete sequence positions,) and the corresponding oligomers
        are encoded by alphanet.weight.
        """
        # get the sequence length
        seq_len = self.kmer_ref.shape[1]

        # make sure all weights have unit l2-norm
        norm = self.weight.data.view(
            self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1)
        norm.clamp_(min=EPS)
        self.weight.data.div_(norm)

        # transform weights into sequence positions
        anchors = [int(Decimal((np.arccos(anchor[0].item()) / np.pi) * (seq_len - 1)).quantize(Decimal('1.'),
                                                                                               rounding=ROUND_HALF_DOWN)
                       )
                   for anchor in self.weight]

        # initialize weight tensor
        #weight = torch.zeros([self.out_channels, self.in_channels])

        # fill the tensor by projecting anchor point positions onto the upper half of the unit circle
        #for i in range(self.out_channels):
        #    # calculate the x coordinate
        #    weight[i, 0] = np.cos((anchors[i] / seq_len) * np.pi)
        #    weight[i, 1] = np.sin((anchors[i] / seq_len) * np.pi)

        # make sure that the layer weights and the auxiliary weight variable have the same shape
        #weight = weight.view_as(self.weight)

        # update the layer weight
        #self.weight.data = weight.data
        self._need_lintrans_computed = True

        # initialize alphanet weight tensor
        weight_alphanet = self.alphanet.new_zeros([self.out_channels, self.in_channels])

        # fill alphanet weight tensor with the oligomer encodings that correspond to the initialized anchor points
        for i in range(self.out_channels):
            weight_alphanet[i, :] = self.kmer_ref[:, anchors[i]]

        # make sure that the alphanet weights and the auxiliary weight variable have the same shape
        weight_alphanet = weight_alphanet.view_as(self.alphanet)

        # update alphanet's weights
        self.alphanet.data = weight_alphanet.data


class CONLayerOld(nn.Conv1d):
    """ Convolutional Oligo Kernel Network Layer

    This class implements one layer of a convolutional Oligo Kernel Network (CON) using the old version of an oligo
    kernel layer.
    """
    def __init__(self, out_channels, ref_kmerPos, dilation=1, groups=1, subsampling=1,
                 kernel_func="exp", kernel_args=[1, 1], kernel_args_trainable=False):
        """Constructor of a CON layer

        - **Parameters**::

            :param out_channels: Number of output channels of the layer (this is equal to the number of anchor points)
                :type out_channels: Integer
            :param ref_kmerPos: Distribution of k-mers in the reference sequence. Used for the evaluation of phi(.)
                                for the anchor points
                :type ref_kmerPos: Tensor
            :param dilation: Controls the spacing between the kernel points; also known as the à trous algorithm
                :type dilation: Integer (Default: 1)
            :param groups: Controls the connections between inputs and outputs
                :type groups: Integer (Default: 1)
            :param subsampling: Controls the amount of subsampling in the current layer
                :type subsampling: Integer (Default: 1)
            :param kernel_func: Specified the kernel function that is used in the current layer
                :type kernel_func: String (Default: "exp")
            :param kernel_args: All parameters of the kernel function
                :type kernel_args: List (Default: [1, 1])
            :param kernel_args_trainable: Specifies if the kernel arguments are trainable
                :type kernel_args_trainable: Boolean
        """
        # set the number of input channels
        #   -> this value is always 2 for the oligo kernel layer, therefore it is not a parameter that can be assigned
        #      by the user
        self.in_channels = 2

        # set filter size and padding parameter for the underlying convolutional network architecture.
        # For a convolutional oligo kernel layer, the filter size is always 1 and the padding parameter is always set to
        # 0.
        filter_size = 1
        padding = 0

        # initialize parent class
        super(CONLayerOld, self).__init__(self.in_channels, out_channels, filter_size, stride=1, padding=padding,
                                          dilation=dilation, groups=groups, bias=False)

        # set parameters
        self.subsampling = subsampling
        self.filter_size = filter_size
        self.patch_dim = self.in_channels * self.filter_size

        # add the oligomer position encoding matrix for the reference sequence as a buffer to the model
        self.register_buffer("ref_kmerPos", torch.Tensor(ref_kmerPos.shape[0], ref_kmerPos.shape[1]))
        self.ref_kmerPos = ref_kmerPos

        # register a buffer to hold the oligomer position comparison term
        self.register_buffer('poscomp', None)

        # specify if the linear transformation factor will be calculated
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
        self.kappa = lambda x: kernel_func(x, *self.kernel_args)

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

    def _compute_poscomp(self, phi):
        """Compute the oligomer position comparison matrix E_{X,W}

        - **Parameters**::

            :param phi: Matrix containing decoded oligomer starting position information about the input sequence
                :type phi: tensor (batch_size x #oligomers x seq_len)
        """
        # register the position comparison matrix as a buffer
        if (self.poscomp is None) or (self.poscomp.shape[0] != phi.shape[0]):
            self.poscomp = phi.new_empty((phi.shape[0], self.out_channels, phi.shape[-1]))

        # get the sequence position of each anchor point
        z_pos = (torch.acos(self.weight[:, 0]) / np.pi) * (phi.shape[-1] - 1)
        z_pos = z_pos.view(-1).detach()

        # calculate the phi(z) vector for the current anchor point
        #   -> since the position will be a float in most cases, this vector is a weighted combination
        #      from two positions in self.ref_kmerPos
        #   -> make sure to prevent out of index errors
        if all(z_pos < (phi.shape[-1] - 1)):
            phi_z = ((1 - (z_pos % 1)) * self.ref_kmerPos[:, (z_pos // 1).numpy()] +
                     (z_pos % 1) * self.ref_kmerPos[:, (z_pos // 1).numpy() + 1]) / 2
        else:
            phi_z = self.ref_kmerPos[:, (z_pos // 1).numpy()]

        # fill out the dot product tensor by iterating in following order
        #   1. over the batch size
        #   2. over all anchor points
        #   3. over all positions
        #for i in range(self.poscomp.shape[0]):
        #    for j in range(self.poscomp.shape[1]):
        #        for k in range(self.poscomp.shape[2]):
        #            # calculate the dot product
        #            # dot_phi[j, :, i] = torch.bmm(phi[j, :, i].view(1, 1, -1),
        #            #                             phi_z.type(x_in.type()).view(1, -1, len(z_pos)))
        #            self.poscomp.data[i, j, k] = torch.dot(phi_z[:, j], phi[i, :, k])
        for i in range(self.poscomp.shape[0]):
            for j in range(self.poscomp.shape[2]):
                self.poscomp.data[i, :, j] = (phi[i, :, j].view(-1, 1) * phi_z).sum(0)

    def _compute_lintrans(self):
        """Compute the linear transformation factor kappa(ZtZ)^(-1/2)

        - **Returns**::

            :return lintrans: Linear transformation factor
                :rtype lintrans: tensor (out_channels x out_channels)
        """
        # return the current linear transformation factor, if no new factor needs to be computed
        if not self._need_lintrans_computed:
            return self.lintrans

        lintrans = self.weight.view(self.out_channels, -1)
        lintrans = lintrans.mm(lintrans.t())
        lintrans = self.kappa(lintrans)
        lintrans = matrix_inverse_sqrt(lintrans)
        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data = lintrans.data

        return lintrans

    def _conv_layer(self, x_in):
        """Convolution layer

        This layer computes the convolution: x_out = <phi(p), phi(Z)> * kappa(Zt p)

        - **Parameters**::

            :param x_in: Input tensor storing the function values phi(p)
                :type x_in: Tensor (batch_size x in_channels x |S|)

        - **Returns**::

            :return x_out: Result of the convolution
                :rtype x_out: Tensor (batch_size x out_channels x |S|)
        """
        # calculate the convolution between input and anchor points
        x_out = super(CONLayer, self).forward(x_in)

        # evaluate kernel function with the result
        x_out = self.kappa(x_out)

        # multiply kernel function evaluation with the position comparison term
        x_out = self.poscomp * x_out * (np.sqrt(np.pi) * self.kernel_args[0])
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

    def forward(self, x_in, phi):
        """Encode function for a CON layer

        - **Parameters**::

            :param x_in: Input tensor storing the function values phi(p)
                :type x_in: Tensor (batch_size x in_channels x |S|)
            :param phi: Tensor storing the one-hot-encoding phi(pos) for each position of the input
                :type phi: Tensor (batch_size x nb_kmer x |S|)
        """
        # compute the oligomer position comparison matrix
        self._compute_poscomp(phi)

        # perform the convolution
        x_out = self._conv_layer(x_in)

        # calculate the linear transformation factor (if needed)
        lintrans = self._compute_lintrans()

        # multiply the convolution result by the linear transformation factor
        x_out = self._mult_layer(x_out, lintrans)
        return x_out

    def unsup_train(self, seq_len):
        """Unsupervised training for a CON layer

        The anchor points of the CON layer will be equidistantly distributed over the whole length of the sequence.

        - **Parameters**::

            :param seq_len: Length of the input sequences
                :type seq_len: Integer

        - **Updates**::

            self.weights (out_channels x in_channels x filter_size): These represent the anchor points
        """
        # get anchor points that are equidistantly distributed over the whole sequence
        dist = seq_len / self.out_channels
        anchors = [dist/2 + i*dist for i in range(self.out_channels)]

        # initialize weight tensor
        weight = torch.zeros([self.out_channels, self.in_channels])
        # fill the tensor by projecting anchor point positions onto the upper half of the unit circle
        for i in range(self.out_channels):
            # calculate the x coordinate
            weight[i, 0] = np.cos((anchors[i]/seq_len) * np.pi)
            weight[i, 1] = np.sin((anchors[i]/seq_len) * np.pi)

        # make sure that the layer weights and the auxiliary weight variable have the same shape
        weight = weight.view_as(self.weight)

        # update the layer weight
        self.weight.data = weight.data
        self._need_lintrans_computed = True

    def normalize_(self):
        """ Function to normalize the layer's weights. The kernel function is valid iff all weights are normalized.
        """
        norm = self.weight.data.view(
            self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1)
        norm.clamp_(min=EPS)
        self.weight.data.div_(norm)


class CKNLayer(nn.Conv1d):
    """ Convolutional Kernel Layer

    This class implements layers of a convolutional kernel network as described by Chen et al., 2019
    "Biological sequence modeling with convolutional kernel networks".

    Author: Dexiong Chen
    Comments added by: Jonas Ditz
    """
    def __init__(self, in_channels, out_channels, filter_size, padding=0, dilation=1, groups=1, subsampling=1,
                 kernel_func="exp_chen", kernel_args=[0.5], kernel_args_trainable=False):
        """ CKN Layer Constructor

        - **Parameters**::

            :param in_channels: Number of input channels
            :param out_channels: Number of output channels
            :param filter_size: Size of the convolutional filter
            :param padding: Used padding type
            :param dilation: Dilation size
            :param groups: Number of groups in the layer
            :param subsampling: Subsampling factor
            :param kernel_func: Kernel function used in this layer
            :param kernel_args: Arguments of the used kernel function
            :param kernel_args_trainable: Indicate whether kernel arguments are trainable
        """

        # set padding parameter dependent on the selected padding type
        if padding == "SAME":
            padding = (filter_size - 1)//2
        else:
            padding = 0

        # call constructor of parent class
        super(CKNLayer, self).__init__(in_channels, out_channels, filter_size, stride=1, padding=padding,
                                       dilation=dilation, groups=groups, bias=False)

        # set layer parameters
        self.subsampling = subsampling
        self.filter_size = filter_size
        self.patch_dim = self.in_channels * self.filter_size

        self._need_lintrans_computed = True

        # initialize a simple-to-use handle of the kernel function
        self.kernel_args_trainable = kernel_args_trainable
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == "exp" or kernel_func == "add_exp":
            kernel_args = [1./kernel_arg ** 2 for kernel_arg in kernel_args]
        self.kernel_args = kernel_args
        if kernel_args_trainable:
            self.kernel_args = nn.ParameterList([nn.Parameter(torch.Tensor([kernel_arg]))
                                                 for kernel_arg in kernel_args])
        kernel_func = kernels[kernel_func]
        self.kappa = lambda x: kernel_func(x, *self.kernel_args)

        # initialize pooling variables
        ones = torch.ones(1, self.in_channels // self.groups, self.filter_size)
        self.register_buffer("ones", ones)
        self.init_pooling_filter()

        # initialize variable for the linear transformation
        self.register_buffer("lintrans",
                             torch.Tensor(out_channels, out_channels))

    def init_pooling_filter(self):
        """ Initialization of pooling filter

        This function initializes the pooling filter based on the desired subsampling factor.
        """
        # return, if no subsampling is requested
        if self.subsampling <= 1:
            return

        # use a 1D Gaussian filter mask, if subsampling is requested
        size = 2 * self.subsampling - 1
        pooling_filter = gaussian_filter_1d(size)
        pooling_filter = pooling_filter.expand(self.out_channels, 1, size)

        # store the pooling filter
        self.register_buffer("pooling_filter", pooling_filter)

    def train(self, mode=True):
        super(CKNLayer, self).train(mode)
        if self.training is True:
            self._need_lintrans_computed = True

    def _compute_lintrans(self):
        """Compute the linear transformation factor kappa(ZtZ)^(-1/2)
        Returns:
            lintrans: out_channels x out_channels
        """
        if not self._need_lintrans_computed:
            return self.lintrans

        lintrans = self.weight.view(self.out_channels, -1)
        lintrans = lintrans.mm(lintrans.t())
        lintrans = self.kappa(lintrans)
        lintrans = matrix_inverse_sqrt(lintrans)
        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data = lintrans.data

        return lintrans

    def _conv_layer(self, x_in):
        """Convolution layer
        Compute x_out = ||x_in|| x kappa(Zt x_in/||x_in||)
        Args:
            x_in: batch_size x in_channels x H
            self.filters: out_channels x in_channels x filter_size
            x_out: batch_size x out_channels x (H - filter_size + 1)
        """
        patch_norm = torch.sqrt(F.conv1d(x_in.pow(2), self.ones,
                                padding=self.padding, dilation=self.dilation,
                                groups=self.groups))
        patch_norm = patch_norm.clamp(EPS)
        x_out = super(CKNLayer, self).forward(x_in)
        x_out = x_out / patch_norm
        x_out = self.kappa(x_out)
        x_out = patch_norm * x_out
        return x_out

    def _mult_layer(self, x_in, lintrans):
        """Multiplication layer
        Compute x_out = kappa(ZtZ)^(-1/2) x x_in
        Args:
            x_in: batch_size x out_channels x H
            lintrans: out_channels x out_channels
            x_out: batch_size x out_channels x H
        """
        batch_size, out_c, _ = x_in.size()
        if x_in.dim() == 2:
            return torch.mm(x_in, lintrans)
        return torch.bmm(lintrans.expand(batch_size, out_c, out_c), x_in)

    def _pool_layer(self, x_in):
        """Pooling layer
        Compute I(z) = \sum_{z'} phi(z') x exp(-\beta_1 ||z'-z||_2^2)
        """
        if self.subsampling <= 1:
            return x_in
        x_out = F.conv1d(x_in, self.pooling_filter, stride=self.subsampling,
                         padding=self.subsampling-1, groups=self.out_channels)
        return x_out

    def forward(self, x_in):
        """Encode function for a CKN layer
        Args:
            x_in: batch_size x in_channels x H x W
        """
        x_out = self._conv_layer(x_in)
        x_out = self._pool_layer(x_out)
        lintrans = self._compute_lintrans()
        x_out = self._mult_layer(x_out, lintrans)
        return x_out

    def compute_mask(self, mask=None):
        if mask is None:
            return mask
        mask = mask.float().unsqueeze(1)
        mask = F.avg_pool1d(mask, kernel_size=self.filter_size,
                            stride=self.subsampling)
        mask = mask.squeeze(1) != 0
        return mask

    def extract_1d_patches(self, input, mask=None):
        output = input.unfold(-1, self.filter_size, 1).transpose(1, 2)
        output = output.contiguous().view(-1, self.patch_dim)
        if mask is not None:
            mask = mask.float().unsqueeze(1)
            mask = F.avg_pool1d(mask, kernel_size=self.filter_size, stride=1)
            # option 2: mask = mask.view(-1) == 1./self.filter_size
            mask = mask.view(-1) != 0
            output = output[mask]
        return output

    def sample_patches(self, x_in, mask=None, n_sampling_patches=1000):
        """Sample patches from the given Tensor
        Args:
            x_in (Tensor batch_size x in_channels x H)
            n_sampling_patches (int): number of patches to sample
        Returns:
            patches: (batch_size x (H - filter_size + 1)) x (in_channels x filter_size)
        """
        patches = self.extract_1d_patches(x_in, mask)
        n_sampling_patches = min(patches.size(0), n_sampling_patches)

        indices = torch.randperm(patches.size(0))[:n_sampling_patches]
        patches = patches[indices]
        normalize_(patches)
        return patches

    def unsup_train(self, patches, init=None):
        """Unsupervised training for a CKN layer
        Args:
            patches: n x in_channels x H
        Updates:
            filters: out_channels x in_channels x filter_size
        """
        weight = spherical_kmeans(patches, self.out_channels, init=init)
        weight = weight.view_as(self.weight)
        self.weight.data = weight.data
        self._need_lintrans_computed = True

    def normalize_(self):
        """ Function to normalize the layer's weights. The kernel function is valid iff all weights are normalized.
        """
        norm = self.weight.data.view(
            self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1)
        norm.clamp_(min=EPS)
        self.weight.data.div_(norm)


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
