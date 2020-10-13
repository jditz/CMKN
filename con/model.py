################################################
# Python class containing the CON model        #
#                                              #
# Author: Jonas Ditz                           #
# Contact: ditz@informatik.uni-tuebingen.de    #
################################################

import sys
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
from timeit import default_timer as timer
import copy

from .layers import POOLINGS, CONLayer, CKNLayer, LinearMax
from .utils import category_from_output, ClassBalanceLoss


class CONSequential(nn.Module):
    """CON Layer Wrapper

    This class is a wrapper for a CON (multi-)layer. Every network can use a CON layer using this wrapper.
    """

    def __init__(self, out_channels_list, ref_kmerPos, filter_sizes, subsamplings, kernel_funcs=None,
                 kernel_args_list=None, kernel_args_trainable=False, **kwargs):
        """Constructor of a CON layer

        - **Parameters**::

            :param out_channels_list: Number of output channels of each layer
                :type out_channels_list: List of Integer
            :param ref_kmerPos: Distribution of k-mers in the reference sequence. Used for the evaluation of phi(.)
                                for the anchor points
                :type ref_kmerPos: Tensor
            :param filter_sizes: Size of the filter for each layer
                :type filter_sizes: List of Integer
            :param subsamplings: List of subsampling factors for each layer
                :type subsamplings: List of Integer
            :param kernel_funcs: Specifies the kernel function used in the CON layers
                :type kernel_funcs: String (Default: None)
            :param kernel_args_list: List of arguments for the used kernel.
                :type kernel_args_list: List (Default: None)
            :param kernel_args_trainable: List that indicates for each layer if the kernel parameters used in
                                          this layer are trainable.
                :type kernel_args_trainable: List of Boolean (Default: None)

        - **Exceptions**::

            :raise ValueError if out_channels_list, filter_sizes, and subsamplings are not of the same length.
        """

        # out_channels_list, filter_sizes, and subsamplings have to be of same length
        #   -> therefore, raise an AssertionError if the lengths differ
        if not len(out_channels_list) == len(filter_sizes)+1 == len(subsamplings):
            raise ValueError('Incompatible dimensions! \n'
                             '            out_channels_list, filter_sizes, and subsamplings have to be of same length.')

        # in_channel of the CON layer will always be 2
        #   -> set variable in_channel for convenience sake
        in_channels = 2

        # initialize parent class
        super(CONSequential, self).__init__()

        # set class parameters of the CON layer
        #   -> number of layers is defined by the length of out_channels_list, i.e. how many different
        #      output channel numbers are specified (each output channel number specified the output dimensionality
        #      of one layer)
        self.n_layers = len(out_channels_list)
        self.in_channels = in_channels
        self.out_channels = out_channels_list[-1]
        self.filter_sizes = filter_sizes
        self.subsamplings = subsamplings

        # auxiliary list to temporarily hold all CON layers of the current network
        con_layers = []

        # iterate over all CON layers an initialize each one, separately
        for i in range(self.n_layers):
            # set the kernel function for all layers separately
            if kernel_funcs is None:
                kernel_func = "exp"
            else:
                kernel_func = kernel_funcs[i]

            # set the kernel hyperparameter (e.g. sigma) for all layers, separately
            if kernel_args_list is None:
                kernel_arg = [1, 1]
                kernel_arg_trainable = False
            else:
                kernel_arg = kernel_args_list[i]
                kernel_arg_trainable = kernel_args_trainable[i]

            # initialize every CON layer using all predefined parameters
            #   -> in a multi-layer construction, only the first layer is a specialized CONLayer; following layers are
            #      standard CKNLayers
            if i == 0:
                con_layer = CONLayer(out_channels_list[i], ref_kmerPos, subsampling=subsamplings[i],
                                     kernel_func=kernel_func,
                                     kernel_args=kernel_arg,
                                     kernel_args_trainable=kernel_arg_trainable,
                                     **kwargs)
            else:
                con_layer = CKNLayer(in_channels, out_channels_list[i],
                                     filter_sizes[i-1], subsampling=subsamplings[i],
                                     kernel_func=kernel_func,
                                     kernel_args=kernel_arg,
                                     kernel_args_trainable=kernel_arg_trainable,
                                     **kwargs)

            con_layers.append(con_layer)

            # number of input channels of the next layer is equal to the number of output channels in the current layer
            in_channels = out_channels_list[i]

        # make CON layers usable
        self.con_layers = nn.Sequential(*con_layers)

    def __getitem__(self, idx):
        return self.con_layers[idx]

    def __len__(self):
        return len(self.con_layers)

    def __iter__(self):
        return iter(self.con_layers._modules.values())

    def forward_at(self, x, i=0):
        """ Fast access to the forward function of a specific layer

        - **Parameters**::

            :param x: Input to the layer
                :type x: Tensor
            :param i: Number of the layer that will be accessed
                :type i: Integer

        - **Returns**::

            :return: Evaluation of the forward function of layer i using input x

        - **Exceptions**::

            :raise ValueError if layer index i is out of bound
            :raise ValueError if input x does not have the correct number of input channels for layer i
        """
        # check if layer index i is out of bound
        if i >= self.n_layers:
            raise ValueError('Layer index out of bound!\n'
                             '            Number of layers: {}'.format(self.n_layers))

        # check if input x is suitable for layer i
        if i > 0 and not x.size(1) == self.con_layers[i].in_channels:
            raise ValueError('Incompatible dimensions!\n            Given input does not have the correct number of '
                             'input channels for layer {}.'.format(i))

        return self.con_layers[i](x)

    def forward(self, x):
        """ Overwritten forward function for CONSequential objects

        - **Parameters**::
            :param x: Input to the CONSequential object
                :type x: Tensor

        - **Returns**::

            :return: Sequential evaluation of the input using all layers of the CONSequential object
        """
        return self.con_layers(x)

    def representation(self, x, n=0):
        """ Function to use a subset of CONSequential's layers for encoding the input

        - **Parameters**::

            :param x: Input to the network
                :type x: Tensor
            :param n: Number of layers used for input encoding
                :type n: Integer

        - **Returns**::

            :return: Representation of the input using only the specified subset of CONSequential's layers
        """
        if n == -1:
            n = self.n_layers
        for i in range(n):
            x = self.forward_at(x, i)
        return x

    def compute_mask(self, mask=None, n=-1):
        """ Function to compute the mask for a specific layer of the CONSequential object

        - **Parameters**::

            :param mask: Initial mask
            :param n: Layer for which the mask will be computed
                :type n: Integer

        - **Returns**::

            :return: Mask of the specified layer or None if no mask is used

        - **Exceptions**::

            :raise ValueError if layer index n is out of bound
        """
        if mask is None:
            return mask
        if n > self.n_layers:
            raise ValueError('Layer index out of bound!\n'
                             'Number of layers: {}'.format(self.n_layers))
        if n == -1:
            n = self.n_layers
        for i in range(1, n):
            mask = self.con_layers[i].compute_mask(mask)
        return mask

    def normalize_(self):
        """ Function to normalize the weights of each layer. The kernel function is valid iff all weights are
        normalized.
        """
        for module in self.con_layers:
            module.normalize_()


class CON(nn.Module):
    """Convolutional Oligo Kernel Network

    This implementation of a convolutional oligo kernel network combines a convolutional oligo kernel layer with CKN
    layers as introduced by Julien Mairal (Mairal et al., 2016).
    """

    def __init__(self, out_channels_list, ref_kmerPos, filter_sizes, subsamplings, kernel_funcs=None,
                 kernel_args_list=None, kernel_args_trainable=False, alpha=0., fit_bias=True, global_pool='sum',
                 penalty='l2', scaler='standard_row', num_classes=1, **kwargs):
        """Constructor of the CON class.

        - **Parameters**::

            :param out_channels_list: Number of output channels of each layer
                :type out_channels_list: List of Integer
            :param ref_kmerPos: Distribution of k-mers in the reference sequence. Used for the evaluation of phi(.)
                                for the anchor points
                :type ref_kmerPos: Tensor (number of kmers x length of sequence)
            :param filter_sizes: Size of the filter for each layer
                :type filter_sizes: List of Integer
            :param subsamplings: List of subsampling factors for each layer
                :type subsamplings: List of Integer
            :param kernel_funcs: Specifies the kernel function used in the CON layers
                :type kernel_funcs: String (Default: None)
            :param kernel_args_list: List of arguments for the used kernel.
                :type kernel_args_list: List (Default: None)
            :param kernel_args_trainable: List that indicates for each layer if the kernel parameters used in
                                          this layer are trainable.
                :type kernel_args_trainable: List of Boolean (Default: None)
            :param alpha: Parameter of the classification layer
                :type alpha: Float
            :param fit_bias: Indicates whether the bias of the classification layer should be fitted
                :type fit_bias: Boolean (Default: True)
            :param global_pool: Indicates which method should be used for global pooling
                :type global_pool: String (Default: 'mean')
            :param penalty: Indicates which penalty method should be used
                :type penalty: String (Default: 'l2')
            :param scaler: Specifies which scaler will be used
                :type scaler: String
            :param num_classes: Number of classes in the current classification problem
                :type num_classes: Integer

        - **Exceptions**::

            :raise ValueError if out_channels_list, filter_sizes, and subsamplings are not of the same length.
        """

        # initialize parent class
        super(CON, self).__init__()

        # store the length of sequences used as input to this network
        self.seq_len = ref_kmerPos.size(1)
        self.num_classes = num_classes

        # initialize the CON layers and catch thrown exceptions
        try:
            self.con_model = CONSequential(out_channels_list, ref_kmerPos, filter_sizes, subsamplings, kernel_funcs,
                                           kernel_args_list, kernel_args_trainable, **kwargs)
        except ValueError as valerr:
            print('ValueError: {}'.format(valerr))
            sys.exit(1)

        # initialize the global pooling layer
        self.global_pool = POOLINGS[global_pool]()

        # set the number of output features
        #   -> this is the number of output channels of the last CON layer
        self.out_features = out_channels_list[-1]

        # initialize the classification layer
        self.initialize_scaler(scaler)
        self.classifier = LinearMax(self.out_features, num_classes, alpha=alpha, fit_bias=fit_bias, penalty=penalty)

    def initialize_scaler(self, scaler=None):
        pass

    def normalize_(self):
        """ Function to normalize the weights of each layer of the convolutional oligo kernel network
        """
        self.con_model.normalize_()

    def representation_at(self, input, mask=None, n=0):
        """ Function to access a specific subset CON layers

        - **Parameters**::

            :param input: Input to the network
                :type input: Tensor
            :param mask: Initial mask
            :param n: Index of the layer that will be used for the evaluation
                :type n: Integer

        - **Returns**::

            :return: Evaluation of the n-th CON layer using the given input
        """
        output = self.con_model.representation(input, n)
        mask = self.con_model.compute_mask(mask, n)
        return output, mask

    def representation(self, input, mask=None):
        """ Function to combine CON layer and pooling layer evaluation

        - **Parameters**::

            :param input: Input to the network
                :type input: Tensor
            :param mask: Initial mask

        - **Returns**::

            :return: Evaluation of the CON layer(s) with subsequent pooling using the given input
        """
        output = self.con_model(input)
        mask = self.con_model.compute_mask(mask)
        output = self.global_pool(output, mask)
        return output

    def forward(self, input, proba=False):
        """ Overwritten forward function for CON objects

        - **Parameters**::
            :param input: Input to the CONSequential object
                :type input: Tensor
            :param proba: Indicates whether the network should produce probabilistic output
                :type proba: Boolean

        - **Returns**::

            :return: Evaluation of the input
        """
        output = self.representation(input)
        return self.classifier(output, proba)

    def unsup_train_con(self, data_loader, n_sampling_patches=100000, init=None, use_cuda=False):
        """ This function initializes the anchor points for each CON layer in an unsupervised fashion

        - **Parameters**::

            :param data_loader: PyTorch DataLoader that handles data
                :type data_loader: torch.utils.data.DataLoader
            :param n_sampling_patches: Number of patches that will be sampled for the spherical k-means at each CKN
                                       layer
                :type n_sampling_patches: Integer
            :param init: Can be set to "kmeans++" for a more sophisticated initialization of the spherical k-means
                         algorithm
                :type init: String
            :param use_cuda: Specified whether all computations will be performed on the GPU
                :type use_cuda: Boolean
        """
        self.train(False)
        if use_cuda:
            self.cuda()
        for i, con_layer in enumerate(self.con_model):
            # initialize the CON layer
            if i == 0:
                print("Training layer {} (oligo layer)".format(i))

                # initialize the anchor points of the CON layer in an unsupervised fashion
                con_layer.unsup_train(self.seq_len)

            # initialize all of the remaining CKN layers
            else:
                print("Training layer {} (kernel layer)".format(i))
                n_patches = 0
                # set the number of patches per batch
                try:
                    n_patches_per_batch = (n_sampling_patches + len(data_loader) - 1) // len(data_loader)
                except:
                    n_patches_per_batch = 1000

                # initialize the tensor that will store all patches and transport this tensor to the GPU if
                # necessary
                patches = torch.Tensor(n_sampling_patches, con_layer.patch_dim)
                if use_cuda:
                    patches = patches.cuda()

                # iterate through all data
                for data, _ in data_loader:
                    # skip if already enough patches were sampled
                    if n_patches >= n_sampling_patches:
                        continue

                    # transfer data to GPU if necessary
                    if use_cuda:
                        data = data.cuda()

                    # do not keep track of the gradients
                    with torch.no_grad():
                        # do a forward propagation to the current layer and retrieve representation and mask
                        data, mask = self.representation_at(data, n=i)

                        # sample patches from the current layer using the representation from the forward propagation
                        data_patches = con_layer.sample_patches(data, mask, n_patches_per_batch)

                    # make sure that the specified number of patches is used for the unsupervised initialization of the
                    # layer's anchor points
                    size = data_patches.size(0)
                    if n_patches + size > n_sampling_patches:
                        size = n_sampling_patches - n_patches
                        data_patches = data_patches[:size]
                    patches[n_patches: n_patches + size] = data_patches
                    n_patches += size

                print("total number of patches: {}".format(n_patches))
                patches = patches[:n_patches]

                # initialize the anchor points of the current CKN layer in an unsupervised fashion
                con_layer.unsup_train(patches, init=init)

    def unsup_train_classifier(self, data_loader, criterion=None, use_cuda=False):
        """ This function initializes the classification layer in an unsupervised fashion

        - **Parameters**::

            :param data_loader: PyTorch DataLoader that handles data
                :type data_loader: torch.utils.data.DataLoader
            :param criterion: Specifies the loss function. If set to None, torch.nn.BCEWithLogitsLoss() will be used.
                :type criterion: PyTorch Loss function (e.g. torch.nn.L1Loss)
            :param use_cuda: Specified whether all computations will be performed on the GPU
                :type use_cuda: Boolean
        """
        # perform an initial prediction using the network
        encoded_train, encoded_target = self.predict(data_loader, True, use_cuda=use_cuda)

        # check if a scaler was defined and not yet fitted
        if hasattr(self, 'scaler') and not self.scaler.fitted:
            self.scaler.fitted = True
            size = encoded_train.shape[0]
            # fit the scaler
            encoded_train = self.scaler.fit_transform(encoded_train.view(-1, self.out_features)).view(size, -1)

        # fit the classification layer with the initial prediction
        self.classifier.fit(encoded_train, encoded_target, criterion)

    def predict(self, data_loader, only_representation=False, proba=False, use_cuda=False):
        """ CON prediction function

        - **Parameters**::

            :param data_loader: PyTorch DataLoader that handles data
                :type data_loader: torch.utils.data.DataLoader
            :param only_representation:
                :type only_representation: Boolean
            :param proba: Indicates whether the network should produce probabilistic output
                :type proba: Boolean
            :param use_cuda: Specified whether all computations will be performed on the GPU
                :type use_cuda: Boolean

        - **Returns**::

            :return output:
            :return target_output:
        """
        # set training mode of the model to False
        self.train(False)

        # move model either to CPU or GPU
        if use_cuda:
            self.cuda()

        # detect the number of samples that will be classified and initialize tensor that stores the targets of each
        # sample
        n_samples = len(data_loader.dataset)

        # iterate over all samples
        batch_start = 0
        for i, (data, target, *_) in enumerate(data_loader):
            batch_size = data.shape[0]

            # transfer sample data to GPU if computations are performed there
            if use_cuda:
                data = data.cuda()

            # do not keep track of the gradients during the forward propagation
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data, proba).data.cpu()

            # combine the result of the forward propagation
            #batch_out = torch.cat((batch_out[:batch_size], batch_out[batch_size:]), dim=-1)

            # initialize tensor that holds the results of the forward propagation for each sample
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = torch.Tensor(n_samples, target.shape[-1])

            # update output and target_output tensor with the current results
            output[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = target

            # continue with the next batch
            batch_start += batch_size

        # return the forward propagation results and the real targets
        output.squeeze_(-1)
        return output, target_output

    def sup_train(self, train_loader, criterion, optimizer, lr_scheduler=None, init_train_loader=None, epochs=100,
                  val_loader=None, n_sampling_patches=500000, unsup_init=None, use_cuda=False, early_stop=True):
        """ Perform supervised training of the CON model

        - **Parameters**::

            :param train_loader: PyTorch DataLoader that handles data
                :type train_loader: torch.utils.data.DataLoader
            :param criterion: Specifies the loss function.
                :type criterion: PyTorch Loss function (e.g. torch.nn.L1Loss)
            :param optimizer: Optimization algorithm used during training
                :type optimizer: PyTorch Optimizer (e.g. torch.optim.Adam)
            :param lr_scheduler: Algorithm used for learning rate adjustment
                :type lr_scheduler: PyTorch LR Scheduler (e.g. torch.optim.lr_scheduler.LambdaLR) (Default: None)
            :param init_train_loader: PyTorch DataLoader that handles data during the unsupervised initialization of
                                      anchor points. Data handled by train_loader is used if no DataLoader is specified.
                :type init_train_loader: torch.utils.data.DataLoader (Default_ None)
            :param epochs: Number of epochs during training
                :type epochs: Integer (Default: 100)
            :param val_loader: PyTorch DataLoader that handles data during the validation phase
                :type val_loader: torch.utils.data.DataLoader (Default: None)
            :param n_sampling_patches: Number of patches used during anchor points initialization
                :type n_sampling_patches: Integer (Default: 500000)
            :param unsup_init: Can be set to "kmeans++" for a more sophisticated initialization of the spherical k-means
                         algorithm
                :type unsup_init: String (Default: None)
            :param use_cuda: Specified whether all computations will be performed on the GPU
                :type use_cuda: Boolean (Default: False)
            :param early_stop: Specifies if early stopping will be used during training
                :type early_stop: Boolean (Default: True)

        - **Returns**::

            :return trained model
        """
        print("Initializing CON layers")
        tic = timer()

        # initialize the anchor points of all layers that use anchor points in an unsupervised fashion
        if init_train_loader is not None:
            self.unsup_train_con(init_train_loader, n_sampling_patches, init=unsup_init, use_cuda=use_cuda)
        else:
            self.unsup_train_con(train_loader, n_sampling_patches, init=unsup_init, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic) / 60))

        # specify the data used for each phase
        #   -> ATTENTION: a validation phase only exists if val_loader is not None
        phases = ['train']
        data_loader = {'train': train_loader}
        if val_loader is not None:
            phases.append('val')
            data_loader['val'] = val_loader

        # initialize variables to keep track of the epoch's loss, the best loss, and the best accuracy
        epoch_loss = None
        best_loss = float('inf')
        best_acc = 0

        # iterate over all epochs
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            # set the models train mode to False
            self.train(False)

            # for each epoch, calculate a new fit for the linear classifier using the current state of the model
            self.unsup_train_classifier(train_loader, criterion, use_cuda=use_cuda)

            # iterate over all phases of the training process
            for phase in phases:

                # if there is a 'train' phase, set model's train mode to True and initialize the Learning Rate Scheduler
                # (if one was specified)
                if phase == 'train':
                    if lr_scheduler is not None:

                        # if the learning rate scheduler is 'ReduceLROnPlateau' and there is a current loss, the next
                        # lr step needs the current loss as input
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)

                        # otherwise call the step() function of the learning rate scheduler
                        else:
                            lr_scheduler.step()

                        # print the current learning rate
                        print("current LR: {}".format(
                            optimizer.param_groups[0]['lr']))

                    # set model's train mode to True
                    self.train(True)
                else:
                    self.train(False)

                # initialize variables to keep track of the loss and the number of correctly classified samples in the
                # current epoch
                running_loss = 0.0
                running_corrects = 0

                # iterate over the data that will be used in the current phase
                for data, target, *_ in data_loader[phase]:
                    size = data.size(0)
                    target = target.float()

                    # if the computations take place on the GPU, send data to GPU
                    if use_cuda:
                        data = data.cuda()
                        target = target.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward propagation through the model
                    #   -> do not keep track of the gradients if we are in the validation phase
                    if phase == 'val':
                        with torch.no_grad():
                            output = self(data)
                            pred = (output.data > 0).float()

                            # multiclass prediction needs special call of loss function
                            if self.num_classes > 2:
                                loss = criterion(output, target.argmax(1))
                            else:
                                loss = criterion(output, target)
                    else:
                        output = self(data)
                        pred = (output > 0).float()

                        # multiclass prediction needs special call of loss function
                        if self.num_classes > 2:
                            loss = criterion(output, target.argmax(1))
                        else:
                            loss = criterion(output, target)

                    # backward propagate + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        self.normalize_()

                    # update statistics
                    running_loss += loss.item() * size
                    running_corrects += torch.sum(pred == target.data).item()

                # calculate loss and accuracy in the current epoch
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if (phase == 'val') and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    # store model parameters only if the generalization error improved (i.e. early stopping)
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            print()

        # report training results
        print('Finish at epoch: {}'.format(epoch + 1))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best val loss: {:4f}'.format(best_loss))

        # if early stopping is enabled, make sure that the parameters are used, which resulted in the best
        # generalization error
        if early_stop:
            self.load_state_dict(best_weights)

        return self


class CON2(nn.Module):
    """Convolutional Oligo Kernel Network

    This class implements a version of the Convolutional Oligo Kernel Network that combines a convolutional oligo kernel
    layer with standard CNN layers. Therefore, no additional CKN layers are used.
    """

    def __init__(self, out_channels_list, ref_kmerPos, filter_sizes, strides, paddings, kernel_func=None,
                 kernel_args=None, kernel_args_trainable=False, alpha=0., fit_bias=True, batch_norm=True, dropout=False,
                 pool='mean', penalty='l2', scaler='standard_row', num_classes=1, **kwargs):
        """Constructor of the CON class.

        - **Parameters**::

            :param out_channels_list: Number of output channels of each layer
                :type out_channels_list: List of Integer
            :param ref_kmerPos: Distribution of k-mers in the reference sequence. Used for the evaluation of phi(.)
                                for the anchor points
                :type ref_kmerPos: Tensor (number of kmers x length of sequence)
            :param filter_sizes: Size of the filter for each layer
                :type filter_sizes: List of Integer
            :param strides: List of stride factors for each pooling layer
                :type strides: List of Integer
            :param paddings: List of padding factors for each convolutional layer
                :type paddings: List of String
            :param kernel_funcs: Specifies the kernel function used in the CON layers
                :type kernel_funcs: String (Default: None)
            :param kernel_args_list: List of arguments for the used kernel.
                :type kernel_args_list: List (Default: None)
            :param kernel_args_trainable: List that indicates for each layer if the kernel parameters used in
                                          this layer are trainable.
                :type kernel_args_trainable: List of Boolean (Default: None)
            :param alpha: Parameter of the classification layer
                :type alpha: Float
            :param fit_bias: Indicates whether the bias of the classification layer should be fitted
                :type fit_bias: Boolean (Default: True)
            :param batchNorm: Indicates whether batch normalization should be used for convolutional layers.
                :type batchNorm: Boolean (Default: True)
            :param dropout: Indicates whether Dropout should be used for convolutional layers.
                :type dropout: Boolean (Default: False)
            :param pool: Indicates the pooling layer used after each convolutional layer
                :type pool: String (Default: 'mean')
            :param global_pool: Indicates which method should be used for global pooling
                :type global_pool: String (Default: 'sum')
            :param penalty: Indicates which penalty method should be used
                :type penalty: String (Default: 'l2')
            :param scaler: Specifies which scaler will be used
                :type scaler: String
            :param num_classes: Number of classes in the current classification problem
                :type num_classes: Integer

        - **Exceptions**::

            :raise ValueError if out_channels_list, filter_sizes, and subsamplings are not of the same length.
        """

        # check if out_channels_list, strides, filter_sizes, and paddings have an acceptable length
        #   -> therefore, raise an AssertionError if the lengths differ
        if not len(out_channels_list) == len(strides) == len(filter_sizes) + 1 == len(paddings) + 1:
            raise ValueError('Incompatible dimensions! \n'
                             '            out_channels_list and strides have to be of same length while filter_sizes '
                             'and paddings have to be one entry shorter than the other two.')

        # initialize parent class
        super(CON2, self).__init__()

        # auxiliary variable to map the pooling choice onto a PyTorch layer
        poolings = {'mean': nn.AvgPool1d, 'max': nn.MaxPool1d, 'lpp': nn.LPPool1d, 'adaMax': nn.AdaptiveMaxPool1d,
                    'adaMean': nn.AdaptiveAvgPool1d}

        # store the length of sequences used as input to this network
        self.seq_len = ref_kmerPos.size(1)
        self.num_classes = num_classes

        # set the default kernel function if none was specified
        if kernel_func is None:
            kernel_func = "exp"

        # set the default kernel hyper-parameters (e.g. sigma) if none were specified
        if kernel_args is None:
            kernel_args = [1, 1]
            kernel_args_trainable = False

        # initialize the CON layer using all predefined parameters
        self.oligo = CONLayer(out_channels_list[0], ref_kmerPos, subsampling=strides[0], kernel_func=kernel_func,
                              kernel_args=kernel_args, kernel_args_trainable=kernel_args_trainable, **kwargs)

        # initialize the additional "normal" convolutional layers
        convlayers = []
        for i in range(1, len(out_channels_list)):

            # set padding parameter dependent on the selected padding type
            if paddings[i-1] == "SAME":
                padding = (filter_sizes[i-1] - 1) // 2
            else:
                padding = 0

            # initialize the "normal" convolutional layers
            #   -> ATTENTION: in_channels is equal to the number of output channels of the previous layer
            convlayers.append(nn.Conv1d(out_channels_list[i-1], out_channels_list[i], kernel_size=filter_sizes[i-1],
                                        stride=1, padding=padding))

            # perform batch normalization after each conv layer (if set to True)
            if batch_norm:
                convlayers.append(nn.BatchNorm1d(out_channels_list[i]))

            # use rectifiedLinearUnit as activation function
            convlayers.append(nn.ReLU(inplace=True))

            # use dropout after each conv layer (if set to True)
            if dropout:
                convlayers.append(nn.Dropout())

            # add max pooling
            convlayers.append(poolings[pool](kernel_size=filter_sizes[i-1], stride=strides[i], padding=padding))

        # combine convolutional oligo kernel layer and all "normal" conv layers into a Sequential layer
        self.conv = nn.Sequential(*convlayers)

        # set the number of output features
        #   -> this is the number of output channels of the last CON layer
        self.out_features = out_channels_list[-1] * self.seq_len

        # initialize the classification layer
        self.initialize_scaler(scaler)
        self.fc = nn.Linear(self.out_features, self.num_classes*100, fit_bias)
        self.classifier = nn.Linear(self.num_classes*100, self.num_classes, fit_bias)
        #self.classifier = LinearMax(self.out_features, self.num_classes, alpha=alpha, fit_bias=fit_bias, penalty=penalty)

    def initialize_scaler(self, scaler=None):
        pass

    def normalize_(self):
        """ Function to normalize the weights of  the convolutional oligo kernel layer
        """
        self.oligo.normalize_()

    def forward(self, input, phi, proba=False):
        """ Overwritten forward function for CON objects

        - **Parameters**::
            :param input: Input to the CONSequential object
                :type input: Tensor (batch_size x 2 x |S|)
            :param phi: Tensor storing the one-hot-encoding phi(pos) for each position of the input
                :type phi: Tensor (batch_size x nb_kmer x |S|)
            :param proba: Indicates whether the network should produce probabilistic output
                :type proba: Boolean

        - **Returns**::

            :return: Evaluation of the input
        """
        output = self.oligo(input, phi)
        output = self.conv(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.classifier(output)
        if proba:
            # activate with sigmoid function only for binary classification
            if self.num_classes == 2:
                return output.sigmoid()

            # activate with softmax function for multi-class classification
            else:
                return F.softmax(output, dim=1)
        else:
            return output

    def unsup_train_classifier(self, data_loader, criterion=None, use_cuda=False):
        """ This function initializes the classification layer in an unsupervised fashion

        - **Parameters**::

            :param data_loader: PyTorch DataLoader that handles data
                :type data_loader: torch.utils.data.DataLoader
            :param criterion: Specifies the loss function. If set to None, torch.nn.BCEWithLogitsLoss() will be used.
                :type criterion: PyTorch Loss function (e.g. torch.nn.L1Loss)
            :param use_cuda: Specified whether all computations will be performed on the GPU
                :type use_cuda: Boolean
        """
        # perform an initial prediction using the network
        encoded_train, encoded_target = self.predict(data_loader, True, use_cuda=use_cuda)

        # check if a scaler was defined and not yet fitted
        if hasattr(self, 'scaler') and not self.scaler.fitted:
            self.scaler.fitted = True
            size = encoded_train.shape[0]
            # fit the scaler
            encoded_train = self.scaler.fit_transform(encoded_train.view(-1, self.out_features)).view(size, -1)

        # fit the classification layer with the initial prediction
        self.classifier.fit(encoded_train, encoded_target, criterion)

    def predict(self, data_loader, only_representation=False, proba=False, use_cuda=False):
        """ CON2 prediction function

        - **Parameters**::

            :param data_loader: PyTorch DataLoader that handles data
                :type data_loader: torch.utils.data.DataLoader
            :param only_representation:
                :type only_representation: Boolean
            :param proba: Indicates whether the network should produce probabilistic output
                :type proba: Boolean
            :param use_cuda: Specified whether all computations will be performed on the GPU
                :type use_cuda: Boolean

        - **Returns**::

            :return output: Result of the forward propagation without the classification layer
            :return target_output: Real labels of the input data
        """
        # set training mode of the model to False
        self.train(False)

        # move model either to CPU or GPU
        if use_cuda:
            self.cuda()

        # detect the number of samples that will be classified and initialize tensor that stores the targets of each
        # sample
        n_samples = len(data_loader.dataset)

        # iterate over all samples
        batch_start = 0
        for i, (phi, target, *_) in enumerate(data_loader):
            batch_size = phi.shape[0]
            phi.requires_grad = False
            data = torch.zeros(phi.size(0), 2, phi.size(-1))
            for j in range(phi.size(-1)):
                # project current position on the upper half of the unit circle
                x_circle = np.cos(((j + 1) / phi.size(-1)) * np.pi)
                y_circle = np.sin(((j + 1) / phi.size(-1)) * np.pi)

                # fill the input tensor
                data[:, 0, j] = x_circle
                data[:, 1, j] = y_circle

            # transfer sample data to GPU if computations are performed there
            if use_cuda:
                data = data.cuda()

            # do not keep track of the gradients during the forward propagation
            with torch.no_grad():
                if only_representation:
                    batch_out = self.oligo(data, phi)
                    batch_out = self.conv(batch_out)
                    batch_out = batch_out.view(batch_size, -1)
                    batch_out = self.fc(batch_out)
                else:
                    batch_out = self(data, phi, proba).data.cpu()

            # combine the result of the forward propagation
            #batch_out = torch.cat((batch_out[:batch_size], batch_out[batch_size:]), dim=-1)

            # initialize tensor that holds the results of the forward propagation for each sample
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = torch.Tensor(n_samples, target.shape[-1])

            # update output and target_output tensor with the current results
            output[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = target

            # continue with the next batch
            batch_start += batch_size

        # return the forward propagation results and the real targets
        output.squeeze_(-1)
        return output, target_output

    def initialize(self):
        """ Function to initialize parameters of the network
        """
        # initialize the anchor points of the CON layer in an unsupervised fashion
        print("Initializing layer 0 (oligo kernel layer)...")
        self.oligo.unsup_train(self.seq_len)

        # iterate over all convolutional layers
        for i, layer in enumerate(self.conv):

            # initialize the convolutional layer
            if isinstance(layer, nn.Conv1d):
                print("Initializing layer {} (conv layer)...".format(i+1))

                # initializing weights using the He initialization (also called Kaiming initialization)
                #   -> only use this initialization if ReLU activation is used
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

    def sup_train(self, train_loader, criterion, optimizer, lr_scheduler=None, epochs=100, val_loader=None,
                  use_cuda=False, early_stop=True):
        """ Perform supervised training of the CON model

        - **Parameters**::

            :param train_loader: PyTorch DataLoader that handles data
                :type train_loader: torch.utils.data.DataLoader
            :param criterion: Specifies the loss function.
                :type criterion: PyTorch Loss function (e.g. torch.nn.L1Loss)
            :param optimizer: Optimization algorithm used during training
                :type optimizer: PyTorch Optimizer (e.g. torch.optim.Adam)
            :param lr_scheduler: Algorithm used for learning rate adjustment
                :type lr_scheduler: PyTorch LR Scheduler (e.g. torch.optim.lr_scheduler.LambdaLR) (Default: None)
            :param epochs: Number of epochs during training
                :type epochs: Integer (Default: 100)
            :param val_loader: PyTorch DataLoader that handles data during the validation phase
                :type val_loader: torch.utils.data.DataLoader (Default: None)
            :param use_cuda: Specified whether all computations will be performed on the GPU
                :type use_cuda: Boolean (Default: False)
            :param early_stop: Specifies if early stopping will be used during training
                :type early_stop: Boolean (Default: True)

        - **Returns**::

            :return training and validation accuracies and losses of each epoch
        """

        print("Initializing CON layers")
        tic = timer()

        # initialize the anchor points of all layers that use anchor points in an unsupervised fashion and initialize
        # weights of all convolutional layers
        self.initialize()

        toc = timer()
        print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # specify the data used for each phase
        #   -> ATTENTION: a validation phase only exists if val_loader is not None
        phases = ['train']
        data_loader = {'train': train_loader}
        if val_loader is not None:
            phases.append('val')
            data_loader['val'] = val_loader

        # initialize variables to keep track of the epoch's loss, the best loss, and the best accuracy
        epoch_loss = None
        best_loss = float('inf')
        best_acc = 0

        # iterate over all epochs
        list_acc = {'train': [], 'val': []}
        list_loss = {'train': [], 'val': []}
        for epoch in range(epochs):
            tic = timer()
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            # set the models train mode to False
            self.train(False)

            # for each epoch, calculate a new fit for the linear classifier using the current state of the model
            #self.unsup_train_classifier(train_loader, criterion, use_cuda=use_cuda)

            # iterate over all phases of the training process
            for phase in phases:

                # if the current phase is 'train', set model's train mode to True and initialize the Learning Rate
                # Scheduler (if one was specified)
                if phase == 'train':
                    if lr_scheduler is not None:

                        # if the learning rate scheduler is 'ReduceLROnPlateau' and there is a current loss, the next
                        # lr step needs the current loss as input
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)

                        # otherwise call the step() function of the learning rate scheduler
                        else:
                            lr_scheduler.step()

                        # print the current learning rate
                        print("current LR: {}".format(optimizer.param_groups[0]['lr']))

                    # set model's train mode to True
                    self.train(True)

                # if the current phase is not 'train', set the model's train mode to False. In this case, the learning
                # rate is irrelevant.
                else:
                    self.train(False)

                # initialize variables to keep track of the loss and the number of correctly classified samples in the
                # current epoch
                running_loss = 0.0
                running_corrects = 0

                # iterate over the data that will be used in the current phase
                for phi, target, *_ in data_loader[phase]:
                    # create input data
                    data = torch.zeros(phi.size(0), 2, phi.size(-1))
                    for i in range(phi.size(-1)):
                        # project current position on the upper half of the unit circle
                        x_circle = np.cos(((i + 1) / phi.size(-1)) * np.pi)
                        y_circle = np.sin(((i + 1) / phi.size(-1)) * np.pi)

                        # fill the input tensor
                        data[:, 0, i] = x_circle
                        data[:, 1, i] = y_circle

                    size = data.size(0)
                    phi.requires_grad = False
                    target = target.float()

                    # if the computations take place on the GPU, send data to GPU
                    if use_cuda:
                        data = data.cuda()
                        phi = phi.cuda()
                        target = target.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward propagation through the model
                    #   -> do not keep track of the gradients if we are in the validation phase
                    if phase == 'val':
                        with torch.no_grad():
                            output = self(data, phi)

                            # create prediction tensor
                            pred = torch.zeros(output.shape)
                            for i in range(output.shape[0]):
                                pred[i, category_from_output(output[i, :])] = 1

                            # multiclass prediction needs special call of loss function
                            if self.num_classes > 2 or isinstance(criterion, ClassBalanceLoss):
                                loss = criterion(output, target.argmax(1))
                            else:
                                loss = criterion(output, target)
                    else:
                        output = self(data, phi)

                        # create prediction tensor
                        pred = torch.zeros(output.shape)
                        for i in range(output.shape[0]):
                            pred[i, category_from_output(output[i, :])] = 1

                        # multiclass prediction needs special call of loss function
                        if self.num_classes > 2 or isinstance(criterion, ClassBalanceLoss):
                            loss = criterion(output, target.argmax(1))
                        else:
                            loss = criterion(output, target)

                    # backward propagate + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        self.normalize_()

                    # update statistics
                    running_loss += loss.item() * size
                    running_corrects += torch.sum(torch.sum(pred == target.data, 1) ==
                                                  torch.ones(pred.shape[0]) * self.num_classes).item()

                # calculate loss and accuracy in the current epoch
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)

                # print the statistics of the current epoch
                list_acc[phase].append(epoch_acc)
                list_loss[phase].append(epoch_loss)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if (phase == 'val') and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    # store model parameters only if the generalization error improved (i.e. early stopping)
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            toc = timer()
            print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # report training results
        print('Finish at epoch: {}'.format(epoch + 1))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best val loss: {:4f}'.format(best_loss))

        # if early stopping is enabled, make sure that the parameters are used, which resulted in the best
        # generalization error
        if early_stop:
            self.load_state_dict(best_weights)

        return list_acc, list_loss
