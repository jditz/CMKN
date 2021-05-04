################################################
# Python class containing the CON model        #
#                                              #
# Author: Jonas Ditz                           #
# Contact: ditz@informatik.uni-tuebingen.de    #
################################################

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from timeit import default_timer as timer
import copy

from .layers import POOLINGS, CONLayer, LinearMax
from .utils import category_from_output, ClassBalanceLoss


class CON(nn.Module):
    """
    Convolutional Oligo Kernel Network
    """
    def __init__(self, in_channels, out_channels_list, filter_sizes, strides, paddings, kernel_func=None,
                 kernel_args=None, kernel_args_trainable=False, alpha=0., fit_bias=True, batch_norm=True, dropout=False,
                 pool_global=None, pool_conv='mean', penalty='l2', scaler=None, num_classes=1, **kwargs):
        """Constructor of the CON class.

        - **Parameters**::

            :param in_channels: Dimensionality of the alphabet that generates the input sequences
                :type in_channels: Integer
            :param out_channels_list: Number of output channels of each layer
                :type out_channels_list: List of Integer
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
            :param alpha: Strength of the regularization in the classification layer (only used if the classification
                          layer is a LinearMax layer)
                :type alpha: Float
            :param fit_bias: Indicates whether the bias of the classification layer should be fitted
                :type fit_bias: Boolean (Default: True)
            :param batchNorm: Indicates whether batch normalization should be used for convolutional layers.
                :type batchNorm: Boolean (Default: True)
            :param dropout: Indicates whether Dropout should be used for convolutional layers.
                :type dropout: Boolean (Default: False)
            :param pool_conv: Indicates the pooling layer used after each convolutional layer
                :type pool_conv: String (Default: 'mean')
            :param global_pool: Indicates which method should be used for global pooling
                :type global_pool: String (Default: 'sum')
            :param penalty: Indicates which penalty method should be used for regularization in the classification layer
                            (only used if the classfication layer is a LinearMax layer)
                :type penalty: String (Default: 'l2')
            :param scaler: If set to a String, a LinearMixin layer will be used for classification and the String
                           specifies the used scalar, e.g. 'standard_row'. If set to an Integer, a standard fully
                           connected layer will be used for classification and the Integer determines the input
                           dimensionality of that layer.
                :type scaler: String or Integer
            :param num_classes: Number of classes in the current classification problem
                :type num_classes: Integer

        - **Exceptions**::

            :raise ValueError if out_channels_list, filter_sizes, and subsamplings are not of the same length.
        """

        # check if out_channels_list, strides, filter_sizes, and paddings have an acceptable length
        #   -> therefore, raise an AssertionError if the lengths differ
        if not len(out_channels_list) == len(strides) == len(filter_sizes) == len(paddings):
            raise ValueError('Incompatible dimensions! \n'
                             '            out_channels_list, filter_sizes, strides and paddings must have the same '
                             'length!')

        # initialize parent class
        super(CON, self).__init__()

        # auxiliary variable to map the pooling choice onto a valid PyTorch layer
        poolings = {'mean': nn.AvgPool1d, 'max': nn.MaxPool1d, 'lpp': nn.LPPool1d, 'adaMax': nn.AdaptiveMaxPool1d,
                    'adaMean': nn.AdaptiveAvgPool1d}

        # store the number of output classes
        self.num_classes = num_classes

        # set the default kernel function if none was specified
        if kernel_func is None:
            kernel_func = "exp_oli"

        # set the default kernel hyper-parameters (e.g. sigma) if none were specified
        if kernel_args is None:
            kernel_args = [1, 1, 1]
            kernel_args_trainable = False

        # store value of parameter sigma for forward pass
        self.sigma = kernel_args[0]

        # initialize the Oligo Kernel Layer
        self.oligo = CONLayer(in_channels, out_channels_list[0], filter_sizes[0], padding=paddings[0],
                              subsampling=strides[0], kernel_func=kernel_func, kernel_args=kernel_args,
                              kernel_args_trainable=kernel_args_trainable, **kwargs)

        # initialize the additional "normal" convolutional layers if any should be used
        self.nb_conv_layers = len(out_channels_list) - 1
        if self.nb_conv_layers > 0:
            convlayers = []
            for i in range(1, len(out_channels_list)):

                # set padding parameter dependent on the selected padding type
                if paddings[i - 1] == "SAME":
                    padding = (filter_sizes[i - 1] - 1) // 2
                else:
                    padding = 0

                # initialize the "normal" convolutional layers
                #   -> ATTENTION: in_channels is equal to the number of output channels of the previous layer
                convlayers.append(
                    nn.Conv1d(out_channels_list[i], out_channels_list[i], kernel_size=filter_sizes[i],
                              stride=strides[i], padding=padding))

                # perform batch normalization after each conv layer (if set to True)
                if batch_norm:
                    convlayers.append(nn.BatchNorm1d(out_channels_list[i]))

                # use rectifiedLinearUnit as activation function
                convlayers.append(nn.ReLU(inplace=True))

                # use dropout after each conv layer (if set to True)
                if dropout:
                    convlayers.append(nn.Dropout())

                # add max pooling
                convlayers.append(
                    poolings[pool_conv](kernel_size=filter_sizes[i - 1], stride=strides[i], padding=padding))

            # combine convolutional oligo kernel layer and all "normal" conv layers into a Sequential layer
            self.conv = nn.Sequential(*convlayers)

        # Initialization of pooling and classification layer; use a standard fully connected layer if scaler is an
        # Integer. Furthermore, use two fully connected layers for more stability, if no global pooling layer is used.
        if isinstance(scaler, int):
            if pool_global is None:
                # set the number of output features
                #   -> this is the number of output channels of the last CON layer multiplied by the sequence length
                self.out_features = out_channels_list[-1] * scaler
                self.global_pool = None

                # initialize two FC layer for stability reason, since no global pooling layer is used
                self.fc = nn.Linear(self.out_features, self.num_classes * 100, bias=fit_bias)
                self.classifier = nn.Linear(self.num_classes * 100, self.num_classes, bias=fit_bias)
            else:
                # set the number of output features
                #   -> this is the number of output channels of the last CON layer
                self.out_features = out_channels_list[-1]

                # initialize fully connected linear layer
                self.classifier = nn.Linear(self.out_features, self.num_classes, bias=fit_bias)
        else:
            # set the specified global pooling layer
            self.global_pool = POOLINGS[pool_global]()

            # set the number of output features
            #   -> this is the number of output channels of the last CON layer
            self.out_features = out_channels_list[-1]

            # initialize the scaler and LinearMixin layer
            self.initialize_scaler(scaler)
            self.classifier = LinearMax(self.out_features, self.num_classes, alpha=alpha, fit_bias=fit_bias,
                                        penalty=penalty)

    def initialize_scaler(self, scaler=None):
        pass

    def normalize_(self):
        """ Function to normalize the weights of each layer of the convolutional oligo kernel network
        """
        self.oligo.normalize_()

    def representation(self, x_in, mask=None):
        """ Function to combine CON layer and pooling layer evaluation

        - **Parameters**::
            :param x_in: Input to the model
                :type x_in: Tensor (batch_size x |A| x |S|)
            :param mask: Initial mask

        - **Returns**::

            :return: Evaluation of the CON layer(s) with subsequent pooling using the given input
        """
        x_out = self.oligo(x_in)

        if self.nb_conv_layers > 0:
            x_out = self.conv(x_out)

        if self.global_pool is None:
            x_out = x_out.view(x_out.size(0), -1)
        else:
            x_out = self.global_pool(x_out, mask)

        return x_out

    def forward(self, x_in, proba=False):
        """ Overwritten forward function for CON objects

        - **Parameters**::
            :param x_in: Input to the model
                :type x_in: Tensor (batch_size x |A| x |S|)
            :param proba: Indicates whether the network should produce probabilistic output
                :type proba: Boolean

        - **Returns**::

            :return: Evaluation of the input
        """
        x_out = self.representation(x_in)
        if isinstance(self.classifier, LinearMax):
            return self.classifier(x_out, proba)
        else:
            if hasattr(self, 'fc'):
                x_out = self.fc(x_out)
            x_out = self.classifier(x_out)
            if proba:
                # activate with sigmoid function only for binary classification
                if self.num_classes == 2:
                    return x_out.sigmoid()

                # activate with softmax function for multi-class classification
                else:
                    return F.softmax(x_out, dim=1)
            else:
                return x_out

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

        # detect the number of samples that will be classified and initialize tensor that stores the targets of each
        # sample
        n_samples = len(data_loader.dataset)

        # iterate over all samples
        batch_start = 0
        for i, (oli_in, target, *_) in enumerate(data_loader):

            batch_size = oli_in.shape[0]

            # transfer sample data to GPU if computations are performed there
            if use_cuda:
                oli_in = oli_in.cuda()

            # do not keep track of the gradients during the forward propagation
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(oli_in).data.cpu()
                else:
                    batch_out = self(oli_in, proba).data.cpu()

            # combine the result of the forward propagation
            #batch_out = torch.cat((batch_out[:batch_size], batch_out[batch_size:]), dim=-1)

            # initialize tensor that holds the results of the forward propagation for each sample
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = torch.Tensor(n_samples, target.shape[-1]).type_as(target)

            # update output and target_output tensor with the current results
            output[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = target

            # continue with the next batch
            batch_start += batch_size

        # return the forward propagation results and the real targets
        output.squeeze_(-1)
        return output, target_output

    def initialize(self, data_loader, distance, n_sampling_olis=100000, init=None, max_iters=100, use_cuda=False):
        """ Function to initialize parameters of the network

        - **Parameters**::

            :param data_loader: PyTorch DataLoader object that handles access to training data
                :type data_loader: torch.utils.data.DataLoader
            :param distance: Distance measure used in the k-Means algorithm
                :type distance: String
            :param n_sampling_olis: Number of oligomers that will be sampled to initialize oligomer anchor points
                                       using the spherical k-Means algorithm
                :type n_sampling_olis: Integer
            :param init: Initialization parameter for the spherical k-Means algorithm
                :type init: String
            :param max_iters: Maximal number of iterations used in the K-Means clustering
                :type max_iters: Integer
            :param use_cuda: Parameter to determine whether to do calculations on the GPU or CPU.
                :type use_cuda: Boolean
        """

        # turn train mode of
        self.train(False)

        # send model to GPU if use_cuda flag was set to True
        if use_cuda:
            self.cuda()

        # initialize the Oligo Kernel layer
        print('    Initializing Oligo Kernel Layer')
        n_oligomers = 0

        # determine the number of oligomers sampled from each batch
        #   -> if this number cannot be calculated, sample 1000 oligomers from each batch until n_sampling_olis are
        #      sampled
        try:
            n_oligomers_per_batch = (n_sampling_olis + len(data_loader) - 1) // len(data_loader)
        except:
            n_oligomers_per_batch = 1000

        # depending on the chosen distance measure for the k-Means algorithm, oligomers will incorporate positional
        # information
        if distance == 'euclidean':
            # initialize tensor that stores the sampled oligomers and make sure it is on the same device as the model
            oligomers = self.oligo.weight.new_zeros(n_sampling_olis, self.oligo.patch_dim + 1)

            # make sure that oligomers contain positional information
            include_pos = True
        else:
            # initialize tensor that stores the sampled oligomers and make sure it is on the same device as the model
            oligomers = self.oligo.weight.new_zeros(n_sampling_olis, self.oligo.patch_dim)

            # make sure that oligomers will not contain positional information
            include_pos = False

        # get batches using the DataLoader object
        seq_len = None
        for data, _ in data_loader:
            # stop sampling oligomers if the maximum number of oligomers is already achieved
            if n_oligomers >= n_sampling_olis:
                break

            # get the length of the sequences; this will be needed later
            if seq_len is None:
                seq_len = data.shape[-1]

            # send data to GPU if use_cuda flag was set
            if use_cuda:
                data = data.cuda()

            # sample the specified number of oligomers using the current batch of data
            with torch.no_grad():
                data_oliogmers = self.oligo.sample_oligomers(data, n_oligomers_per_batch, include_pos)

            # only use a subset of the sampled oligomers in this batch, if this batch would exceed the maximum number
            # of sampled oligomers
            size = data_oliogmers.size(0)
            if n_oligomers + size > n_sampling_olis:
                size = n_sampling_olis - n_oligomers
                data_oliogmers = data_oliogmers[:size]

            # update the oligomer tensor with the currently sampled oligomers and the amount of already sampled
            # oligomers
            oligomers[n_oligomers: n_oligomers + size] = data_oliogmers
            n_oligomers += size

        # initialize the oligomer and position anchor points of the Oligo Kernel layer
        print('        total number of sampled oligomers: {}'.format(n_oligomers))
        self.oligo.initialize_weights(distance, oligomers, seq_len, init=init, max_iters=max_iters)

        # iterate over all convolutional layers
        if self.nb_conv_layers > 0:
            for i, layer in enumerate(self.conv):

                # initialize the convolutional layer
                if isinstance(layer, nn.Conv1d):
                    print("    Initializing layer {} (conv layer)...".format(i + 1))

                    # initializing weights using the He initialization (also called Kaiming initialization)
                    #   -> only use this initialization if ReLU activation is used
                    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

    def sup_train(self, train_loader, criterion, optimizer, lr_scheduler=None, init_train_loader=None,
                  distance='euclidean', n_sampling_olis=100000, kmeans_init=None, epochs=100, val_loader=None,
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
            :param init_train_loader: PyTorch DataLoader object that handles access to training data used during
                                      initialization of the Oligo Kernel layer.
                :type init_train_loader: torch.utils.data.DataLoader
            :param distance: Distance measure used in the k-Means algorithm
                :type distance: String
            :param n_sampling_olis: Number of oligomers that will be sampled to initialize oligomer anchor points
                                       using the spherical k-Means algorithm
                :type n_sampling_olis: Integer
            :param kmeans_init: Initialization parameter for the spherical k-Means algorithm
                :type kmeans_init: String
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
        if init_train_loader is not None:
            self.initialize(init_train_loader, distance=distance, n_sampling_olis=n_sampling_olis,
                            init=kmeans_init, use_cuda=use_cuda)
        else:
            self.initialize(train_loader, distance=distance, n_sampling_olis=n_sampling_olis,
                            init=kmeans_init, use_cuda=use_cuda)

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
        best_epoch = 0
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
            # (do that iff classification layer is LinearMax)
            if isinstance(self.classifier, LinearMax):
                self.unsup_train_classifier(train_loader, criterion, use_cuda=use_cuda)

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
                for oli, target, *_ in data_loader[phase]:
                    size = oli.size(0)
                    target = target.float()

                    # if the computations take place on the GPU, send data to GPU
                    if use_cuda:
                        oli = oli.cuda()
                        target = target.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward propagation through the model
                    #   -> do not keep track of the gradients if we are in the validation phase
                    if phase == 'val':
                        with torch.no_grad():
                            output = self(oli)

                            # create prediction tensor
                            pred = target.new_zeros(output.shape)
                            for i in range(output.shape[0]):
                                pred[i, category_from_output(output[i, :])] = 1

                            # multiclass prediction needs special call of loss function
                            if self.num_classes > 2 or isinstance(criterion, ClassBalanceLoss):
                                loss = criterion(output, target.argmax(1))
                            else:
                                loss = criterion(output, target)
                    else:
                        output = self(oli)

                        # create prediction tensor
                        pred = target.new_zeros(output.shape)
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
                        #torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                        optimizer.step()
                        self.normalize_()

                    # update statistics
                    running_loss += loss.item() * size
                    running_corrects += torch.sum(torch.sum(pred == target.data, 1) ==
                                                  target.new_ones(pred.shape[0]) * self.num_classes).item()

                # calculate loss and accuracy in the current epoch
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)

                # print the statistics of the current epoch
                list_acc[phase].append(epoch_acc)
                list_loss[phase].append(epoch_loss)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if (phase == 'val') and epoch_loss < best_loss:
                    best_epoch = epoch + 1
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    # store model parameters only if the generalization error improved (i.e. early stopping)
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            toc = timer()
            print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # report training results
        print('Finish at epoch: {}'.format(epoch + 1))
        print('Best epoch: {} with Acc = {:4f} and loss = {:4f}'.format(best_epoch, best_acc, best_loss))

        # if early stopping is enabled, make sure that the parameters are used, which resulted in the best
        # generalization error
        if early_stop:
            self.load_state_dict(best_weights)

        return list_acc, list_loss
