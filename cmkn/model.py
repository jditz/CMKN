"""Module that contains a basic implementation of a convolutional motif kernel network (CMKN).

Authors:
    Jonas C. Ditz: jonas.ditz@uni-tuebingen.de
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from timeit import default_timer as timer
import copy

from .layers import POOLINGS, CMKNLayer, LinearMax
from .utils import category_from_output, ClassBalanceLoss


class CMKN(nn.Module):
    """Convolutional Motif Kernel Network

    This class implements a simple example for a Convolutional Motif Kernel Network (CMKN). The model will consist of a
    single CMKNLayer followed by a variational number of traditional convolutional layers (using convolutional layer is
    optional). The classification can either be done with two fully connected layers or a combination of a global
    pooling layer and a LinearMax layer.
    """

    def __init__(self, in_channels, out_channels_list, filter_sizes, strides, paddings, kernel_func=None,
                 kernel_args=None, kernel_args_trainable=False, alpha=0., fit_bias=True, batch_norm=True, dropout=False,
                 pool_global=None, pool_conv='mean', penalty='l2', scaler=None, num_classes=1, **kwargs):
        """Constructor of the CON class.

        Args:
            in_channels (:obj:`int`): Dimensionality of the alphabet that generates the input sequences.
            out_channels_list (:obj:`list` of :obj:`int`): Number of output channels of each layer (CMKNLayer and
                Conv1d layer only).
            filter_sizes (:obj:`list` of :obj:`int`): Size of the filter for each layer. For the CMKNLayer this
                determines the length of motifs used for the kernel evaluation.
            strides (:obj:`list` of :obj:`int`): List of stride factors for each Conv1d layer. For the CMKNLayer this
                argument determines the subsampling property.
            paddings (:obj:`list` of :obj:`str`): List of padding factors for the CMKNLayer and each Conv1d layer.
            kernel_funcs (:obj:`str`): Specifies the kernel function used in the CMKNLayer. Defaults to None.
            kernel_args_list (:obj:`tuple` of :obj:`int`): Tuple of arguments for the used kernel. Defaults to None.
            kernel_args_trainable (:obj:`bool`): Indicates for the CMKNLayer if the kernel parameters used in this
                layer are trainable.
            alpha (:obj:`float`): Strength of the regularization in the classification layer (only used if the
                classification layer is a LinearMax layer)
            fit_bias (:obj:`bool`): Indicates whether the bias of the classification layer should be fitted.
            batchNorm (:obj:`bool`): Indicates whether batch normalization should be used for Covd1d layers.
            dropout (:obj:`bool`): Indicates whether Dropout should be used for Conv1d layers.
            global_pool (:obj:`str`): Indicates which method should be used for global pooling.
            pool_conv (:obj:`str`): Indicates the pooling layer used after each convolutional layer.
            penalty (:obj:`str`): Indicates which penalty method should be used for regularization in the classification
                layer (only used if the classfication layer is a LinearMax layer)
            scaler: If set to a String, a LinearMixin layer will be used for classification and the String
                specifies the used scalar, e.g. 'standard_row'. If set to an Integer, a standard fully
                connected layer will be used for classification and the Integer determines the input
                dimensionality of that layer.
            num_classes (:obj:`int`): Number of classes in the current classification problem

        Raises:
            ValueError: If out_channels_list, filter_sizes, strides, and paddings are not of the same length.
        """

        # check if out_channels_list, strides, filter_sizes, and paddings have an acceptable length
        #   -> therefore, raise an AssertionError if the lengths differ
        if not len(out_channels_list) == len(strides) == len(filter_sizes) == len(paddings):
            raise ValueError('Incompatible dimensions! \n'
                             '            out_channels_list, filter_sizes, strides and paddings must have the same '
                             'length!')

        # initialize parent class
        super(CMKN, self).__init__()

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

        # initialize the Convolutional Motif Kernel Layer
        self.cmkn_layer = CMKNLayer(in_channels, out_channels_list[0], filter_sizes[0], padding=paddings[0],
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
        """ Function to access the normalization routine of the CMKNLayer
        """
        self.cmkn_layer.normalize_()

    def representation(self, x_in, mask=None):
        """ Function to combine CMKN layer and pooling layer evaluation

        Args:
            x_in (Tensor): Input to the model given as a Tensor of size (batch_size x self.in_channels x seq_len)
            mask (Tensor): Initial mask (usually not needed for CMKN models). Defaults to None

        Returns:
            Evaluation of the CMKN layer(s) with subsequent pooling using the given input. If no pooling layer was
            specified, the function will return the flatten output of the convolutional layer(s).
        """
        x_out = self.cmkn_layer(x_in)

        if self.nb_conv_layers > 0:
            x_out = self.conv(x_out)

        if self.global_pool is None:
            x_out = x_out.view(x_out.size(0), -1)
        else:
            x_out = self.global_pool(x_out, mask)

        return x_out

    def forward(self, x_in, proba=False):
        """ Overwritten forward function for CON objects

        Args:
            x_in (Tensor): Input to the model given as a Tensor of size (batch_size x self.in_channels x seq_len)
            proba (:obj:`bool`): Indicates whether the network should produce probabilistic output

        Returns:
            Result of a forward pass through the model using the specified input. If 'proba' is set to True, the output
            will be the probabilities assigned to each class by the model.
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

        To call this function is only needed if the classification layer is a LinearMax layer. Otherwise, there is no
        need for initializing the fully connected layer by an unsupervised training procedure. Standard initialization
        protocols for fully connected layers can be used instead.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch DataLoader that handles data
            criterion (Pytorch Loss Object): Specifies the loss function. If set to None, torch.nn.BCEWithLogitsLoss()
                will be used.
            use_cuda (:obj:`bool`): Specified whether all computations will be performed on the GPU
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

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch DataLoader that handles data
            only_representation (:obj:`bool`): If set to True, the function will return the input to the classification
                layer. Otherwise the result of a forward pass is returned. Defaults to False.
            proba (:obj:`bool`): Indicates whether the network should produce probabilistic output.
            use_cuda (:obj:`bool`): Specified whether all computations will be performed on the GPU

        Returns:
            The computed output for each sample in the DataLoader together with the real target output of each sample.
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

    def initialize(self, data_loader, distance, n_sampling_motifs=100000, init=None, max_iters=100, use_cuda=False):
        """ Function to initialize parameters of the network
        
        This function initializes the convolutional layers of a CMKN model. The CMKNLayer is initialized by, first, 
        sampling a specified number of motifs or motif-position pairs (depending on selected distance) from the 
        training data and, second, performing k-Means (or k-Means++) clustering on the sampled data to initialize
        anchor points with the cluster centers. Traditional convolutional layers are initialized using the He 
        initialization.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch DataLoader object that handles access to training data
            distance (:obj:`str`): Distance measure used in the k-Means algorithm.
            n_sampling_motifs (:obj:`int`): Number of motifs or motif-position pairs that will be sampled to initialize 
                anchor points using the k-Means algorithm. Defaults to 100000.
            init (:obj:`str`): Initialization parameter for the k-Means algorithm. Defaults to None.
            max_iters (:obj:`int`): Maximal number of iterations used in the K-Means clustering. Defaults to 100.
            use_cuda (:obj:`bool`): Parameter to determine whether to do calculations on the GPU or CPU. Defaults to 
                False.
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
            n_oligomers_per_batch = (n_sampling_motifs + len(data_loader) - 1) // len(data_loader)
        except:
            n_oligomers_per_batch = 1000

        # depending on the chosen distance measure for the k-Means algorithm, oligomers will incorporate positional
        # information
        if distance == 'euclidean':
            # initialize tensor that stores the sampled oligomers and make sure it is on the same device as the model
            oligomers = self.cmkn_layer.weight.new_zeros(n_sampling_motifs, self.cmkn_layer.patch_dim + 1)

            # make sure that oligomers contain positional information
            include_pos = True
        else:
            # initialize tensor that stores the sampled oligomers and make sure it is on the same device as the model
            oligomers = self.cmkn_layer.weight.new_zeros(n_sampling_motifs, self.cmkn_layer.patch_dim)

            # make sure that oligomers will not contain positional information
            include_pos = False

        # get batches using the DataLoader object
        seq_len = None
        for data, _ in data_loader:
            # stop sampling oligomers if the maximum number of oligomers is already achieved
            if n_oligomers >= n_sampling_motifs:
                break

            # get the length of the sequences; this will be needed later
            if seq_len is None:
                seq_len = data.shape[-1]

            # send data to GPU if use_cuda flag was set
            if use_cuda:
                data = data.cuda()

            # sample the specified number of oligomers using the current batch of data
            with torch.no_grad():
                data_oliogmers = self.cmkn_layer.sample_oligomers(data, n_oligomers_per_batch, include_pos)

            # only use a subset of the sampled oligomers in this batch, if this batch would exceed the maximum number
            # of sampled oligomers
            size = data_oliogmers.size(0)
            if n_oligomers + size > n_sampling_motifs:
                size = n_sampling_motifs - n_oligomers
                data_oliogmers = data_oliogmers[:size]

            # update the oligomer tensor with the currently sampled oligomers and the amount of already sampled
            # oligomers
            oligomers[n_oligomers: n_oligomers + size] = data_oliogmers
            n_oligomers += size

        # initialize the oligomer and position anchor points of the Oligo Kernel layer
        print('        total number of sampled motifs: {}'.format(n_oligomers))
        self.cmkn_layer.initialize_weights(distance, oligomers, seq_len, init=init, max_iters=max_iters)

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
                  distance='euclidean', n_sampling_motifs=100000, kmeans_init=None, epochs=100, val_loader=None,
                  use_cuda=False, early_stop=True):
        """ Perform supervised training of the CON model

        This function will first initialize all convolutional layers of the model. Afterwards, a normal training routine
        for ANNs follows. If validation data is given, the performance on the validation data is calculate after each
        epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): PyTorch DataLoader that handles training data.
            criterion (PyTorch Loss Object): Specifies the loss function.
            optimizer (PyTorch Optimizer): Optimization algorithm used during training.
            lr_scheduler (PyTorch LR Scheduler): Algorithm used for learning rate adjustment. Defaults to None.
            init_train_loader (torch.utils.data.DataLoader): This DataLoader can be set if different datasets should be
                used for initializing the convolutional motif kernel layer of this model and training the model. Data
                accessed by this DataLoader will be used for initialization of CMK layers. Defaults to None.
            distance (:obj:`str`): Distance measure used in the k-Means algorithm during initialization of convolutional
                motif kernel layers. Defaults to 'euclidean'.
            n_sampling_motifs (:obj:`int`): Number of motifs that will be sampled to initialize oligomer anchor points
                using the k-Means algorithm. Defaults to 100000.
            kmeans_init (:obj:`str`): Initialization parameter for the spherical k-Means algorithm.
            epochs (:obj:`int`): Number of epochs during training. Defaults to 100.
            val_loader (torch.utils.data.DataLoader): PyTorch DataLoader that handles data during the validation phase.
                Defaults to None.
            use_cuda (:obj:`bool`): Specified whether all computations will be performed on the GPU. Defaults to False.
            early_stop (:obj:`bool`): Specifies if early stopping will be used during training. Defaults to True.

        Returns:
            Training and validation accuracies and losses of each epoch.
        """
        print("Initializing CON layers")
        tic = timer()

        # initialize the anchor points of all layers that use anchor points in an unsupervised fashion and initialize
        # weights of all convolutional layers
        if init_train_loader is not None:
            self.initialize(init_train_loader, distance=distance, n_sampling_motifs=n_sampling_motifs,
                            init=kmeans_init, use_cuda=use_cuda)
        else:
            self.initialize(train_loader, distance=distance, n_sampling_motifs=n_sampling_motifs,
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
