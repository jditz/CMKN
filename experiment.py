################################################
# Example experiment using a CON model.        #
#                                              #
# Author: Jonas Ditz                           #
# Contact: ditz@informatik.uni-tuebingen.de    #
################################################

import os
import argparse

from torch.utils.data import DataLoader
import torch
from torch import nn, autograd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
import numpy as np

from con import CON, CON2, CONDataset, kmer2dict, build_kmer_ref, compute_metrics, plot_grad_flow, Hook# register_hooks

import matplotlib.pyplot as plt
from timeit import default_timer as timer


# MACROS
DEBUGGING = True


name = 'example_experiment'
data_dir = './data/'


# extend custom data handler for the used dataset
class CustomHandler(CONDataset):
    def __init__(self, filepath):
        super(CustomHandler, self).__init__(filepath, alphabet='PROTEIN_AMBI')

        self.all_categories = ['H', 'M', 'L']

    def __getitem__(self, idx):
        # get a sample using the parent __getitem__ function
        sample = super(CustomHandler, self).__getitem__(idx)

        # initialize label tensor
        if len(sample[1]) == 1:
            labels = torch.zeros(3)
        else:
            labels = torch.zeros(len(sample[1]), 3)

        # iterate through all id strings and update the label tensor, accordingly
        for i, id_str in enumerate(sample[1]):
            try:
                aux_lab = id_str.split('|')[7]
                if aux_lab != 'NA':
                    if len(sample[1]) == 1:
                        labels[self.all_categories.index(aux_lab)] = 1.0
                    else:
                        labels[i, self.all_categories.index(aux_lab)] = 1.0

            except:
                continue

        # return the sample with updated label tensor
        sample = (sample[0], labels)
        return sample


def load_args():
    """
    Function to create an argument parser
    """
    parser = argparse.ArgumentParser(description="CON example experiment")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='M',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument("--num-anchors", dest="n_anchors", metavar="m", default=[64], nargs='+', type=int,
                        help="number of anchor points for each layer (default [64])")
    parser.add_argument("--subsamplings", dest="subsamplings", metavar="s", default=[1], nargs='+', type=int,
                        help="subsampling for each layer (default: [1])")
    parser.add_argument("--sigma", dest="kernel_params", default=[0.3], nargs='+', type=float,
                        help="sigma for each layer (default: [0.3])")
    parser.add_argument("--scale", dest="scale_param", default=100, type=int,
                        help="scaling parameter for the convolutional kernel layer (Default: 100)")
    parser.add_argument("--sampling-patches", dest="n_sampling_patches", default=250000, type=int,
                        help="number of sampled patches (default: 250000)")
    parser.add_argument("--kfold", dest="kfold", default=5, type=int, help="k-fold cross validation (default: 5)")
    parser.add_argument("--ntrials", dest="ntrials", default=1, type=int,
                        help="number of trials for training (default: 1)")
    parser.add_argument("--penalty", metavar="penal", dest="penalty", default='l2', type=str, choices=['l2', 'l1'],
                        help="regularization used in the last layer (default: l2)")
    parser.add_argument("--outdir", metavar="outdir", dest="outdir", default='output', type=str,
                        help="output path(default: '')")
    parser.add_argument("--regularization", type=float, default=1e-6, help="regularization parameter for CON model")
    parser.add_argument("--preprocessor", type=str, default='standard_row', choices=['standard_row', 'standard_col'],
                        help="preprocessor for CON (default: standard_row)")
    parser.add_argument("--use-cuda", action='store_true', default=False, help="use gpu (default: False)")
    parser.add_argument("--pooling", default='sum', choices=['mean', 'max', 'sum'], type=str,
                        help='specifies global pooling layer (default: sum)')
    parser.add_argument("--noise", type=float, default=0.0, help="perturbation percent")

    # parse the arguments
    args = parser.parse_args()

    # GPU will only be used if available
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    # 'num-anchors', 'subsampling', and 'sigma' have to be of same length
    if not len(args.n_anchors) == len(args.subsampling) == len(args.kernel_params):
        raise ValueError('Mismatching lengths!\nVectors given for the arguments --num-anchors, --subsampling, and '
                         '--sigma have to be of same length')

    # number of layers is equal to length of the 'num-anchors' vector
    args.n_layer = len(args.n_anchors)

    # set the random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # if an output directory is specified, create the dir structure to store the output of the current run
    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/{}".format(name)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/noise_{}".format(args.noise)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/{}".format(args.pooling)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/{}_{}_{}'.format(
            args.n_layers, args.n_anchors, args.subsamplings)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/{}'.format(args.kernel_params)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        args.outdir = outdir

    return args


def test_exp():
    # set parameters for the test run
    filepath = './data/test_dataset.fasta'
    extension = 'fasta'
    kmer_size = 3
    alphabet = 'ARNDCQEGHILKMFPSTWYVXBZJUO'

    # create dictionary that maps kmers to index
    kmer_dict = kmer2dict(kmer_size, alphabet)

    # build tensor holding reference positions
    ref_pos = build_kmer_ref(filepath, extension, kmer_dict, kmer_size)

    # initialize con model
    #model = CON([40, 128], ref_pos, [3], [1, 3], num_classes=3, kernel_funcs=['exp', 'exp_chen'],
    #            kernel_args_list=[[1.25, 1], [0.5]], kernel_args_trainable=[False, False])
    #model = CON([40], ref_pos, [], [1], num_classes=3, kernel_funcs=['exp'],
    #            kernel_args_list=[[0.5, 1]], kernel_args_trainable=[False])
    model = CON2(out_channels_list=[40, 40, 30, 20, 10], ref_kmerPos=ref_pos, filter_sizes=[3, 3, 5, 3],
                 strides=[1, 3, 3, 5, 3], paddings=['SAME', 'SAME', 'SAME', 'SAME'], num_classes=3)

    # load data
    data = CustomHandler(filepath)

    # Creating data indices for training and validation splits:
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PyTorch data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    loader_train = DataLoader(data, batch_size=4, sampler=train_sampler)
    loader_val = DataLoader(data, batch_size=4, sampler=val_sampler)

    # initialize optimizer and loss function
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.con_model.parameters(), lr=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

    # DEBUGGING START
    if DEBUGGING:

        model.initialize(loader_train)
        return

        # iterate over all parameter
        print(model)

        for name, parameter in model.named_parameters():
            print(name, parameter, "required_grad = {}".format(parameter.requires_grad))

        # register forward and backward hooks
        hookF = [Hook(list(list(model._modules.items())[0][1])[0])]
        hookF.append(Hook(list(list(model._modules.items())[0][1])[1]))
        hookF.append(Hook(list(model._modules.items())[1][1]))
        hookF.append(Hook(list(model._modules.items())[2][1]))
        hookB = [Hook(list(list(model._modules.items())[0][1])[0], backward=True)]
        hookB.append(Hook(list(list(model._modules.items())[0][1])[1], backward=True))
        hookB.append(Hook(list(model._modules.items())[1][1], backward=True))
        hookB.append(Hook(list(model._modules.items())[2][1], backward=True))

        for data, target, *_ in loader_train:
            # with autograd.detect_anomaly():
            #     # perform one forward step
            #     out = model(data)
            #
            #     # register hooks on the out tensor
            #     #get_dot = register_hooks(out)
            #
            #     # get the dot and save it
            #     #dot = get_dot()
            #     #dot.save('tmp.dot')
            #
            #     #print(target.shape[1])
            #
            #     # backprop to get backward hooks
            #     #out.backward(target, retain_graph=True)
            #     loss = criterion(out, target.argmax(1))
            #     loss.backward()

            out = model(data)
            loss = criterion(out, target.argmax(1))
            loss.backward()
            plot_grad_flow(model.named_parameters())

            # print hooks
            print()
            print('***' * 4 + '  Forward Hooks Inputs & Outputs  ' + '***' * 4 + '\n')
            for hook in hookF:
                try:
                    print([x.shape for x in hook.input])#(hook.input)#.shape)
                except:
                    print(hook.input[0].shape, hook.input[1:])
                print(hook.output.shape)#.shape)
                print('\n' + '---' * 27 + '\n')
            print('\n')
            print('***' * 4 + '  Backward Hooks Inputs & Outputs  ' + '***' * 4 + '\n')
            for hook in hookB:
                print([x.shape for x in hook.input])#(hook.input)#.shape)
                print([x.shape for x in hook.output])#(hook.output)#.shape)
                print('\n' + '---' * 27 + '\n')

            # print the output of the CON layer as a heatmap
            print('\nThe correct labels are: {}'.format(target))
            items = [0, 1, 2, 3]
            fig, axs = plt.subplots(2, 2)
            for item, ax in zip(items, axs.ravel()):
                im = ax.imshow(hookF[0].output[item, :, :].detach().numpy(), cmap='hot', interpolation=None, aspect='auto')
                ax.set_title("target = %s" % str(target[item, :]))
                fig.colorbar(im, ax=ax)
            plt.show()

            break

        return

    # DEBUGGING END

    # train model
    model.sup_train(loader_train, criterion, optimizer, lr_scheduler, val_loader=loader_val, epochs=5)

    # iterate through dataset
    #for i_batch, sample_batch in enumerate(loader):
    #    print('shape of batch {}: {}'.format(i_batch, sample_batch[0].shape))
    #    print('shape of target {}: {}'.format(i_batch, sample_batch[1].shape))
    #    print('')

    # perform "testing" using the validation data point (only for debugging purpose)
    y_pred, y_true = model.predict(loader_val, proba=True)
    scores = sum(y_pred == y_true) / len(y_pred)

    # compute_metrics might only work for binary classification
    #scores = compute_metrics(y_pred, y_true)

    print(scores)


def main(filepath):
    print('main')


if __name__ == '__main__':
    #main('./data/test_dataset.fasta')
    test_exp()
