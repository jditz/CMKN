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
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
import numpy as np

from con import CON, CONDataset

from timeit import default_timer as timer

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
        labels = torch.zeros(len(sample['label']), 3)

        # iterate through all id strings and update the label tensor, accordingly
        for i, id_str in enumerate(sample['label']):
            try:
                aux_lab = id_str.split('|')[7]
                if aux_lab != 'NA':
                    labels[i, self.all_categories.index(aux_lab)] = 1.0

            except:
                continue

        # return the sample with updated label tensor
        sample = {'data': sample['data'], 'label': labels}
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


def main(filepath):
    data = CustomHandler(filepath)
    loader = DataLoader(data, batch_size=4, shuffle=True)

    for i_batch, sample_batched in enumerate(loader):
        print(i_batch, sum(sum(sum(sample_batched['data']))), sample_batched['label'])
        break


if __name__ == '__main__':
    main('./data/processed_PI_DataSet_sample_labels.fasta')
