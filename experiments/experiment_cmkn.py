#############################################
# This file contains scripts to perform the #
# Convolutional Oligo Kernel Network        #
# experiments.                              #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import os
import argparse
import pickle

from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from cmkn import (CMKN, CMKNDataset, ClassBalanceLoss, kmer2dict, build_kmer_ref_from_file, build_kmer_ref_from_list,
                  compute_metrics, create_consensus, oli2number)

from Bio import SeqIO


# MACROS

NAME = 'CMKN_experiment'
DATA_DIR = '../data/'


# extend custom data handler for the used dataset
class CustomHandler(CMKNDataset):
    def __init__(self, filepath, kmer_size=3, drug='SQV', nb_classes=2, clean_set=True, encode='onehot'):
        self.drug_nb = {'FPV': 1, 'ATV': 2, 'IDV': 3, 'LPV': 4, 'NFV': 5, 'SQV': 6, 'TPV': 7, 'DRV': 8,
                        '3TC': 1, 'ABC': 2, 'AZT': 3, 'D4T': 4, 'DDI': 5, 'TDF': 6,
                        'EFV': 1, 'NVP': 2, 'ETR': 3, 'RPV': 4}
        self.drug = drug

        if clean_set:
            aux_tup = (self.drug_nb[drug], 'NA')
        else:
            aux_tup = None

        super(CustomHandler, self).__init__(filepath, kmer_size=kmer_size, alphabet='PROTEIN_FULL',
                                            clean_set=aux_tup, seq_encode=encode)

        self.nb_classes = nb_classes
        if nb_classes == 2:
            self.class_to_idx = {'L': 0, 'M': 1, 'H': 1}
            self.all_categories = ['susceptible', 'resistant']
        else:
            self.class_to_idx = {'L': 0, 'M': 1, 'H': 2}
            self.all_categories = ['L', 'M', 'H']

    def __getitem__(self, idx):
        # get a sample using the parent __getitem__ function
        sample = super(CustomHandler, self).__getitem__(idx)

        # initialize label tensor
        if len(sample[1]) == 1:
            labels = torch.zeros(self.nb_classes)
        else:
            labels = torch.zeros(len(sample[1]), self.nb_classes)

        # iterate through all id strings and update the label tensor, accordingly
        for i, id_str in enumerate(sample[1]):
            try:
                aux_lab = id_str.split('|')[self.drug_nb[self.drug]]
                if aux_lab != 'NA':
                    if len(sample[1]) == 1:
                        labels[self.class_to_idx[aux_lab]] = 1.0
                    else:
                        labels[i, self.class_to_idx[aux_lab]] = 1.0

            except:
                continue

        # return the sample with updated label tensor
        sample = (sample[0], labels)
        return sample


# custom handler for the HIV dataset used by Steiner et al., 2020
class HivHandler(CMKNDataset):
    def __init__(self, filepath, kmer_size=3):
        super(HivHandler, self).__init__(filepath, kmer_size=kmer_size, alphabet='PROTEIN_FULL')

    def __getitem__(self, idx):
        # get a sample using the parent __getitem__ function
        sample = super(HivHandler, self).__getitem__(idx)

        # initialize label tensor
        if len(sample[1]) == 1:
            labels = torch.zeros(2)
        else:
            labels = torch.zeros(len(sample[1]), 2)

        # iterate through all id strings and update the label tensor, accordingly
        for i, id_str in enumerate(sample[1]):
            try:
                aux_lab = int(id_str.split('_')[1])
                if len(sample[1]) == 1:
                    labels[aux_lab] = 1.0
                else:
                    labels[i, aux_lab] = 1.0

            except Exception as e:
                print('Exception at index {}:'.format(idx), e)
                continue

        # return the sample with updated label tensor
        sample = (sample[0], labels)
        return sample


# custom handler for the ENCODE dataset used by DeepBind and CKN papers
class EncodeHandler(CMKNDataset):
    def __init__(self, datadir, ext='seq.gz', kmer_size=8, tfid=None, nb_classes=2, encode='encode'):
        super(EncodeHandler, self).__init__(datadir, ext=ext, kmer_size=kmer_size, tfid=tfid, encode=encode)

        self.nb_classes = nb_classes

    def __getitem__(self, idx):
        sample = super(EncodeHandler, self).__getitem__(idx)

        # initialize label tensor
        if len(sample[1]) == 1:
            labels = torch.zeros(self.nb_classes)
        else:
            labels = torch.zeros(len(sample[1]), self.nb_classes)

        # iterate over each samples and update the labels tensor, accordingly
        for i, y in enumerate(sample[1]):
            if len(sample[1]) == 1:
                labels[y] = 1.0
            else:
                labels[i, y] = 1.0

        # return the sample with updated label tensor
        sample = (sample[0], labels)
        return sample


def load_args():
    """
    Function to create an argument parser
    """
    parser = argparse.ArgumentParser(description="CON example experiment")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--type', dest='type', default='HIV', type=str, choices=['HIV'],
                        help="specify the type of experiment, i.e. the used dataset. Currently, only 'HIV' is "
                             "supported.")
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=64, metavar='M',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', dest="nb_epochs", type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument("--out-channels", dest="out_channels", metavar="m", default=[99], nargs='+',
                        type=int,
                        help="number of out channels for each oligo kernel and conv layer (default [40])")
    parser.add_argument("--strides", dest="strides", metavar="s", default=[1], nargs='+', type=int,
                        help="stride value for each layer (default: [1])")
    parser.add_argument("--paddings", dest="paddings", metavar="p", default=['SAME'], nargs="+",
                        type=str, help="padding type for each convolutional layer (default ['SAME'])")
    parser.add_argument("--kernel-sizes", dest="kernel_sizes", metavar="k", default=[1], nargs="+", type=int,
                        help="kernel sizes for oligo and convolutional layers (default [1])")
    parser.add_argument("--sigma", dest="sigma", default=1, type=float,
                        help="sigma for the oligo kernel layer (default: 4)")
    parser.add_argument("--alpha", dest="alpha", default=1, type=int,
                        help="alpha parameter used in the exponential function of the oligo layer; if set to -1, the "
                             "value will be depending on the number of oligomers (default: -1)")
    parser.add_argument("--scale", dest="scale", default=-1, type=int,
                        help="scaling parameter for the oligo kernel layer (default: -1). If set to -1, the scaling "
                             "parameter will be determined depending on the length of the input sequences.")
    parser.add_argument("--num-classes", dest="num_classes", default=2, type=int,
                        help="number of classes in the prediction task")
    parser.add_argument("--kfold", dest="kfold", default=5, type=int, help="k-fold cross validation (default: 5)")
    parser.add_argument("--penalty", metavar="penal", dest="penalty", default='l2', type=str, choices=['l2', 'l1'],
                        help="regularization used in the last layer (default: l2)")
    parser.add_argument("--regularization", type=float, default=1e-6,
                        help="regularization parameter for sup CON")
    parser.add_argument("--preprocessor", type=str, default='standard_row',
                        help="preprocessor for last layer of CON (default: standard_row). Set to 'standard_row' or "
                             "'standard_column' in order to use a LinearMixin layer. Or set it to the length of input"
                             "sequences to use a standard fully-connected layer.")
    parser.add_argument("--outdir", metavar="outdir", dest="outdir", default='output', type=str,
                        help="output path(default: '')")
    parser.add_argument("--use-cuda", action='store_true', default=True, help="use gpu (default: False)")
    parser.add_argument("--noise", type=float, default=0.0, help="perturbation percent")
    parser.add_argument("--file", dest="filepath", default="./data/hivdb", type=str,
                        help="path to the file containing the dataset.")
    parser.add_argument("--extension", dest="extension", default="fasta", type=str,
                        help="extension of the file containing the dataset (default fasta)")
    parser.add_argument("--drug", dest="drug", default="SQV", type=str,
                        help="specifies the drug that will be used to classify virus resilience (default SQV)")
    parser.add_argument("--encodeset", dest="encodeset", default="CTCF_A549_CTCF_UT-A", type=str,
                        help="specifies which ENCODE dataset will be used for training a model; if set to 'optim', " +
                             "the hyperparameters will be optimized using 100 randomly selected ENCODE datasets")

    # parse the arguments
    args = parser.parse_args()

    # GPU will only be used if available
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    # store length of oligomers for a simple rpoxy access
    args.kmer_size = args.kernel_sizes[0]

    # 'num-anchors', 'subsampling', and 'sigma' have to be of same length
    if not len(args.out_channels) == len(args.strides) == len(args.paddings) == len(args.kernel_sizes):
        raise ValueError('The size combination of out_channels, strides, paddings, and kernel_sizes is invalid!\n' +
                         '    out_channels, strides, paddings, and kernel_sizes must have the same size')

    # number of layers is equal to length of the 'num-anchors' vector
    args.n_layer = len(args.out_channels)

    # make sure that args.prerocessor is set up, properly
    try:
        args.preprocessor = int(args.preprocessor)
    except ValueError:
        if args.preprocessor not in ['standard_row', 'standard_column']:
            raise ValueError('preprocessor must be either an Integer or one of the following strings: standard_row, '
                             'standard_column')

    # for a HIV experiment, also store the type of antiviral drug
    if args.type == 'HIV':
        if args.drug in ['FPV', 'ATV', 'IDV', 'LPV', 'NFV', 'SQV', 'TPV', 'DRV']:
            args.drug_type = 'PI'
        elif args.drug in ['3TC', 'ABC', 'AZT', 'D4T', 'DDI', 'TDF']:
            args.drug_type = 'NRTI'
        else:
            args.drug_type = 'NNRTI'

        # add the correct file to the given filepath
        args.filepath = args.filepath.rstrip(os.path.sep) + '{}{}_DataSet.fasta'.format(os.path.sep, args.drug_type)

    # for an ENCODE experiment, make sure that the path to the datafiles does not have a trailing path separator
    elif args.type == 'ENCODE':
        args.filepath = args.filepath.rstrip(os.path.sep)

    # set the random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # if an output directory is specified, create the dir structure to store the output of the current run
    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True
        if args.type == 'HIV':
            aux_dir = args.drug
        else:
            aux_dir = args.encodeset
        aux_out = '/{}/{}/classes_{}/kmer_{}/params_{}_{}_{}/anchors_{}/layers_{}'.format(NAME, aux_dir,
                                                                                          args.num_classes,
                                                                                          args.kmer_size, args.sigma,
                                                                                          args.scale, args.alpha,
                                                                                          args.out_channels[0],
                                                                                          args.n_layer)
        args.outdir += aux_out
        if not os.path.exists(args.outdir):
            try:
                os.makedirs(args.outdir)
            except:
                pass

    return args


def count_classes_hiv(filepath, verbose=False, drug=7, num_classes=2):
    class_count = [0] * num_classes
    label_vec = []
    nb_samples = 0

    with open(filepath, 'rU') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if drug == -1:
                aux_lab = int(record.id.split('_')[1])
                class_count[aux_lab] += 1
                label_vec.append(aux_lab)
            else:
                aux_lab = record.id.split('|')[drug]
                if num_classes == 2:
                    if aux_lab == 'H':
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == 'M':
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == 'L':
                        class_count[0] += 1
                        label_vec.append(0)
                    else:
                        continue
                else:
                    if aux_lab == 'H':
                        class_count[2] += 1
                        label_vec.append(2)
                    elif aux_lab == 'M':
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == 'L':
                        class_count[0] += 1
                        label_vec.append(0)
                    else:
                        continue
            nb_samples += 1

    if verbose:
        print("number of samples in dataset: {}".format(nb_samples))
        if num_classes == 2:
            print("class balance: resistant = {}, susceptible = {}".format(class_count[1] / nb_samples,
                                                                           class_count[0] / nb_samples))
        else:
            print("class balance: H = {}, M = {}, L = {}".format(class_count[2] / nb_samples, class_count[1] /
                                                                 nb_samples, class_count[0] / nb_samples))

    if num_classes == 2:
        expected_loss = -(class_count[0] / nb_samples) * np.log(0.5) - \
                        (class_count[1] / nb_samples) * np.log(0.5)
    else:
        expected_loss = -(class_count[0] / nb_samples) * np.log(0.33) - \
                        (class_count[1] / nb_samples) * np.log(0.33) -\
                        (class_count[2] / nb_samples) * np.log(0.33)

    if verbose:
        print("expected loss: {}".format(expected_loss))

    if num_classes == 2:
        rand_guess = (class_count[0] / nb_samples) ** 2 + (class_count[1] / nb_samples) ** 2
    else:
        rand_guess = (class_count[0] / nb_samples) ** 2 + (class_count[1] / nb_samples) ** 2 + \
                     (class_count[2] / nb_samples) ** 2

    if verbose:
        print("expected accuracy with random guessing: {}".format(rand_guess))

    return class_count, expected_loss, np.array(label_vec)


def count_classes_encode(labels, verbose=True):
    class_count = [0, 0]
    nb_samples = 0

    for label in labels:
        nb_samples += 1
        class_count[label] += 1

    if verbose:
        print("number of samples in dataset: {}".format(nb_samples))
        print("class balance: pos = {}, neg = {}".format(class_count[1] / nb_samples, class_count[0] / nb_samples))

    expected_loss = -(class_count[0] / nb_samples) * np.log(0.5) - (class_count[1] / nb_samples) * np.log(0.5)

    if verbose:
        print("expected loss: {}".format(expected_loss))

    rand_guess = (class_count[0] / nb_samples) ** 2 + (class_count[1] / nb_samples) ** 2

    if verbose:
        print("expected accuracy with random guessing: {}".format(rand_guess))

    return class_count, expected_loss


def train_hiv(args):
    args.alphabet = 'ARNDCQEGHILKMFPSTWYVXBZJUO'

    # load data
    data_all = CustomHandler(args.filepath, kmer_size=args.kmer_size, drug=args.drug, encode='onehot')

    # determine if oligo kernel's scale parameter should be set depending on sequence length
    if args.scale == -1:
        args.scale = len(data_all.data[0]) * (len(data_all.data[0]) / 10)

    # set random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load indices for 5-fold cross-validation
    fold_filepath = args.filepath.split('/')[:-1]
    fold_filepath = '/'.join(fold_filepath) + '/hivdb_stratifiedFolds.pkl'
    with open(fold_filepath, 'rb') as in_file:
        folds = pickle.load(in_file)
        folds = folds[args.drug_type][args.drug]

    # perform 5-fold stratified cross-validation using the predefined folds
    for fold_nb, fold in enumerate(folds):
        # store training and validation indices for the current fold
        args.train_indices, args.val_indices = fold[0], fold[1]

        # initialize con model
        model = CMKN(in_channels=len(args.alphabet), out_channels_list=args.out_channels,
                     filter_sizes=args.kernel_sizes, strides=args.strides, paddings=args.paddings,
                     num_classes=args.num_classes, kernel_args=[args.sigma, args.scale, args.alpha],
                     scaler=args.preprocessor, pool_global=None)

        # get labels of each entry for stratified shuffling and distribution of classes for class balance loss
        args.class_count, args.expected_loss, _ = count_classes_hiv(args.filepath, True, data_all.drug_nb[args.drug],
                                                                    args.num_classes)

        # set arguments for the DataLoader
        loader_args = {}
        if args.use_cuda:
            loader_args = {'num_workers': 1, 'pin_memory': True}

        # Creating PyTorch data Subsets using the indices for the current fold
        data_train = Subset(data_all, args.train_indices)
        data_val = Subset(data_all, args.val_indices)

        # create PyTorch DataLoader for training and validation data
        loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=False, **loader_args)
        loader_val = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, **loader_args)

        # initialize optimizer and loss function
        if args.num_classes == 2:
            criterion = ClassBalanceLoss(args.class_count, args.num_classes, 'sigmoid', 0.99, 1.0)
        else:
            criterion = ClassBalanceLoss(args.class_count, args.num_classes, 'cross_entropy', 0.99, 1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

        # train model
        if args.use_cuda:
            model.cuda()
        acc, loss = model.sup_train(loader_train, criterion, optimizer, lr_scheduler, epochs=args.nb_epochs,
                                    early_stop=False, use_cuda=args.use_cuda, kmeans_init='kmeans++',
                                    distance='euclidean')

        # compute performance metrices on validation data
        pred_y, true_y = model.predict(loader_val, proba=True, use_cuda=args.use_cuda)
        scores = compute_metrics(true_y, pred_y)

        # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
        # model
        torch.save({'args': args, 'state_dict': model.state_dict(), 'acc': acc, 'loss': loss,
                    'val_performance': scores.to_dict()},
                   args.outdir + "/CON_results_epochs" + str(args.nb_epochs) + "_fold" + str(fold_nb) + ".pkl")

        try:
            # try to import pyplot
            import matplotlib.pyplot as plt

            # show the evolution of the acc and loss
            fig2, axs2 = plt.subplots(2, 2)
            axs2[0, 0].plot(acc['train'])
            axs2[0, 0].set_title("train accuracy")
            axs2[0, 0].set(xlabel='epoch', ylabel='accuracy')
            axs2[0, 1].plot(acc['val'])
            axs2[0, 1].set_title("val accuracy")
            axs2[0, 1].set(xlabel='epoch', ylabel='accuracy')
            axs2[1, 0].plot(loss['train'])
            axs2[1, 0].set_title("train loss")
            axs2[1, 0].set(xlabel='epoch', ylabel='loss')
            axs2[1, 1].plot(loss['val'])
            axs2[1, 1].set_title("val loss")
            axs2[1, 1].set(xlabel='epoch', ylabel='loss')
            # plt.show()
            plt.savefig(args.outdir + "/acc_loss.png")

            # show the position of the anchor points as a histogram
            anchor = (torch.acos(model.cmkn_layer.pos_anchors[:, 0]) / np.pi) * (len(data_all.data[0]) - 1)
            anchor = anchor.detach().cpu().numpy()
            fig3 = plt.figure()
            fig3.set_size_inches(w=20, h=10)
            plt.hist(anchor, bins=range(len(data_all.data[0])))
            # plt.xlim([0, ref_oli.size(1)])
            plt.xlabel('Position')
            plt.ylabel('# Anchor Points')
            plt.title('Distribution of anchor points')
            # plt.show()
            plt.savefig(args.outdir + "/anchor_positions.png")

        except ImportError:
            print("Cannot import matplotlib.pyplot")

        except Exception as e:
            print("Unexpected error while trying to plot training visualisation:")
            print(e)


def main():
    # read parameter
    args = load_args()

    if args.type == 'HIV':
        train_hiv(args)
    else:
        raise ValueError('Unknown experiment! Received the following argument: {}'.format(args.type))


if __name__ == '__main__':
    main()
