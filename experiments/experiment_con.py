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

from con import (CON, CONDataset, ClassBalanceLoss, kmer2dict, build_kmer_ref_from_file, build_kmer_ref_from_list,
                 compute_metrics, create_consensus, oli2number)

from Bio import SeqIO


# MACROS

NAME = 'CON_experiment'
DATA_DIR = './data/'


# extend custom data handler for the used dataset
class CustomHandler(CONDataset):
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
                                            clean_set=aux_tup, encode=encode)

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
class HivHandler(CONDataset):
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
class EncodeHandler(CONDataset):
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
    parser.add_argument('--type', dest='type', default='HIV', type=str, choices=['HIV', 'ENCODE'],
                        help="specify the type of experiment, i.e. the used dataset. Currently only 'HIV' and " +
                             "'ENCODE' are supported choices")
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=64, metavar='M',
                        help='input batch size for training (default: 4)')
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
    parser.add_argument("--scale", dest="scale", default=100, type=int,
                        help="scaling parameter for the oligo kernel layer (Default: 100)")
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
    parser.add_argument("--file", dest="filepath", default="./data/hivdb/", type=str,
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

    # for an HIV experiment, also store the type of antiviral drug
    if args.type == 'HIV':
        if args.drug in ['FPV', 'ATV', 'IDV', 'LPV', 'NFV', 'SQV', 'TPV', 'DRV']:
            args.drug_type = 'PI'
        elif args.drug in ['3TC', 'ABC', 'AZT', 'D4T', 'DDI', 'TDF']:
            args.drug_type = 'NRTI'
        else:
            args.drug_type = 'NNRTI'
        args.filepath += '{}_DataSet.fasta'.format(args.drug_type)

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

    # set random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load indices for 5-fold cross-validation
    fold_filepath = args.filepath.split('/')[:-1]
    fold_filepath = '/'.join(fold_filepath) + '/hivdb_stratifiedFolds.pkl'
    folds = pickle.load(fold_filepath)
    folds = folds[args.drug_type][args.drug]

    # perform 5-fold stratified cross-validation using the predefined folds
    for fold_nb, fold in enumerate(folds):
        # store training and validation indices for the current fold
        args.train_indices, args.val_indices = fold[0], fold[1]

        # initialize con model
        model = CON(in_channels=len(args.alphabet), out_channels_list=args.out_channels, filter_sizes=args.kernel_sizes,
                    strides=args.strides, paddings=args.paddings, num_classes=args.num_classes,
                    kernel_args=[args.sigma, args.scale, args.alpha], scaler=args.preprocessor, pool_global=None)

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
            anchor = (torch.acos(model.oligo.pos_anchors[:, 0]) / np.pi) * (len(data_all.data[0]) - 1)
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


def train_encode(args):
    args.alphabet = 'ACGTN'

    # specify the 100 randomly selected datasets used for hyper-parameter optimization
    if args.encodeset == "optim":
        train_ids = ['BHLHE40_A549_BHLHE40_Stanford', 'RAD21_H1-hESC_Rad21_HudsonAlpha', 'CTCF_AG04450_CTCF_UW',
                     'CTCF_HSMMtube_CTCF_Broad', 'CTCF_HCFaa_CTCF_UW', 'TEAD4_K562_TEAD4_(SC-101184)_HudsonAlpha',
                     'RXRA_HepG2_RXRA_HudsonAlpha', 'HDAC2_K562_HDAC2_(A300-705A)_Broad', 'CTCF_H1-hESC_CTCF_UT-A',
                     'SIX5_H1-hESC_SIX5_HudsonAlpha', 'SIRT6_K562_SIRT6_Harvard', 'HMGN3_K562_HMGN3_Harvard',
                     'TBP_HeLa-S3_TBP_Stanford', 'EZH2_HSMMtube_EZH2_(39875)_Broad', 'TAF1_K562_TAF1_HudsonAlpha',
                     'TAF1_SK-N-SH_TAF1_HudsonAlpha', 'EP300_SK-N-SH-(Retinoic-Acid)_p300_HudsonAlpha',
                     'HDAC2_K562_HDAC2_(A300-705A)_Broad', 'POLR2A_ProgFib_Pol2_UT-A', 'CTCF_NH-A_CTCF_Broad',
                     'EP300_HeLa-S3_p300_(SC-584)_Stanford', 'TCF3_GM12878_TCF3_(SC-349)_HudsonAlpha',
                     'YY1_HepG2_YY1_(SC-281)_HudsonAlpha', 'POLR2A_HCT-116_Pol2_USC', 'ETS1_K562_ETS1_HudsonAlpha',
                     'CTCF_NHEK_CTCF_UT-A', 'CTCF_RPTEC_CTCF_UW', 'EGR1_GM12878_Egr-1_HudsonAlpha',
                     'NR2C2_GM12878_TR4_USC', 'MYC_K562_c-Myc_Stanford', 'HNF4A_HepG2_HNF4A_(SC-8987)_HudsonAlpha',
                     'CTCF_HeLa-S3_CTCF_UT-A', 'E2F6_K562_E2F6_HudsonAlpha', 'BRCA1_GM12878_BRCA1_(A300-000A)_Stanford',
                     'TCF7L2_MCF-7_TCF7L2_USC', 'NFYB_GM12878_NF-YB_Harvard', 'CHD1_K562_CHD1_(A301-218A)_Broad',
                     'TAF1_SK-N-SH_TAF1_HudsonAlpha', 'RAD21_IMR90_Rad21_Stanford', 'CTCF_GM12872_CTCF_UW',
                     'YY1_GM12891_YY1_(SC-281)_HudsonAlpha', 'GATA3_MCF-7_GATA3_(SC-268)_USC',
                     'HDAC6_K562_HDAC6_(A301-341A)_Broad', 'MAZ_HepG2_MAZ_(ab85725)_Stanford',
                     'REST_K562_NRSF_HudsonAlpha', 'RCOR1_HeLa-S3_COREST_(sc-30189)_Stanford', 'CTCF_HMF_CTCF_UW',
                     'CTCF_MCF-7_CTCF_UT-A', 'TBL1XR1_GM12878_TBLR1_(ab24550)_Stanford', 'NFE2_K562_NF-E2_Yale',
                     'PHF8_K562_PHF8_(A301-772A)_Broad', 'EZH2_NHEK_EZH2_(39875)_Broad', 'CTCF_GM12865_CTCF_UW',
                     'MXI1_GM12878_Mxi1_(AF4185)_Stanford', 'POLR2A_GM12892_Pol2_HudsonAlpha',
                     'CHD1_K562_CHD1_(A301-218A)_Broad', 'RAD21_H1-hESC_Rad21_Stanford', 'CTCF_HEK293_CTCF_UW',
                     'MEF2C_GM12878_MEF2C_(SC-13268)_HudsonAlpha', 'SMC3_HepG2_SMC3_(ab9263)_Stanford',
                     'BHLHE40_HepG2_BHLHE40_(NB100-1800)_Stanford', 'POLR2A_HepG2_Pol2_UT-A',
                     'JUN_HeLa-S3_c-Jun_Stanford', 'REST_H1-hESC_NRSF_HudsonAlpha', 'E2F6_K562_E2F6_HudsonAlpha',
                     'ATF1_K562_ATF1_(06-325)_Harvard', 'EZH2_NHEK_EZH2_(39875)_Broad', 'MAX_A549_Max_Stanford',
                     'RXRA_GM12878_RXRA_HudsonAlpha', 'CTCF_HEEpiC_CTCF_UW', 'RXRA_HepG2_RXRA_HudsonAlpha',
                     'POLR2A_HUVEC_Pol2_HudsonAlpha', 'ELK1_GM12878_ELK1_(1277-1)_Stanford', 'JUN_HepG2_c-Jun_Stanford',
                     'CTCF_SAEC_CTCF_UW', 'MXI1_H1-hESC_Mxi1_(AF4185)_Stanford', 'SP2_H1-hESC_SP2_(SC-643)_HudsonAlpha',
                     'TBP_H1-hESC_TBP_Stanford', 'EZH2_HSMMtube_EZH2_(39875)_Broad', 'NFYB_GM12878_NF-YB_Harvard',
                     'GATA2_HUVEC_GATA-2_USC', 'CTBP2_H1-hESC_CtBP2_USC', 'UBTF_K562_UBTF_(SAB1404509)_Stanford',
                     'CTCF_Gliobla_CTCF_UT-A', 'CTCF_HCM_CTCF_UW', 'SRF_K562_SRF_HudsonAlpha', 'ELK4_HeLa-S3_ELK4_USC',
                     'SMC3_GM12878_SMC3_(ab9263)_Stanford', 'CTCF_GM12873_CTCF_UW', 'GTF2B_K562_GTF2B_Harvard',
                     'THAP1_K562_THAP1_(SC-98174)_HudsonAlpha', 'UBTF_K562_UBF_(sc-13125)_Stanford',
                     'EP300_SK-N-SH-(Retinoic-Acid)_p300_HudsonAlpha', 'MXI1_H1-hESC_Mxi1_(AF4185)_Stanford',
                     'CTCF_A549_CTCF_UT-A', 'POLR2A_GM12878_Pol2_UT-A', 'POLR2A_GM15510_Pol2_Stanford',
                     'CTCF_H1-hESC_CTCF_(SC-5916)_HudsonAlpha', 'CTCF_NHDF-Ad_CTCF_Broad',
                     'POLR2A_GM19099_Pol2_Stanford'
                     ]
    else:
        train_ids = [args.encodeset]

    # create ENCODE dataset handle
    data_all = EncodeHandler(args.filepath, kmer_size=args.kmer_size, encode='onehot')

    # iterate over all ids used in this trainings round
    for tfid in train_ids:
        # set random seeds
        torch.manual_seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # update the dataset to the correct transcription factor
        data_all.update_dataset(tfid)

        # Creating data indices for training and validation splits:
        validation_split = .2
        shuffle_dataset = True
        random_seed = args.seed
        dataset_size = len(data_all)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        args.train_indices, args.val_indices = indices[split:], indices[:split]

        # Creating PyTorch data samplers and loaders:
        data_train = Subset(data_all, args.train_indices)
        data_val = Subset(data_all, args.val_indices)

        # set arguments for the DataLoader
        loader_args = {}
        if args.use_cuda:
            loader_args = {'num_workers': 1, 'pin_memory': True}

        loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=False, **loader_args)
        loader_val = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, **loader_args)

        # get distribution of classes for class balance loss
        args.class_count, args.expected_loss = count_classes_encode(data_all.labels, True)

        # initialize CON model
        model = CON(in_channels=len(args.alphabet), out_channels_list=args.out_channels, filter_sizes=args.kernel_sizes,
                    strides=args.strides, paddings=args.paddings, num_classes=args.num_classes,
                    kernel_args=[args.sigma, args.scale, args.alpha], scaler=args.preprocessor, pool_global=None)

        # initialize optimizer and loss function
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

        # train model
        if args.use_cuda:
            model.cuda()
        acc, loss = model.sup_train(loader_train, criterion, optimizer, lr_scheduler, val_loader=loader_val,
                                    epochs=args.nb_epochs, early_stop=False, use_cuda=args.use_cuda,
                                    kmeans_init='kmeans++', distance='euclidean')

        # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
        # model
        torch.save({'args': args, 'state_dict': model.state_dict(), 'acc': acc, 'loss': loss},
                   args.outdir + "/CON_results_epochs" + str(args.nb_epochs) + "_" + tfid + ".pkl")

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
            anchor = (torch.acos(model.oligo.pos_anchors[:, 0]) / np.pi) * (len(data_all.data[0]) - 1)
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
    elif args.type == 'ENCODE':
        train_encode(args)
    else:
        raise ValueError('Unknown experiment! Received the following argument: {}'.format(args.type))


if __name__ == '__main__':
    main()
