#############################################
# This file contains scripts to perform the #
# Convolutional Oligo Kernel Network        #
# experiments.                              #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import argparse
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

try:
    from imblearn.under_sampling import RandomUnderSampler
except ModuleNotFoundError:
    print("No module named 'imblearn'. Skipping import.\n\n")

from cmkn import CMKN, ClassBalanceLoss, CMKNDataset, compute_metrics

# MACROS

NAME = "CMKN_experiment"
DATA_DIR = "../data/"


# extend custom data handler for the used dataset
class CustomHandler(CMKNDataset):
    def __init__(
        self,
        filepath,
        kmer_size=3,
        drug="SQV",
        nb_classes=2,
        clean_set=True,
        encode="onehot",
        alphabet="PROTEIN_FULL",
        experiment="hiv",
    ):
        self.drug_nb = {
            "FPV": 1,
            "ATV": 2,
            "IDV": 3,
            "LPV": 4,
            "NFV": 5,
            "SQV": 6,
            "TPV": 7,
            "DRV": 8,
            "3TC": 1,
            "ABC": 2,
            "AZT": 3,
            "D4T": 4,
            "DDI": 5,
            "TDF": 6,
            "EFV": 1,
            "NVP": 2,
            "ETR": 3,
            "RPV": 4,
        }
        self.drug = drug
        self.exp = experiment

        if clean_set and experiment == "hiv":
            aux_tup = (self.drug_nb[drug], "NA")
        else:
            aux_tup = None

        super(CustomHandler, self).__init__(
            filepath,
            kmer_size=kmer_size,
            alphabet=alphabet,
            clean_set=aux_tup,
            seq_encode=encode,
        )

        self.nb_classes = nb_classes
        if nb_classes == 2:
            if experiment == "hiv":
                self.class_to_idx = {"L": 0, "M": 1, "H": 1}
                self.all_categories = ["susceptible", "resistant"]
            elif experiment in ["splice", "synthetic"]:
                self.class_to_idx = {"0": 0, "1": 1}
            else:
                raise ValueError("Unknown experiment type: {}".format(experiment))
        else:
            if experiment == "hiv":
                self.class_to_idx = {"L": 0, "M": 1, "H": 2}
                self.all_categories = ["L", "M", "H"]
            elif experiment in ["splice", "synthetic"]:
                raise ValueError(
                    "Unsupported number of classes for synthetic / splice site prediction "
                    "({} were selected but only 2 is supported)".format(nb_classes)
                )
            else:
                raise ValueError("Unknown experiment type: {}".format(experiment))

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
                # process the id string depending on the selected experiment
                if self.exp == "hiv":
                    aux_lab = id_str.split("|")[self.drug_nb[self.drug]]
                elif self.exp in ["splice", "synthetic"]:
                    aux_lab = id_str.split("_")[-1]
                else:
                    raise ValueError("Unknown experiment type: {}".format(self.exp))

                # set the corresponding field in the label tensor to 1.0
                if aux_lab != "NA":
                    if len(sample[1]) == 1:
                        labels[self.class_to_idx[aux_lab]] = 1.0
                    else:
                        labels[i, self.class_to_idx[aux_lab]] = 1.0

            except:
                continue

        # return the sample with updated label tensor
        sample = (sample[0], labels)
        return sample


def load_args():
    """
    Function to create an argument parser
    """
    parser = argparse.ArgumentParser(description="CMKN example experiment")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--type",
        dest="type",
        default="HIV",
        type=str,
        choices=["HIV", "SPLICE_NN269", "SPLICE_DGSPLICER", "synthetic"],
        help="specify the type of experiment, i.e. the used dataset. Currently, only 'HIV' is "
        "supported.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        metavar="M",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        dest="nb_epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--out-channels",
        dest="out_channels",
        metavar="m",
        default=[99],
        nargs="+",
        type=int,
        help="number of out channels for each oligo kernel and conv layer (default [40])",
    )
    parser.add_argument(
        "--strides",
        dest="strides",
        metavar="s",
        default=[1],
        nargs="+",
        type=int,
        help="stride value for each layer (default: [1])",
    )
    parser.add_argument(
        "--paddings",
        dest="paddings",
        metavar="p",
        default=["SAME"],
        nargs="+",
        type=str,
        help="padding type for each convolutional layer (default ['SAME'])",
    )
    parser.add_argument(
        "--kernel-sizes",
        dest="kernel_sizes",
        metavar="k",
        default=[1],
        nargs="+",
        type=int,
        help="kernel sizes for oligo and convolutional layers (default [1])",
    )
    parser.add_argument(
        "--sigma",
        dest="sigma",
        default=1,
        type=float,
        help="sigma for the oligo kernel layer (default: 4)",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        default=1,
        type=int,
        help="alpha parameter used in the exponential function of the motif kernel layer; if set to -1, the "
        "value will be depending on the number of oligomers (default: -1)",
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        default=-1,
        type=int,
        help="scaling parameter for the motif kernel layer (default: -1). If set to -1, the scaling "
        "parameter will be determined depending on the length of the input sequences.",
    )
    parser.add_argument(
        "--num-classes",
        dest="num_classes",
        default=2,
        type=int,
        help="number of classes in the prediction task",
    )
    parser.add_argument(
        "--kfold",
        dest="kfold",
        default=5,
        type=int,
        help="k-fold cross validation (default: 5)",
    )
    parser.add_argument(
        "--penalty",
        metavar="penal",
        dest="penalty",
        default="l2",
        type=str,
        choices=["l2", "l1"],
        help="regularization used in the last layer (default: l2)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="regularization parameter for sup CON",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="standard_row",
        help="preprocessor for last layer of CON (default: standard_row). Set to 'standard_row' or "
        "'standard_column' in order to use a LinearMixin layer. Or set it to the length of input"
        "sequences to use a standard fully-connected layer.",
    )
    parser.add_argument(
        "--outdir",
        metavar="outdir",
        dest="outdir",
        default="output",
        type=str,
        help="output path(default: '')",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", default=True, help="use gpu (default: False)"
    )
    parser.add_argument("--noise", type=float, default=0.0, help="perturbation percent")
    parser.add_argument(
        "--file",
        dest="filepath",
        default="./data/hivdb",
        type=str,
        help="path to the file containing the dataset.",
    )
    parser.add_argument(
        "--extension",
        dest="extension",
        default="fasta",
        type=str,
        help="extension of the file containing the dataset (default fasta)",
    )
    parser.add_argument(
        "--drug",
        dest="drug",
        default="SQV",
        type=str,
        help="specifies the drug that will be used to classify virus resilience (default SQV)",
    )
    parser.add_argument(
        "--encodeset",
        dest="encodeset",
        default="CTCF_A549_CTCF_UT-A",
        type=str,
        help="specifies which ENCODE dataset will be used for training a model; if set to 'optim', "
        + "the hyperparameters will be optimized using 100 randomly selected ENCODE datasets",
    )
    parser.add_argument(
        "--splice-type",
        dest="splice_type",
        default="acceptor",
        type=str,
        choices=["acceptor", "donor"],
        help="type of the used splice site (this option is only used if the experiment is set to "
        + "'SPLICE')",
    )
    parser.add_argument(
        "--loss-beta",
        dest="loss_beta",
        default=0.99,
        type=float,
        help="beta parameter of the ClassBalanceLoss class.",
    )
    parser.add_argument(
        "--loss-gamma",
        dest="loss_gamma",
        default=1.0,
        type=float,
        help="gamma parameter of the ClassBalanceLoss class.",
    )

    # parse the arguments
    args = parser.parse_args()

    # GPU will only be used if available
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    # store length of oligomers for a simple rpoxy access
    args.kmer_size = args.kernel_sizes[0]

    # 'num-anchors', 'subsampling', and 'sigma' have to be of same length
    if (
        not len(args.out_channels)
        == len(args.strides)
        == len(args.paddings)
        == len(args.kernel_sizes)
    ):
        raise ValueError(
            "The size combination of out_channels, strides, paddings, and kernel_sizes is invalid!\n"
            + "    out_channels, strides, paddings, and kernel_sizes must have the same size"
        )

    # number of layers is equal to length of the 'num-anchors' vector
    args.n_layer = len(args.out_channels)

    # make sure that args.prerocessor is set up, properly
    try:
        args.preprocessor = int(args.preprocessor)
    except ValueError:
        if args.preprocessor not in ["standard_row", "standard_column"]:
            raise ValueError(
                "preprocessor must be either an Integer or one of the following strings: standard_row, "
                "standard_column"
            )

    # for a HIV experiment, also store the type of antiviral drug
    if args.type == "HIV":
        if args.drug in ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"]:
            args.drug_type = "PI"
        elif args.drug in ["3TC", "ABC", "AZT", "D4T", "DDI", "TDF"]:
            args.drug_type = "NRTI"
        else:
            args.drug_type = "NNRTI"

        # add the correct file to the given filepath
        args.filepath = args.filepath.rstrip(os.path.sep) + "{}{}_DataSet.fasta".format(
            os.path.sep, args.drug_type
        )

    # for a SPLICE experiment, the filepath has to leave an insertable space for train and test data
    if args.type == "SPLICE_NN269":
        args.filepath = (
            args.filepath.rstrip(os.path.sep)
            + os.path.sep
            + "NN269_{}_".format(args.splice_type)
            + "{}.fasta"
        )
    elif args.type == "SPLICE_DGSPLICER":
        args.filepath = (
            args.filepath.rstrip(os.path.sep)
            + os.path.sep
            + "DGSplicer_{}_".format(args.splice_type)
            + "{}.fasta"
        )

    # for an ENCODE experiment, make sure that the path to the datafiles does not have a trailing path separator
    elif args.type == "ENCODE":
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
        if args.type == "HIV":
            aux_dir = args.drug
        elif args.type in ["SPLICE_NN269", "SPLICE_DGSPLICER"]:
            aux_dir = "{}_{}".format(args.type, args.splice_type)
        elif args.type == "ENCODE":
            aux_dir = args.encodeset
        else:
            aux_dir = "synthetic"
        aux_out = "/{}/{}/classes_{}/kmer_{}/params_{}_{}_{}/anchors_{}/layers_{}".format(
            NAME,
            aux_dir,
            args.num_classes,
            args.kmer_size,
            args.sigma,
            args.scale,
            args.alpha,
            args.out_channels[0],
            args.n_layer,
        )
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

    with open(filepath, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if drug == -1:
                aux_lab = int(record.id.split("_")[1])
                class_count[aux_lab] += 1
                label_vec.append(aux_lab)
            else:
                aux_lab = record.id.split("|")[drug]
                if num_classes == 2:
                    if aux_lab == "H":
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == "M":
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == "L":
                        class_count[0] += 1
                        label_vec.append(0)
                    else:
                        continue
                else:
                    if aux_lab == "H":
                        class_count[2] += 1
                        label_vec.append(2)
                    elif aux_lab == "M":
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == "L":
                        class_count[0] += 1
                        label_vec.append(0)
                    else:
                        continue
            nb_samples += 1

    if verbose:
        print("number of samples in dataset: {}".format(nb_samples))
        if num_classes == 2:
            print(
                "class balance: resistant = {}, susceptible = {}".format(
                    class_count[1] / nb_samples, class_count[0] / nb_samples
                )
            )
        else:
            print(
                "class balance: H = {}, M = {}, L = {}".format(
                    class_count[2] / nb_samples,
                    class_count[1] / nb_samples,
                    class_count[0] / nb_samples,
                )
            )

    if num_classes == 2:
        expected_loss = -(class_count[0] / nb_samples) * np.log(0.5) - (
            class_count[1] / nb_samples
        ) * np.log(0.5)
    else:
        expected_loss = (
            -(class_count[0] / nb_samples) * np.log(0.33)
            - (class_count[1] / nb_samples) * np.log(0.33)
            - (class_count[2] / nb_samples) * np.log(0.33)
        )

    if verbose:
        print("expected loss: {}".format(expected_loss))

    if num_classes == 2:
        rand_guess = (class_count[0] / nb_samples) ** 2 + (
            class_count[1] / nb_samples
        ) ** 2
    else:
        rand_guess = (
            (class_count[0] / nb_samples) ** 2
            + (class_count[1] / nb_samples) ** 2
            + (class_count[2] / nb_samples) ** 2
        )

    if verbose:
        print("expected accuracy with random guessing: {}".format(rand_guess))

    return class_count, expected_loss, np.array(label_vec)


def count_classes_splice(filepath, mode, verbose=True):
    class_count = [0, 0]
    label_vec = []
    nb_samples = 0

    with open(filepath.format(mode), "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            aux_lab = int(record.id.split("_")[-1])
            class_count[aux_lab] += 1
            label_vec.append(aux_lab)
            nb_samples += 1

    if verbose:
        print("number of samples in dataset: {}".format(nb_samples))
        print(
            "class balance: pos = {}, neg = {}".format(
                class_count[1] / nb_samples, class_count[0] / nb_samples
            )
        )

    expected_loss = -(class_count[0] / nb_samples) * np.log(0.5) - (
        class_count[1] / nb_samples
    ) * np.log(0.5)

    if verbose:
        print("expected loss: {}".format(expected_loss))

    rand_guess = (class_count[0] / nb_samples) ** 2 + (class_count[1] / nb_samples) ** 2

    if verbose:
        print("expected accuracy with random guessing: {}".format(rand_guess))

    return class_count, expected_loss, np.array(label_vec)


def train_hiv(args):
    args.alphabet = "ARNDCQEGHILKMFPSTWYVXBZJUO"

    # load data
    data_all = CustomHandler(
        args.filepath, kmer_size=args.kmer_size, drug=args.drug, encode="onehot"
    )

    # determine if oligo kernel's scale parameter should be set depending on sequence length
    if args.scale == -1:
        args.scale = len(data_all.data[0]) * (len(data_all.data[0]) / 10)

    # set random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load indices for 5-fold cross-validation
    fold_filepath = args.filepath.split("/")[:-1]
    fold_filepath = "/".join(fold_filepath) + "/hivdb_stratifiedFolds.pkl"
    with open(fold_filepath, "rb") as in_file:
        folds = pickle.load(in_file)
        folds = folds[args.drug_type][args.drug]

    # perform 5-fold stratified cross-validation using the predefined folds
    for fold_nb, fold in enumerate(folds):
        # store training and validation indices for the current fold
        args.train_indices, args.val_indices = fold[0], fold[1]

        # initialize the CMKN model
        model = CMKN(
            in_channels=len(args.alphabet),
            out_channels_list=args.out_channels,
            filter_sizes=args.kernel_sizes,
            strides=args.strides,
            paddings=args.paddings,
            num_classes=args.num_classes,
            kernel_args=[args.sigma, args.scale, args.alpha],
            scaler=args.preprocessor,
            pool_global=None,
        )

        # get distribution of classes for class balance loss
        args.class_count, args.expected_loss, _ = count_classes_hiv(
            args.filepath, True, data_all.drug_nb[args.drug], args.num_classes
        )

        # set arguments for the DataLoader
        loader_args = {}
        if args.use_cuda:
            loader_args = {"num_workers": 1, "pin_memory": True}

        # Creating PyTorch data Subsets using the indices for the current fold
        data_train = Subset(data_all, args.train_indices)
        data_val = Subset(data_all, args.val_indices)

        # create PyTorch DataLoader for training and validation data
        loader_train = DataLoader(
            data_train, batch_size=args.batch_size, shuffle=False, **loader_args
        )
        loader_val = DataLoader(
            data_val, batch_size=args.batch_size, shuffle=False, **loader_args
        )

        # initialize optimizer and loss function
        if args.num_classes == 2:
            criterion = ClassBalanceLoss(
                args.class_count,
                args.num_classes,
                "sigmoid",
                args.loss_beta,
                args.loss_gamma,
            )
        else:
            criterion = ClassBalanceLoss(
                args.class_count,
                args.num_classes,
                "cross_entropy",
                args.loss_beta,
                args.loss_gamma,
            )
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

        # train model
        if args.use_cuda:
            model.cuda()
        acc, loss = model.sup_train(
            loader_train,
            criterion,
            optimizer,
            lr_scheduler,
            epochs=args.nb_epochs,
            early_stop=False,
            use_cuda=args.use_cuda,
            kmeans_init="k-means++",
            distance="euclidean",
        )

        # compute performance metrices on validation data
        pred_y, true_y = model.predict(loader_val, proba=True, use_cuda=args.use_cuda)
        scores = compute_metrics(true_y, pred_y)

        # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
        # model
        torch.save(
            {
                "args": args,
                "state_dict": model.state_dict(),
                "acc": acc,
                "loss": loss,
                "val_performance": scores.to_dict(),
            },
            args.outdir + "/CMKN_results_fold" + str(fold_nb) + ".pkl",
        )


def train_splice(args):
    args.alphabet = "ACGTN"

    # load data
    data_all = CustomHandler(
        args.filepath.format("train"),
        kmer_size=args.kmer_size,
        clean_set=False,
        encode="onehot",
        alphabet="DNA_FULL",
        experiment="splice",
    )

    # get labels of each entry for stratified shuffling and distribution of classes for class balance loss
    args.class_count, args.expected_loss, label_vec = count_classes_splice(
        args.filepath, mode="train"
    )

    # determine if position-aware motif kernel's scale parameter should be set depending on sequence length
    if args.scale == -1:
        args.scale = len(data_all.data[0]) * (len(data_all.data[0]) / 10)

    # set random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # perform n-fold stratified cross-validation using scikit-learn
    #   -> n can be set with the command line argument '--kfold n'
    indices = np.arange(len(data_all))
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for fold_nb, split_idx in enumerate(skf.split(indices, label_vec)):

        # initialize the CMKN model
        model = CMKN(
            in_channels=len(args.alphabet),
            out_channels_list=args.out_channels,
            filter_sizes=args.kernel_sizes,
            strides=args.strides,
            paddings=args.paddings,
            num_classes=args.num_classes,
            kernel_args=[args.sigma, args.scale, args.alpha],
            scaler=args.preprocessor,
            pool_global=None,
        )

        # under-sample the majority class in the training set for experiments with the DGSplicer dataset
        #     1. create a mock input for the subsampling class
        #     2. initialize the subsampler with a target ration of 0.25 (N_min / N_maj = 0.25)
        #     3. retrieve the indices of the samples included in the new training dataset
        #     4. update training indices and class counts
        if args.type == "SPLICE_DGSPLICER":
            mock_input = split_idx[0].reshape(-1, 1)
            rus = RandomUnderSampler(
                sampling_strategy=0.25, random_state=args.seed, replacement=False
            )
            sub_train, _ = rus.fit_resample(mock_input, label_vec[split_idx[0]])
            args.class_count = [int(len(sub_train) * 0.8), int(len(sub_train) * 0.2)]
            split_idx = (sub_train.reshape(-1), split_idx[1])

        # split dataset into training and validation samples
        args.train_indices, args.val_indices = split_idx[0], split_idx[1]

        # set arguments for the DataLoader
        loader_args = {}
        if args.use_cuda:
            loader_args = {"num_workers": 1, "pin_memory": True}

        # Creating PyTorch data Subsets using the indices for the current fold
        data_train = Subset(data_all, args.train_indices)
        data_val = Subset(data_all, args.val_indices)

        # create PyTorch DataLoader for training and validation data
        loader_train = DataLoader(
            data_train, batch_size=args.batch_size, shuffle=False, **loader_args
        )
        loader_val = DataLoader(
            data_val, batch_size=args.batch_size, shuffle=False, **loader_args
        )

        # initialize optimizer and loss function
        if args.num_classes == 2:
            criterion = ClassBalanceLoss(
                args.class_count,
                args.num_classes,
                "sigmoid",
                args.loss_beta,
                args.loss_gamma,
            )
        else:
            criterion = ClassBalanceLoss(
                args.class_count,
                args.num_classes,
                "cross_entropy",
                args.loss_beta,
                args.loss_gamma,
            )
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

        # train model
        if args.use_cuda:
            model.cuda()
        acc, loss = model.sup_train(
            loader_train,
            criterion,
            optimizer,
            lr_scheduler,
            epochs=args.nb_epochs,
            early_stop=False,
            use_cuda=args.use_cuda,
            kmeans_init="k-means++",
            distance="euclidean",
        )

        # compute performance metrices on validation data
        pred_y, true_y = model.predict(loader_val, proba=True, use_cuda=args.use_cuda)
        scores = compute_metrics(true_y, pred_y)

        # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
        # model
        torch.save(
            {
                "args": args,
                "state_dict": model.state_dict(),
                "acc": acc,
                "loss": loss,
                "val_performance": scores.to_dict(),
            },
            args.outdir + "/CMKN_results_fold" + str(fold_nb) + ".pkl",
        )


def train_synthetic(args):
    """Function to train CMKN models on synthetic data.
    """
    args.alphabet = "ACGT"

    # load data
    data_all = CustomHandler(
        args.filepath,
        kmer_size=args.kmer_size,
        clean_set=False,
        encode="onehot",
        alphabet="DNA",
        experiment="synthetic",
    )

    # get labels of each entry for stratified shuffling and distribution of classes for class balance loss
    args.class_count, args.expected_loss, label_vec = count_classes_splice(
        args.filepath, mode="train"
    )

    # determine if position-aware motif kernel's scale parameter should be set depending on sequence length
    if args.scale == -1:
        args.scale = len(data_all.data[0]) * (len(data_all.data[0]) / 10)

    # set random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.kfold > 1:
        # perform n-fold stratified cross-validation using scikit-learn
        #   -> n can be set with the command line argument '--kfold n'
        indices = np.arange(len(data_all))
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        for fold_nb, split_idx in enumerate(skf.split(indices, label_vec)):

            # initialize the CMKN model
            model = CMKN(
                in_channels=len(args.alphabet),
                out_channels_list=args.out_channels,
                filter_sizes=args.kernel_sizes,
                strides=args.strides,
                paddings=args.paddings,
                num_classes=args.num_classes,
                kernel_args=[args.sigma, args.scale, args.alpha],
                scaler=args.preprocessor,
                pool_global=None,
            )

            # split dataset into training and validation samples
            args.train_indices, args.val_indices = split_idx[0], split_idx[1]

            # set arguments for the DataLoader
            loader_args = {}
            if args.use_cuda:
                loader_args = {"num_workers": 1, "pin_memory": True}

            # Creating PyTorch data Subsets using the indices for the current fold
            data_train = Subset(data_all, args.train_indices)
            data_val = Subset(data_all, args.val_indices)

            # create PyTorch DataLoader for training and validation data
            loader_train = DataLoader(
                data_train, batch_size=args.batch_size, shuffle=False, **loader_args
            )
            loader_val = DataLoader(
                data_val, batch_size=args.batch_size, shuffle=False, **loader_args
            )

            # initialize optimizer and loss function
            if args.num_classes == 2:
                criterion = ClassBalanceLoss(
                    args.class_count,
                    args.num_classes,
                    "sigmoid",
                    args.loss_beta,
                    args.loss_gamma,
                )
            else:
                criterion = ClassBalanceLoss(
                    args.class_count,
                    args.num_classes,
                    "cross_entropy",
                    args.loss_beta,
                    args.loss_gamma,
                )
            optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
            lr_scheduler = ReduceLROnPlateau(
                optimizer, factor=0.5, patience=4, min_lr=1e-4
            )

            # train model
            if args.use_cuda:
                model.cuda()
            acc, loss = model.sup_train(
                loader_train,
                criterion,
                optimizer,
                lr_scheduler,
                epochs=args.nb_epochs,
                early_stop=False,
                use_cuda=args.use_cuda,
                kmeans_init="k-means++",
                distance="euclidean",
            )

            # compute performance metrices on validation data
            pred_y, true_y = model.predict(
                loader_val, proba=True, use_cuda=args.use_cuda
            )
            scores = compute_metrics(true_y, pred_y)

            # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
            # model
            torch.save(
                {
                    "args": args,
                    "state_dict": model.state_dict(),
                    "acc": acc,
                    "loss": loss,
                    "val_performance": scores.to_dict(),
                },
                args.outdir + "/CMKN_results_fold" + str(fold_nb) + ".pkl",
            )

    else:
        # train model on whole dataset for interpretation

        # initialize the CMKN model
        model = CMKN(
            in_channels=len(args.alphabet),
            out_channels_list=args.out_channels,
            filter_sizes=args.kernel_sizes,
            strides=args.strides,
            paddings=args.paddings,
            num_classes=args.num_classes,
            kernel_args=[args.sigma, args.scale, args.alpha],
            scaler=args.preprocessor,
            pool_global=None,
        )

        # set arguments for the DataLoader
        loader_args = {}
        if args.use_cuda:
            loader_args = {"num_workers": 1, "pin_memory": True}

        # create PyTorch DataLoader
        loader = DataLoader(
            data_all, batch_size=args.batch_size, shuffle=False, **loader_args
        )

        # initialize optimizer and loss function
        if args.num_classes == 2:
            criterion = ClassBalanceLoss(
                args.class_count,
                args.num_classes,
                "sigmoid",
                args.loss_beta,
                args.loss_gamma,
            )
        else:
            criterion = ClassBalanceLoss(
                args.class_count,
                args.num_classes,
                "cross_entropy",
                args.loss_beta,
                args.loss_gamma,
            )
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

        # train model
        if args.use_cuda:
            model.cuda()
        acc, loss = model.sup_train(
            loader,
            criterion,
            optimizer,
            lr_scheduler,
            epochs=args.nb_epochs,
            early_stop=False,
            use_cuda=args.use_cuda,
            kmeans_init="k-means++",
            distance="euclidean",
        )

        # compute performance metrices on validation data
        pred_y, true_y = model.predict(loader, proba=True, use_cuda=args.use_cuda)
        scores = compute_metrics(true_y, pred_y)

        # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
        # model
        torch.save(
            {
                "args": args,
                "state_dict": model.state_dict(),
                "acc": acc,
                "loss": loss,
                "val_performance": scores.to_dict(),
            },
            args.outdir + "/CMKN_results.pkl",
        )


def main():
    # read parameter
    args = load_args()

    # perform the training procedure corresponding to the selected experiment
    if args.type == "HIV":
        train_hiv(args)
    elif args.type in ["SPLICE_NN269", "SPLICE_DGSPLICER"]:
        train_splice(args)
    elif args.type == "synthetic":
        train_synthetic(args)
    else:
        raise ValueError(
            "Unknown experiment! Received the following argument: {}".format(args.type)
        )


if __name__ == "__main__":
    main()
