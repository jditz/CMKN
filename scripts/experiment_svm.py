#############################################
# This file contains scripts to perform the #
# Oligo Kernel SVM experiments.             #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import argparse
import itertools
import os
import pickle
import sys
from timeit import default_timer as timer

import numpy as np
from Bio import SeqIO
from sklearn import svm

from cmkn import compute_metrics

try:
    sys.path.append(os.path.expanduser("~") + "/Development/pyBeast/")
    from OligoEncoding.oligo_encoding import oligoEncoding
    from OligoKernel.oligo_kernel import oligoKernel
except ModuleNotFoundError:
    print(
        "WARNING: The non-public oligo kernel implementation used in this experiment was not found. Please make sure "
        "to provide an implementation of the oligo kernel that can be used with scikit-learn."
    )
except Exception as e:
    raise e


# meta variables
PROTEIN = "ARNDCQEGHILKMFPSTWYVXBZJUO~"
DNA = "ACGT"
DNA_FULL = "ACGTN"


# function to parse line arguments
def load_args():
    parser = argparse.ArgumentParser(description="SVM with oligo kernel experiment")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--outdir",
        metavar="outdir",
        dest="outdir",
        default="output",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Use this flag to enter evaluation mode",
    )
    parser.add_argument(
        "--type",
        dest="type",
        default="HIV",
        type=str,
        choices=["HIV", "synthetic"],
        help="specify the type of experiment.",
    )
    parser.add_argument(
        "--hiv-type",
        dest="hiv_type",
        default="PI",
        type=str,
        choices=["PI", "NRTI", "NNRTI"],
        help="Type of the drug used in the experiment (either PI, NRTI, or NNRTI). Used ONLY if "
        + "--type is set to HIV.",
    )
    parser.add_argument(
        "--hiv-name",
        dest="hiv_name",
        default="SQV",
        type=str,
        help="Name of the drug used in the experiment. Used ONLY if --type is set to HIV.",
    )
    parser.add_argument(
        "--hiv-number",
        dest="hiv_number",
        default=6,
        type=int,
        help="Number of the drug used in the experiment. Used ONLY if --type is set to HIV.",
    )
    parser.add_argument(
        "--dataset",
        dest="synthetic_data",
        default="default.fasta",
        type=str,
        help="Path to the synthetic dataset. Only used if --type is set to 'synthetic'.",
    )
    parser.add_argument(
        "--kernel",
        dest="kernel",
        default="poly",
        type=str,
        choices=["poly", "oligo"],
        help="Determines which kernel should be used for the SVM.",
    )
    parser.add_argument(
        "--reg-params",
        dest="reg_params",
        default=[10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10],
        nargs="+",
        type=float,
        help="List of values used for the SVM's regularizaion parameter (often denoted as C)"
        " during model optimization",
    )
    parser.add_argument(
        "--deg-params",
        dest="deg_params",
        default=[1, 2, 3, 4, 5],
        nargs="+",
        type=int,
        help="List of values for the SVM's degree parameter (determining the degree of the used "
        "polynomial kernel) during model optimization",
    )
    parser.add_argument(
        "--k-params",
        dest="k_params",
        default=[1],
        nargs="+",
        type=int,
        help="List of values for the SVM's k-mer length parameter (used only when the oligo kernel is "
        "selected) during model optimization.",
    )
    parser.add_argument(
        "--sigma-params",
        dest="sigma_params",
        default=[1, 2, 4, 8, 16],
        nargs="+",
        type=int,
        help="List of values for the SVM's positional uncertainty parameter (used only when the oligo "
        "kernel is selected) during model optimization.",
    )

    # parse the arguments
    args = parser.parse_args()

    # set the random seeds
    np.random.seed(args.seed)

    # if oligo kernel is selected the user have to provide an implementation of the oligo kernel that can be used with
    # scikit-learn's implementation of an SVM (i.e. that produces a precomputed kernel matrix)
    if args.kernel == "oligo" and not os.path.exists(
        os.path.expanduser("~") + "/Development/pyBeast/"
    ):
        raise NotImplementedError(
            "Please provide an implementation of the oligo kernel and replace this Error with"
            " the corresponding code to access your implementation. All later code fragments "
            "regarding the use of the oligo kernel have to be appropriately changed as well."
        )

    # if an output directory is specified, create the dir structure to store the output of the current run
    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True

        if args.type == "HIV":
            args.outdir = args.outdir + "/SVM_experiment/{}/{}/{}_{}/".format(
                args.kernel, args.type, args.hiv_type, args.hiv_name
            )
        else:
            args.outdir = args.outdir + "/SVM_experiment/{}/{}/".format(
                args.kernel, args.type
            )

        if not os.path.exists(args.outdir):
            try:
                os.makedirs(args.outdir)
            except:
                pass

    return args


# one-hot encoding of strings
def encoding(in_str, alphabet, type="ordinal"):
    if type == "ordinal":
        try:
            vector = [alphabet.index(letter) for letter in in_str]
        except ValueError:
            raise ValueError("unknown letter in input: {}".format(in_str))
    elif type == "one-hot":
        vector = [
            [0 if char != letter else 1 for char in alphabet] for letter in in_str
        ]
    else:
        raise ValueError("Unknown encoding type: {}".format(type))
    return np.array(vector)


def load_data_hiv(type, drug_number, encode_type="ordinal"):
    """Auxiliary function to load HIV resistance data.
    """

    # define path to data file
    filepath = "../data/hivdb/{}_DataSet.fasta".format(type)

    # define dictionary that maps resistance class to label
    class_to_label = {"L": 0, "M": 1, "H": 1}

    # load the fasta file containing the data
    tmp = list(SeqIO.parse(filepath, "fasta"))

    # get the sequences and labels
    if encode_type == "oligo":
        seq = [
            str(i.seq).replace("~", "X")
            for i in tmp
            if i.id.split("|")[drug_number] != "NA"
        ]
    else:
        seq = [
            encoding(str(i.seq), PROTEIN)
            for i in tmp
            if i.id.split("|")[drug_number] != "NA"
        ]
    label = [
        class_to_label[i.id.split("|")[drug_number]]
        for i in tmp
        if i.id.split("|")[drug_number] != "NA"
    ]

    return np.array(seq), np.array(label)


def load_data_synthetic(filepath, alphabet=DNA, encode_type="ordinal"):
    """Auxiliary function to load synthetic data.
    """

    # load the fasta file containing the data
    tmp = list(SeqIO.parse(filepath, "fasta"))

    # get the sequences and labels
    if encode_type == "oligo":
        seq = [str(i.seq) for i in tmp]
    else:
        seq = [encoding(str(i.seq), alphabet, encode_type) for i in tmp]
    label = [int(i.id.split("_")[-1]) for i in tmp]

    return np.array(seq), np.array(label)


def experiment(args):
    """Function to perform the kernel SVM experiment.

    - **Parameters**::
        :param args: Parsed command line arguments
    """

    # load the input data using the correct routine
    if args.type == "HIV":
        # load data
        X, Y = load_data_hiv(args.hiv_type, args.hiv_number, args.kernel)

        # specify the alphabet
        alphabet = PROTEIN[:-1]

        # load stratified folds previously created
        #   -> if your are missing the file 'hivdb_stratifiedFolds.pkl' (for HIV experiments), please run the file
        #      'hivdb_preparation.py' (for HIV experiments) first. This python script can be found in the data folder.
        with open("../data/hivdb/hivdb_stratifiedFolds.pkl", "rb") as in_file:
            folds = pickle.load(in_file)
            folds = folds[args.hiv_type][args.hiv_name]
    elif args.type == "synthetic":
        if args.kernel == "oligo":
            aux_encode = "oligo"
        else:
            aux_encode = "ordinal"

        # load data
        X, Y = load_data_synthetic(args.synthetic_data, encode_type=aux_encode)

        # specify the alphabet
        alphabet = DNA

        # load stratified folds previously created
        #   -> if your are missing the file '{dataset-name}}_folds.pkl' (for experiments with synthetic data), please run the file
        #      'create_data.py' with --type set to folds. This python script can be found in the data folder.
        # folds_file = args.synthetic_data.split(".")[0] + "_folds.pkl"
        folds_file = "data/synthetic/workaround.pkl"
        with open(folds_file, "rb") as in_file:
            folds = pickle.load(in_file)
    else:
        ValueError("Unknown type of experiment: {}".format(args.type))

    # make sure to iterate through the correct params for each kernel type
    if args.kernel == "poly":
        aux_print = "Training SVM with polynomial degree = {} and C = {}..."
        aux_params = args.deg_params
    elif args.kernel == "oligo":
        aux_print = "Training SVM with (k, sigma) = {} and C = {}..."
        if len(args.sigma_params) > len(args.k_params):
            aux_params = [
                list(zip(args.k_params, each_permutation))
                for each_permutation in itertools.permutations(
                    args.sigma_params, len(args.k_params)
                )
            ]
        else:
            aux_params = [
                list(zip(each_permutation, args.sigma_params))
                for each_permutation in itertools.permutations(
                    args.k_params, len(args.sigma_params)
                )
            ]
    else:
        raise ValueError("Unknown kernel type: {}".format(args.kernel))

    # iterate over the specified model parameters to perform a grid search
    for reg in args.reg_params:
        for param in aux_params:
            print(aux_print.format(param, reg))

            # perform 5-fold stratified cross-validation
            results = []
            for fold_nb, fold in enumerate(folds):

                # initialize the SVM classifier using the specified kernel
                if args.kernel == "poly":
                    clf = svm.SVC(
                        C=reg, kernel=args.kernel, degree=param, probability=True
                    )
                    in_train = X[fold[0]]
                    in_test = X[fold[1]]
                elif args.kernel == "oligo":
                    # CAUTION: This part of the script uses a non-public implementation of the oligo kernel. Please
                    # make sure that you have a working implementation of the oligo kernel an change the code,
                    # appropriately.
                    clf = svm.SVC(C=reg, kernel="precomputed", probability=True)

                    # create the oligo encoding for all data
                    sequences = np.concatenate((X[fold[0]], X[fold[1]]))
                    positions, values = oligoEncoding(
                        k_mer_length=param[0][0], alphabet=alphabet
                    ).getEncoding(list(sequences))
                    kernel = oligoKernel(
                        sigma=param[0][1], max_distance=True
                    ).symmetric(positions, values)
                    in_train = kernel[: len(fold[0]), : len(fold[0])]
                    in_test = kernel[len(fold[0]) :, : len(fold[0])]
                else:
                    raise ValueError("Unknown kernel type: {}".format(args.kernel))

                tic = timer()
                # fit classifier using the loaded data
                clf.fit(in_train, Y[fold[0]])
                toc = timer()
                print(
                    "    Finished SVM fit ({} samples), elapsed time: {:.2f}min".format(
                        len(X[fold[0]]), (toc - tic) / 60
                    )
                )

                # perform prediction on validation data
                tic = timer()
                pred_y = clf.predict_proba(in_test)
                toc = timer()
                print(
                    "    Finished prediction ({} samples), elapsed time: {:.2f}min\n".format(
                        len(X[fold[1]]), (toc - tic) / 60
                    )
                )

                # convert real targets into 2-dimensional array
                real_y = np.zeros((len(Y[fold[1]]), 2))
                for i, label in enumerate(Y[fold[1]]):
                    real_y[i, label] = 1

                res = compute_metrics(real_y, pred_y)
                results.append(res.to_dict())

            # store the evaluation results of the current experiment
            filename = "validation_results_{}_{}.pkl".format(param, reg)
            with open(args.outdir + filename, "wb") as out_file:
                pickle.dump(results, out_file, pickle.HIGHEST_PROTOCOL)


def evaluation(args):
    # evaluation needs pandas
    import pandas as pd

    # make sure to iterate through the correct params for each kernel type
    if args.kernel == "poly":
        aux_print = "Result SVM with polynomial degree = {} and C = {}\n"
        aux_params = args.deg_params
    elif args.kernel == "oligo":
        aux_print = "Result SVM with (k, sigma) = {} and C = {}\n"
        if len(args.sigma_params) > len(args.k_params):
            aux_params = [
                list(zip(args.k_params, each_permutation))
                for each_permutation in itertools.permutations(
                    args.sigma_params, len(args.k_params)
                )
            ]
        else:
            aux_params = [
                list(zip(each_permutation, args.sigma_params))
                for each_permutation in itertools.permutations(
                    args.k_params, len(args.sigma_params)
                )
            ]
    else:
        raise ValueError("Unknown kernel type: {}".format(args.kernel))

    # iterate over the specified model parameters for which the validation results will be displayed
    for reg in args.reg_params:
        for param in aux_params:
            try:
                with open(
                    args.outdir + "validation_results_{}_{}.pkl".format(param, reg),
                    "rb",
                ) as in_file:
                    val_results = pickle.load(in_file)
            except FileNotFoundError:
                raise ValueError("Specified result file does not exist")
            except Exception as e:
                raise ("Unknown error: {}".format(e))

            # combine results of all folds into one data frame
            results = []
            for val_res in val_results:
                results.append(pd.DataFrame.from_dict(val_res))
            df_res = pd.concat(results, axis=1)

            # calculate mean and std over all folds
            df_res["mean"] = df_res.mean(axis=1)
            df_res["std"] = df_res[:-1].std(axis=1)

            # print the results
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                None,
            ):
                print(aux_print.format(param, reg))
                print(df_res)
                print(
                    "\n--------------------------------------------------------------------------------\n\n"
                )


def main():
    # load command line arguments
    args = load_args()

    # check if the evaluation mode was specified
    if args.eval:
        evaluation(args)

    # otherwise, run the experiment with the given arguments
    else:
        experiment(args)


if __name__ == "__main__":
    main()
