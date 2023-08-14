#############################################
# This file contains scripts to perform the #
# Random Forest experiments.                #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import argparse
import os
import pickle
from timeit import default_timer as timer

import numpy as np
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier

from cmkn import compute_metrics

# meta variables
PROTEIN = "ARNDCQEGHILKMFPSTWYVXBZJUO~"
DNA = "ACGT"
DNA_FULL = "ACGTN"


# function to parse line arguments
def load_args():
    parser = argparse.ArgumentParser(description="Random forest experiment")
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
        "--n-estimators",
        dest="n_estimators",
        default=[250, 500, 1000],
        type=list,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--max-features",
        dest="max_features",
        default=["sqrt", "log2"],
        type=list,
        help="Number of features to consider when looking for the best split.",
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

    # parse the arguments
    args = parser.parse_args()

    # set the random seeds
    np.random.seed(args.seed)

    # if an output directory is specified, create the dir structure to store the output of the current run
    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True

        if args.type == "HIV":
            args.outdir = args.outdir + "/RF_experiment/{}/{}_{}/".format(
                args.type, args.hiv_type, args.hiv_name,
            )
        else:
            args.outdir = args.outdir + "/RF_experiment/{}/".format(args.type)

        if not os.path.exists(args.outdir):
            try:
                os.makedirs(args.outdir)
            except:
                pass

    return args


# one-hot encoding of strings
def encoding(in_str, alphabet, type="ordinal"):
    if type == "ordinal":
        vector = [alphabet.index(letter) for letter in in_str]
    elif type == "one-hot":
        vector = [
            [0 if char != letter else 1 for char in alphabet] for letter in in_str
        ]
    else:
        raise ValueError("Unknown encoding type: {}".format(type))
    return np.array(vector)


def load_data_hiv(type, drug_number):
    """Auxiliary function to load HIV resistance data."""
    # define path to data file
    filepath = "../data/hivdb/{}_DataSet.fasta".format(type)

    # define dictionary that maps resistance class to label
    class_to_label = {"L": 0, "M": 1, "H": 1}

    # load the fasta file containing the data
    tmp = list(SeqIO.parse(filepath, "fasta"))

    # get the sequences and labels
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
    seq = [encoding(str(i.seq), alphabet, encode_type) for i in tmp]
    label = [int(i.id.split("_")[-1]) for i in tmp]

    return np.array(seq), np.array(label)


# experimental routine
def experiment(args):
    """Function to perform the random forest experiment.

    - **Parameters**::
        :param args: Object holding all arguments
    """

    # load the input data using the correct routine
    if args.type == "HIV":
        # load data
        X, Y = load_data_hiv(args.hiv_type, args.hiv_number)

        # load stratified folds previously created
        #   -> if your are missing the file 'hivdb_stratifiedFolds.pkl' (for HIV experiments), please run the file
        #      'hivdb_preparation.py' (for HIV experiments) first. This python script can be found in the data folder.
        with open("../data/hivdb/hivdb_stratifiedFolds.pkl", "rb") as in_file:
            folds = pickle.load(in_file)
            folds = folds[args.hiv_type][args.hiv_name]
    elif args.type == "synthetic":
        # load data
        X, Y = load_data_synthetic(args.synthetic_data)

        # load stratified folds previously created
        #   -> if your are missing the file '{dataset-name}}_folds.pkl' (for experiments with synthetic data), please run the file
        #      'create_data.py' with --type set to folds. This python script can be found in the data folder.
        folds_file = args.synthetic_data.split(".")[0] + "_folds.pkl"
        with open(folds_file, "rb") as in_file:
            folds = pickle.load(in_file)
    else:
        ValueError("Unknown type of experiment")

    # do the grid search
    for n_esti in args.n_estimators:
        for m_feat in args.max_features:

            # perform 5-fold stratified cross-validation
            results = []
            for fold in folds:
                # initialize the SVM classifier using the oligo kernel
                clf = RandomForestClassifier(n_estimators=n_esti, max_features=m_feat)

                # fit classifier using the loaded data
                tic = timer()
                clf.fit(X[fold[0]], Y[fold[0]])
                toc = timer()
                print(
                    "Finished RF fit ({} samples), elapsed time: {:.2f}min".format(
                        len(X[fold[0]]), (toc - tic) / 60
                    )
                )

                # perform prediction on validation data
                tic = timer()
                pred_y = clf.predict_proba(X[fold[1]])
                toc = timer()
                print(
                    "Finished prediction ({} sample), elapsed time: {:.2f}min".format(
                        len(X[fold[1]]), (toc - tic) / 60
                    )
                )

                # convert real targets into 2-dimensional array
                real_y = np.zeros((len(Y[fold[1]]), 2))
                for i, label in enumerate(Y[fold[1]]):
                    real_y[i, label] = 1

                # compute validation performance
                res = compute_metrics(real_y, pred_y)
                results.append(res.to_dict())

            # store the evaluation results of the current experiment
            filename = "/validation_results_{}_{}.pkl".format(n_esti, m_feat)
            with open(args.outdir + filename, "wb") as out_file:
                pickle.dump(results, out_file, pickle.HIGHEST_PROTOCOL)


def evaluation(args):
    # evaluation needs pandas
    import pandas as pd

    for n_esti in args.n_estimators:
        for m_feat in args.max_features:
            # check if file exists
            try:
                filename = "/validation_results_{}_{}.pkl".format(n_esti, m_feat)
                with open(args.outdir + filename, "rb") as in_file:
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
                print(
                    "Results for RF with {} trees and {} max features\n".format(
                        n_esti, m_feat
                    )
                )
                print(df_res)


def main():
    # load the command line arguments
    args = load_args()

    # check whether the evaluation script should be executed
    if args.eval:
        evaluation(args)

    # otherwise, perform the experiment with the given arguments
    else:
        experiment(args)


if __name__ == "__main__":
    main()
