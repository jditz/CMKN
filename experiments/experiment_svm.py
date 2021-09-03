#############################################
# This file contains scripts to perform the #
# Oligo Kernel SVM experiments.             #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import os
import argparse

import numpy as np
from sklearn import svm
from Bio import SeqIO
import pickle
from timeit import default_timer as timer

from cmkn import compute_metrics


# meta variables
PROTEIN = 'ARNDCQEGHILKMFPSTWYVXBZJUO~'
DNA = 'ACGTN'


# function to parse line arguments
def load_args():
    parser = argparse.ArgumentParser(description="SVM with oligo kernel experiment")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--outdir", metavar="outdir", dest="outdir", default='../output', type=str,
                        help="output path")
    parser.add_argument("--eval", action='store_true', default=False, help="Use this flag to enter evaluation mode")
    parser.add_argument('--type', dest='type', default='HIV', type=str, choices=['HIV'],
                        help="specify the type of experiment.")
    parser.add_argument("--hiv-type", dest="hiv_type", default='PI', type=str, choices=['PI', 'NRTI', 'NNRTI'],
                        help="Type of the drug used in the experiment (either PI, NRTI, or NNRTI). Used ONLY if " +
                             "--type is set to HIV.")
    parser.add_argument("--hiv-name", dest="hiv_name", default='SQV', type=str,
                        help="Name of the drug used in the experiment. Used ONLY if --type is set to HIV.")
    parser.add_argument("--hiv-number", dest="hiv_number", default=6, type=int,
                        help="Number of the drug used in the experiment. Used ONLY if --type is set to HIV.")
    parser.add_argument("--reg_params", dest="reg_params", default=[10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10],
                        type=list, help="List of values used for the SVM's regularizaion parameter (often denoted as C)"
                                        " during model optimization")
    parser.add_argument("--deg_params", dest="deg_params", default=[1, 2, 3, 4, 5], type=list,
                        help="List of values for the SVM's degree parameter (determining the degree of the used "
                             "polynomial kernel) during model optimization")

    # parse the arguments
    args = parser.parse_args()

    # set the random seeds
    np.random.seed(args.seed)

    # if an output directory is specified, create the dir structure to store the output of the current run
    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True
        args.outdir = args.outdir + "/SVM_experiment/{}/{}_{}/".format(args.type, args.hiv_type, args.hiv_name)
        if not os.path.exists(args.outdir):
            try:
                os.makedirs(args.outdir)
            except:
                pass

    return args


# one-hot encoding of strings
def encoding(in_str, alphabet, type='ordinal'):
    if type == 'ordinal':
        try:
            vector = [alphabet.index(letter) for letter in in_str]
        except ValueError:
            raise ValueError('unknown letter in input: {}'.format(in_str))
    elif type == 'one-hot':
        vector = [[0 if char != letter else 1 for char in alphabet] for letter in in_str]
    else:
        raise ValueError('Unknown encoding type: {}'.format(type))
    return np.array(vector)


# loading routine for HIV resistance data
def load_data_hiv(type, drug_number):
    # define path to data file
    filepath = '../data/hivdb/{}_DataSet.fasta'.format(type)

    # define dictionary that maps resistance class to label
    class_to_label = {'L': 0, 'M': 1, 'H': 1}

    # load the fasta file containing the data
    tmp = list(SeqIO.parse(filepath, 'fasta'))

    # get the sequences and labels
    seq = [encoding(str(i.seq), PROTEIN) for i in tmp if i.id.split('|')[drug_number] != 'NA']
    label = [class_to_label[i.id.split('|')[drug_number]] for i in tmp if i.id.split('|')[drug_number] != 'NA']

    return np.array(seq), np.array(label)


# experiment routine
def experiment(args):
    """Function to perform the kernel SVM experiment.

    - **Parameters**::
        :param args: Parsed command line arguments
    """

    # load the input data using the correct routine
    if args.type == 'HIV':
        # load data
        X, Y = load_data_hiv(args.hiv_type, args.hiv_number)

        # load stratified folds previously created
        #   -> if your are missing the file 'hivdb_stratifiedFolds.pkl' (for HIV experiments), please run the file
        #      'hivdb_preparation.py' (for HIV experiments) first. This python script can be found in the data folder.
        with open('../data/hivdb/hivdb_stratifiedFolds.pkl', 'rb') as in_file:
            folds = pickle.load(in_file)
            folds = folds[args.hiv_type][args.hiv_name]
    else:
        ValueError('Unknown type of experiment: {}'.format(args.type))

    # iterate over the specified model parameters to perform a grid search
    for reg in args.reg_params:
        for degree in args.deg_params:
            print("Training SVM with polynomial degree = {} and C = {}...".format(degree, reg))

            # perform 5-fold stratified cross-validation
            results = []
            for fold_nb, fold in enumerate(folds):
                # initialize the SVM classifier using the oligo kernel
                clf = svm.SVC(C=reg, kernel="poly", degree=degree, probability=True)

                tic = timer()
                # fit classifier using the loaded data
                clf.fit(X[fold[0]], Y[fold[0]])
                toc = timer()
                print("    Finished SVM fit ({} samples), elapsed time: {:.2f}min".format(len(X[fold[0]]),
                                                                                          (toc - tic) / 60))

                # perform prediction on validation data
                tic = timer()
                pred_y = clf.predict_proba(X[fold[1]])
                toc = timer()
                print("    Finished prediction ({} samples), elapsed time: {:.2f}min\n".format(len(X[fold[1]]),
                                                                                               (toc - tic) / 60))

                # convert real targets into 2-dimensional array
                real_y = np.zeros((len(Y[fold[1]]), 2))
                for i, label in enumerate(Y[fold[1]]):
                    real_y[i, label] = 1

                res = compute_metrics(real_y, pred_y)
                results.append(res.to_dict())

            # store the evaluation results of the current experiment
            filename = '/validation_results_{}_{}.pkl'.format(degree, reg)
            with open(args.outdir + filename, 'wb') as out_file:
                pickle.dump(results, out_file, pickle.HIGHEST_PROTOCOL)


def evaluation(args):
    # evaluation needs pandas
    import pandas as pd

    # check if file exists
    try:
        with open(args.outdir + '/validation_results.pkl', 'rb') as in_file:
            val_results = pickle.load(in_file)
    except FileNotFoundError:
        raise ValueError('Specified result file does not exist')
    except Exception as e:
        raise('Unknown error: {}'.format(e))

    # combine results of all folds into one data frame
    results = []
    for val_res in val_results:
        results.append(pd.DataFrame.from_dict(val_res))
    df_res = pd.concat(results, axis=1)

    # calculate mean and std over all folds
    df_res['mean'] = df_res.mean(axis=1)
    df_res['std'] = df_res[:-1].std(axis=1)

    #print the results
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df_res)


def main():
    # load command line arguments
    args = load_args()

    # check if the evaluation mode was specified
    if args.eval:
        evaluation(args)

    # otherwise, run the experiment with the given arguments
    else:
        experiment(args)


if __name__ == '__main__':
    main()
