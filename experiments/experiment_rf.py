#############################################
# This file contains scripts to perform the #
# Random Forest experiments.                #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Bio import SeqIO
import pickle


# loading routine for HIV resistance data
def load_data_hiv(type, drug_number):
    # define path to data file
    filepath = '../data/hivdb/{}_DataSet.fasta'.format(type)

    # define dictionary that maps resistance class to label
    class_to_label = {'L': 0, 'M': 1, 'H': 1}

    # load the fasta file containing the data
    tmp = list(SeqIO.parse(filepath, 'fasta'))

    # get the sequences and labels
    seq = [str(i.seq) for i in tmp if i.id.split('|')[drug_number] != 'NA']
    label = [class_to_label[i.id.split('|')[drug_number]] for i in tmp if i.id.split('|')[drug_number] != 'NA']

    return np.array(seq), np.array(label)


# experimental routine
def experiment(info, params):
    """Function to perform the random forest experiment.

    - **Parameters**::
        :param info: Tuple containing all information about the experiment in the form of
                     ('HIV', drug type, drug name, drug number) for HIV experiments and (...) for ...
            :type info: Tuple
        :param params: Tuple containing the parameters for the random forest classifier
            :type params: Tuple
    """

    # load the input data using the correct routine
    if info[0] == 'HIV':
        # load data
        X, Y = load_data_hiv(info[1], info[3])

        # load stratified folds previously created
        #   -> if your are missing the file 'hivdb_stratifiedFolds.pkl' (for HIV experiments), please run the file
        #      'hivdb_preparation.py' (for HIV experiments) first. This python script can be found in the data folder.
        with open('../data/hivdb/hivdb_stratifiedFolds.pkl', 'rb') as in_file:
            folds = pickle.load(in_file)
            folds = folds[info[1]][info[2]]
    else:
        ValueError('Unknown type of experiment')

    # perform 5-fold stratified cross-validation
    for fold in folds:
        # initialize the SVM classifier using the oligo kernel
        clf = RandomForestClassifier(n_estimators=params[0], max_features=params[1])

        # fit classifier using the loaded data
        clf.fit(X[fold[0]], Y[fold[0]])

        # perform prediction on validation data
        pred_y = clf.predict(X[fold[1]])

        print(pred_y, Y[fold[1]])


if __name__ == '__main__':
    experiment(('HIV', 'PI', 'SQV', 6), (500, 'sqrt'))
