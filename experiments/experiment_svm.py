#############################################
# This file contains scripts to perform the #
# Oligo Kernel SVM experiments.             #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import numpy as np
from sklearn import svm
from Bio import SeqIO
import pickle


# define the oligo kernel function
def oligo_kernel(x, y, k, sigma):
    # initialize the gram matrix
    gram_matrix = np.zeros((x.shape[0], y.shape[0]))

    # iterate over each pair of inputs and fill the gram matrix
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):

            res = 0

            # iterate over the sequence positions
            for l in range(len(xi)):
                for m in range(len(yj)):

                    # skip if oligomers starting at position i and j are different
                    if x[l:l+k] != y[m:m+k]:
                        continue

                    res += np.exp(- 1/4*sigma**2 * (i - j)**2)

            gram_matrix[i, j] = res * np.sqrt(np.pi) * sigma

    return gram_matrix


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


# experiment routine
def experiment(info, params):
    """Function to perform the kernel SVM experiment.

    - **Parameters**::
        :param info: Tuple containing all information about the experiment in the form of
                     ('HIV', drug type, drug name, drug number) for HIV experiments and (...) for ...
            :type info: Tuple
        :param params: Tuple containing the parameters for the oligo kernel
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

    # create handle for the oligo kernel function
    oligo = lambda x, y: oligo_kernel(x, y, *params)

    # perform 5-fold stratified cross-validation
    for fold in folds:
        # initialize the SVM classifier using the oligo kernel
        clf = svm.SVC(kernel=oligo)

        # fit classifier using the loaded data
        clf.fit(X[fold[0]], Y[fold[0]])

        # perform prediction on validation data
        pred_y = clf.predict(X[fold[1]])

        print(pred_y, Y[fold[1]])
        return


if __name__ == '__main__':
    experiment(('HIV', 'PI', 'SQV', 6), (1, 1))
