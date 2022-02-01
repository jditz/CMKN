######################################################
# This file prepares the DGSplicer data set for use  #
# in CMKN experiments. The script assumes that       #
# positive and negative sequences are provided in    #
# different files.                                   #
#                                                    #
# Author: Jonas Ditz                                 #
######################################################

import numpy as np
from sklearn.model_selection import train_test_split

# Macros
FILES = {'acceptor': ["A_true_27_9.txt", "A_false_27_9.txt"],
         'donor': ["D_true_9_9.txt", "D_false_9_9.txt"]}


def create_files(seq_type):
    # initialize auxiliary variables
    aux_seq = []
    aux_label = []

    # read in the files holding the sequences
    for i, file in enumerate(FILES[seq_type]):
        with open(file, 'r') as in_file:
            for line in in_file:
                aux_seq.append(line.rstrip().upper())

                # make sure the correct label is stored
                if i == 0:
                    aux_label.append(1)
                else:
                    aux_label.append(0)

    # create stratified train and test split of the data
    seq_train, seq_test, label_train, label_test = train_test_split(aux_seq, aux_label, test_size=0.2, random_state=1,
                                                                    stratify=np.array(aux_label))

    # write training data to file
    with open('DGSplicer_{}_train.fasta'.format(seq_type), 'w') as out_train:
        for i, record in enumerate(zip(seq_train, label_train)):
            out_train.write('>{}_{}\n'.format(i, record[1]))
            out_train.write('{}\n'.format(record[0]))

    # write test data to file
    with open('DGSplicer_{}_test.fasta'.format(seq_type), 'w') as out_test:
        for i, record in enumerate(zip(seq_test, label_test)):
            out_test.write('>{}_{}\n'.format(i, record[1]))
            out_test.write('{}\n'.format(record[0]))


if __name__ == '__main__':
    print('Preprocessing Acceptor Sequences...')
    create_files('acceptor')
    print('Preprocessing Donor Sequences...')
    create_files('donor')
