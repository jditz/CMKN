######################################################
# This file converts raw HIVdb data files (PI, NRTI, #
# NNRTI) into easy-to-use fasta files for            #
# classification.                                    #
#                                                    #
# Author: Jonas Ditz                                 #
######################################################

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold
import pickle

# define consensus sequences for PI and NRTI/NNRTI (these sequences were provided by HIVdb)
PROTEASE = 'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF'
#RT = 'PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYAGIKVKQLCKLLRGTKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKIATESIVIWGKTPKFKLPIQKETWEAWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTDRGRQKVVSLTDTTNQKTELQAIHLALQDSGLEVNIVTDSQYALGIIQAQPDKSESELVSQIIEQLIKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVL'
RT = 'PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWT'

# define resistance cutoffs for each drug
CUTOFFS = {'FPV': (3, 15), 'ATV': (3, 15), 'IDV': (3, 15), 'LPV': (9, 55), 'NFV': (3, 6), 'SQV': (3, 15), 'TPV': (2, 8),
           'DRV': (10, 90),
           '3TC': (5, 25), 'ABC': (2, 6), 'AZT': (3, 15), 'D4T': (1.5, 3), 'DDI': (1.5, 3), 'TDF': (1.5, 3),
           'EFV': (3, 10), 'NVP': (3, 10), 'ETR': (3, 10), 'RPV': (3, 10)}

# define column numbers for each drug in the corresponding txt/fasta files
COLUMNS_PI = {1: 'FPV', 2: 'ATV', 3: 'IDV', 4: 'LPV', 5: 'NFV', 6: 'SQV', 7: 'TPV', 8: 'DRV'}
COLUMNS_NRTI = {1: '3TC', 2: 'ABC', 3: 'AZT', 4: 'D4T', 5: 'DDI', 6: 'TDF'}
COLUMNS_NNRTI = {1: 'EFV', 2: 'NVP', 3: 'ETR', 4: 'RPV'}


def create_fasta_pi():
    """ This function takes the PI_DataSet.txt file provided by HIVdb
    (https://hivdb.stanford.edu/download/GenoPhenoDatasets/PI_DataSet.txt) and converts it into a fasta file where the
    labels for each PI drug (susceptible or resistant) is given in the fasta id string
    """

    # mutation information starts at position 9 in the HIVdb file
    mut_start = 9

    # define the input and output filenames
    in_file = 'PI_DataSet.txt'
    out_file = 'PI_DataSet.fasta'

    # open the output fasta file
    f_out = open(out_file, 'w')

    # open HIVdb file
    with open(in_file, 'r') as f_in:

        # read in the header line
        header = f_in.readline().strip().split('\t')

        # iterate through each entry in HIVdb's file
        for l in f_in:

            # convert string into list
            tmp = l.strip().split('\t')

            # start the id string of the fasta output
            id_string = '>' + tmp[0]

            # iterate through the fold resistance of each drug and convert it into susceptible or resistant label
            for i in range(1, mut_start):

                # add 'NA' to the id string if no fold resistance value is available for the current drug
                if tmp[i] == 'NA':
                    id_string += '|NA'

                # otherwise, convert fold resistance value into label
                else:
                    if float(tmp[i]) < CUTOFFS[COLUMNS_PI[i]][0]:
                        id_string += '|L'
                    elif float(tmp[i]) >= CUTOFFS[COLUMNS_PI[i]][1]:
                        id_string += '|H'
                    else:
                        id_string += '|M'

            # create string to store isolate's sequence
            seq = list(PROTEASE)

            # iterate over each sequence position and replace amino acid in the consensus sequence with the recorded
            # mutation
            for i in range(mut_start, mut_start+99):

                # skip each position without a recorded mutation
                if tmp[i] == '-':
                    continue

                seq[i-mut_start] = tmp[i]

            # write id string and sequence to the output fasta file
            f_out.write(id_string + '\n')
            f_out.write(''.join(seq) + '\n')

    # close the output file
    f_out.close()


def create_fasta_nrti():
    """ This function takes the NRTI_DataSet.txt file provided by HIVdb
    (https://hivdb.stanford.edu/download/GenoPhenoDatasets/NRTI_DataSet.txt) and converts it into a fasta file where
    the labels for each NRTI drug (susceptible or resistant) is given in the fasta id string
    """

    # mutation information starts at position 9 in the HIVdb file
    mut_start = 7

    # define the input and output filenames
    in_file = 'NRTI_DataSet.txt'
    out_file = 'NRTI_DataSet.fasta'

    # open the output fasta file
    f_out = open(out_file, 'w')

    # open HIVdb file
    with open(in_file, 'r') as f_in:

        # read in the header line
        header = f_in.readline().strip().split('\t')

        # iterate through each entry in HIVdb's file
        for l in f_in:

            # convert string into list
            tmp = l.strip().split('\t')

            # start the id string of the fasta output
            id_string = '>' + tmp[0]

            # iterate through the fold resistance of each drug and convert it into susceptible or resistant label
            for i in range(1, mut_start):

                # add 'NA' to the id string if no fold resistance value is available for the current drug
                if tmp[i] == 'NA':
                    id_string += '|NA'

                # otherwise, convert fold resistance value into label
                else:
                    if float(tmp[i]) < CUTOFFS[COLUMNS_NRTI[i]][0]:
                        id_string += '|L'
                    elif float(tmp[i]) >= CUTOFFS[COLUMNS_NRTI[i]][1]:
                        id_string += '|H'
                    else:
                        id_string += '|M'

            # create string to store isolate's sequence
            seq = list(RT)

            # iterate over each sequence position and replace amino acid in the consensus sequence with the recorded
            # mutation
            for i in range(mut_start, mut_start + 240):

                # skip each position without a recorded mutation
                if tmp[i] == '-':
                    continue

                seq[i - mut_start] = tmp[i]

            # write id string and sequence to the output fasta file
            f_out.write(id_string + '\n')
            f_out.write(''.join(seq) + '\n')

    # close the output file
    f_out.close()


def create_fasta_nnrti():
    """ This function takes the NNRTI_DataSet.txt file provided by HIVdb
    (https://hivdb.stanford.edu/download/GenoPhenoDatasets/NNRTI_DataSet.txt) and converts it into a fasta file where
    the labels for each NNRTI drug (susceptible or resistant) is given in the fasta id string
    """

    # mutation information starts at position 9 in the HIVdb file
    mut_start = 5

    # define the input and output filenames
    in_file = 'NNRTI_DataSet.txt'
    out_file = 'NNRTI_DataSet.fasta'

    # open the output fasta file
    f_out = open(out_file, 'w')

    # open HIVdb file
    with open(in_file, 'r') as f_in:

        # read in the header line
        header = f_in.readline().strip().split('\t')

        # iterate through each entry in HIVdb's file
        for l in f_in:

            # convert string into list
            tmp = l.strip().split('\t')

            # start the id string of the fasta output
            id_string = '>' + tmp[0]

            # iterate through the fold resistance of each drug and convert it into susceptible or resistant label
            for i in range(1, mut_start):

                # add 'NA' to the id string if no fold resistance value is available for the current drug
                if tmp[i] == 'NA':
                    id_string += '|NA'

                # otherwise, convert fold resistance value into label
                else:
                    if float(tmp[i]) < CUTOFFS[COLUMNS_NNRTI[i]][0]:
                        id_string += '|L'
                    elif float(tmp[i]) >= CUTOFFS[COLUMNS_NNRTI[i]][1]:
                        id_string += '|H'
                    else:
                        id_string += '|M'

            # create string to store isolate's sequence
            seq = list(RT)

            # iterate over each sequence position and replace amino acid in the consensus sequence with the recorded
            # mutation
            for i in range(mut_start, mut_start + 240):

                # skip each position without a recorded mutation
                if tmp[i] == '-':
                    continue

                seq[i - mut_start] = tmp[i]

            # write id string and sequence to the output fasta file
            f_out.write(id_string + '\n')
            f_out.write(''.join(seq) + '\n')

    # close the output file
    f_out.close()


# function to create indices that can be used for a 5-fold stratified cross-validation
#   -> these lists allow to perform 5-fold stratified cross-validation that is consistent across experiments
def create_folds():
    # set the numpy seed to create reproducable results
    np.random.seed(1)

    # define dictionary that maps resistance class to label
    class_to_label = {'L': 0, 'M': 1, 'H': 1}

    # initalize sklearn's StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # initialize dictionaries to hold all folds
    folds_pi = {}
    folds_nrti = {}
    folds_nnrti = {}

    # iterate through all PI drugs and create the stratified folds
    for drug_number in COLUMNS_PI.keys():

        # parse the fasta file
        tmp = list(SeqIO.parse('PI_DataSet.fasta', 'fasta'))

        # get the sequences and labels
        seq = [i.seq for i in tmp if i.id.split('|')[drug_number] != 'NA']
        label = [class_to_label[i.id.split('|')[drug_number]] for i in tmp if i.id.split('|')[drug_number] != 'NA']

        # create np.array objects for the creation of stratified fold indices
        vec_indices = np.arange(len(seq))
        vec_label = np.array(label)

        # get train and validation indices for each fold
        aux_folds = []
        for fold, split_idx in enumerate(skf.split(vec_indices, vec_label)):
            aux_folds.append((vec_indices[split_idx[0]], vec_indices[split_idx[1]]))

        # add the fold indices to the PI dictionary
        folds_pi[COLUMNS_PI[drug_number]] = aux_folds

    # iterate through all NRTI drugs and create the stratified folds
    for drug_number in COLUMNS_NRTI.keys():

        # parse the fasta file
        tmp = list(SeqIO.parse('NRTI_DataSet.fasta', 'fasta'))

        # get the sequences and labels
        seq = [i.seq for i in tmp if i.id.split('|')[drug_number] != 'NA']
        label = [class_to_label[i.id.split('|')[drug_number]] for i in tmp if i.id.split('|')[drug_number] != 'NA']

        # create np.array objects for the creation of stratified fold indices
        vec_indices = np.arange(len(seq))
        vec_label = np.array(label)

        # get train and validation indices for each fold
        aux_folds = []
        for fold, split_idx in enumerate(skf.split(vec_indices, vec_label)):
            aux_folds.append((vec_indices[split_idx[0]], vec_indices[split_idx[1]]))

        # add the fold indices to the NRTI dictionary
        folds_nrti[COLUMNS_NRTI[drug_number]] = aux_folds

    # iterate through all NNRTI drugs and create the stratified folds
    for drug_number in COLUMNS_NNRTI.keys():

        # parse the fasta file
        tmp = list(SeqIO.parse('NNRTI_DataSet.fasta', 'fasta'))

        # get the sequences and labels
        seq = [i.seq for i in tmp if i.id.split('|')[drug_number] != 'NA']
        label = [class_to_label[i.id.split('|')[drug_number]] for i in tmp if i.id.split('|')[drug_number] != 'NA']

        # create np.array objects for the creation of stratified fold indices
        vec_indices = np.arange(len(seq))
        vec_label = np.array(label)

        # get train and validation indices for each fold
        aux_folds = []
        for fold, split_idx in enumerate(skf.split(vec_indices, vec_label)):
            aux_folds.append((vec_indices[split_idx[0]], vec_indices[split_idx[1]]))

        # add the fold indices to the NNRTI dictionary
        folds_nnrti[COLUMNS_NNRTI[drug_number]] = aux_folds

    # store the fold dictionaries in a file
    with open('hivdb_stratifiedFolds.pkl', 'wb') as out_file:
        pickle.dump({'PI': folds_pi, 'NRTI': folds_nrti, 'NNRTI': folds_nnrti}, out_file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    create_fasta_pi()
    create_fasta_nrti()
    create_fasta_nnrti()
    create_folds()
