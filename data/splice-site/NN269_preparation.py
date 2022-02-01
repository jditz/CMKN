######################################################
# This file prepares the NN269 data set for use in   #
# CMKN experiments. The script assumes that positive #
# and negative sequences are provided in different   #
# files.
#                                                    #
# Author: Jonas Ditz                                 #
######################################################

# Macros
FILES = {'acceptor': {'train': ["NN269_Acceptor_Train_Positive.fasta", "NN269_Acceptor_Train_Negative.fasta"],
                      'test': ["NN269_Acceptor_Test_Positive.fasta", "NN269_Acceptor_Test_Negative.fasta"]},
         'donor': {'train': ["NN269_Donor_Train_Positive.fasta", "NN269_Donor_Train_Negative.fasta"],
                   'test': ["NN269_Donor_Test_Positive.fasta", "NN269_Donor_Test_Negative.fasta"]}}


def concat_files(seq_type='acceptor', mode="train"):
    """This function combines positive and negative sequences
    """
    # make sure that the correct files are selected
    pos_file = FILES[seq_type][mode][0]
    neg_file = FILES[seq_type][mode][1]

    # initialize needed variables
    new_file = "NN269_{}_{}.fasta".format(seq_type, mode)
    lines = []

    # iterate through the positive file and store all lines
    with open(pos_file, 'r') as f_in:
        for l in f_in:
            if l[0] == '>':
                # add the label to the id string
                lines.append(l.strip() + '_1')
            else:
                lines.append(l.strip())

    # iterate through the negative file and store all lines
    with open(neg_file, 'r') as f_in:
        for l in f_in:
            if l[0] == ">":
                # add the label to the id string
                lines.append(l.strip() + "_0")
            else:
                lines.append(l.strip())

    # write all lines to the new file
    with open(new_file, 'w') as f_out:
        for line in lines:
            f_out.write(line + "\n")


if __name__ == '__main__':
    concat_files('acceptor', 'train')
    concat_files('acceptor', 'test')
    concat_files('donor', 'train')
    concat_files('donor', 'test')
