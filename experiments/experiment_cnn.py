#############################################
# This file contains scripts to perform the #
# Convolutional Neural Net experiments.     #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from Bio import SeqIO
import pickle


# MACROS
EMBEDDING_DIR = {"A": 1, "a": 1, "B": 2, "b": 2, "C": 3, "c": 3, "D": 4, "d": 4, "E": 5, "e": 5, "F": 6, "f": 6, "G": 7,
                 "g": 7, "H": 8, "h": 8, "I": 9, "i": 9, "J": 10, "j": 10, "K": 11, "k": 11, "L": 12, "l": 12, "M": 13,
                 "m": 13, "N": 14, "n": 14, "O": 15, "o": 15, "P": 16, "p": 16, "Q": 17, "q": 17, "R": 18, "r": 18,
                 "S": 19, "s": 19, "T": 20, "t": 20, "U": 21, "u": 21, "V": 22, "v": 22, "W": 23, "w": 23, "X": 24,
                 "x": 24, "Y": 25, "y": 25, "Z": 26, "z": 26}


class Net(nn.Module):
    def __init__(self):
        # define the embedding layer of the CNN
        self.embedding = nn.Embedding(num_embeddings=27, embedding_dim=3)

        # define the convolutional layers of the network
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=9),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9),
            nn.ReLU(inplace=True)
        )

        # define the linear layers of the network
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=32 * 19, out_features=200),
            nn.Linear(in_features=200, out_features=2)
        )

    def forward(self, x_in):
        x_out = self.embedding(x_in)
        x_out = self.cnn_layers(x_out)
        x_out = x_out.view(x_out.shape(0), -1)
        x_out = self.linear_layers(x_out)
        return x_out


class CustomDataset(data.Dataset):
    def __init__(self, filepath, extension, exp_params):
        '''exp_params: Parameter of the experiment, e.g. ('HIV', drug_number) for an HIV experiment.'''

        # check which type of experiment should be performed
        if exp_params[0] == 'HIV':
            # load dataset
            self.data = list(SeqIO.parse(filepath, extension))

            # make sure that data only contains valid sequences
            self.data = [i for i in self.data if i.id.split('|')[exp_params[1]] != 'NA']

            # check if all sequences have the same length
            #   -> store length if true, raise exception otherwise
            if all(len(x.seq) == len(self.data[0].seq) for x in self.data):
                self.seq_len = len(self.data[0].seq)
            else:
                raise ValueError('Sequences are of different length!')

            # define class mappings
            self.class_to_idx = {'L': 0, 'M': 1, 'H': 1}
            self.all_categories = ['susceptible', 'resistant']

            # store parameters
            self.drug_nb = exp_params[1]

        else:
            ValueError('Unknown experiment type: {}'.format(exp_params[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # extract the sequence of the item
        seq = str(self.data[item].seq)

        # convert id string into label vector
        label = torch.zeros(2)

        id_str = self.data[item].id
        try:
            aux_lab = id_str.split('|')[self.drug_nb]
            if aux_lab != 'NA':
                label[self.class_to_idx[aux_lab]] = 1.0

        except Exception as e:
            print('Exception at index {}:'.format(item), e)

        return seq, label


def experiment():
    pass


if __name__ == '__main__':
    pass
