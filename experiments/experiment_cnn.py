#############################################
# This file contains scripts to perform the #
# Convolutional Neural Net experiments.     #
#                                           #
# Author: Jonas Ditz                        #
#############################################

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from Bio import SeqIO
import pickle

from cmkn import compute_metrics, ClassBalanceLoss


# MACROS
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
    parser.add_argument("--epochs", dest="epochs", default=500, type=int,
                        help="Number of epochs used for training the CNN model.")
    parser.add_argument("--batch-size", dest="batch_size", default=64, type=int,
                        help="input batch size (default: 64)")

    # parse the arguments
    args = parser.parse_args()

    # set the random seeds
    np.random.seed(args.seed)

    # if an output directory is specified, create the dir structure to store the output of the current run
    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True
        args.outdir = args.outdir + "/CNN_experiment/{}/{}_{}/".format(args.type, args.hiv_type, args.hiv_name)
        if not os.path.exists(args.outdir):
            try:
                os.makedirs(args.outdir)
            except:
                pass

    return args


# one-hot encoding of strings
def encoding(in_str, alphabet, type='ordinal'):
    if type == 'ordinal':
        vector = [alphabet.index(letter) for letter in in_str]
    elif type == 'one-hot':
        vector = [[0 if char != letter else 1 for letter in in_str] for char in alphabet]
    else:
        raise ValueError('Unknown encoding type: {}'.format(type))
    return torch.Tensor(vector)


def count_classes(filepath, verbose=False, drug=7, num_classes=2):
    class_count = [0] * num_classes
    label_vec = []
    nb_samples = 0

    with open(filepath, 'rU') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if drug == -1:
                aux_lab = int(record.id.split('_')[1])
                class_count[aux_lab] += 1
                label_vec.append(aux_lab)
            else:
                aux_lab = record.id.split('|')[drug]
                if num_classes == 2:
                    if aux_lab == 'H':
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == 'M':
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == 'L':
                        class_count[0] += 1
                        label_vec.append(0)
                    else:
                        continue
                else:
                    if aux_lab == 'H':
                        class_count[2] += 1
                        label_vec.append(2)
                    elif aux_lab == 'M':
                        class_count[1] += 1
                        label_vec.append(1)
                    elif aux_lab == 'L':
                        class_count[0] += 1
                        label_vec.append(0)
                    else:
                        continue
            nb_samples += 1

    if verbose:
        print("number of samples in dataset: {}".format(nb_samples))
        if num_classes == 2:
            print("class balance: resistant = {}, susceptible = {}".format(class_count[1] / nb_samples,
                                                                           class_count[0] / nb_samples))
        else:
            print("class balance: H = {}, M = {}, L = {}".format(class_count[2] / nb_samples, class_count[1] /
                                                                 nb_samples, class_count[0] / nb_samples))

    if num_classes == 2:
        expected_loss = -(class_count[0] / nb_samples) * np.log(0.5) - \
                        (class_count[1] / nb_samples) * np.log(0.5)
    else:
        expected_loss = -(class_count[0] / nb_samples) * np.log(0.33) - \
                        (class_count[1] / nb_samples) * np.log(0.33) -\
                        (class_count[2] / nb_samples) * np.log(0.33)

    if verbose:
        print("expected loss: {}".format(expected_loss))

    if num_classes == 2:
        rand_guess = (class_count[0] / nb_samples) ** 2 + (class_count[1] / nb_samples) ** 2
    else:
        rand_guess = (class_count[0] / nb_samples) ** 2 + (class_count[1] / nb_samples) ** 2 + \
                     (class_count[2] / nb_samples) ** 2

    if verbose:
        print("expected accuracy with random guessing: {}".format(rand_guess))

    return class_count, expected_loss, np.array(label_vec)


class Net(nn.Module):
    def __init__(self, type):
        # call __init__ of parent class
        super(Net, self).__init__()

        if type == 'PI':
            out_len = 10
        else:
            out_len = 38

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
            nn.Linear(in_features=32 * out_len, out_features=200),
            nn.Linear(in_features=200, out_features=2)
        )

    def forward(self, x_in, proba=False):
        x_out = self.embedding(x_in)
        x_out = self.cnn_layers(torch.transpose(x_out, 1, 2))
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.linear_layers(x_out)
        if proba:
            return x_out.sigmoid()
        else:
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
            self.encode_type = 'ordinal'

        else:
            ValueError('Unknown experiment type: {}'.format(exp_params[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # extract the sequence of the item
        seq = encoding(self.data[item].seq, PROTEIN, self.encode_type)
        if self.encode_type == 'ordinal':
            seq = seq.type(torch.IntTensor)

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


def experiment(args):
    # load input data using the correct routine
    if args.type == "HIV":
        # create dataset handle
        data_all = CustomDataset("../data/hivdb/{}_DataSet.fasta".format(args.hiv_type), extension="fasta",
                                 exp_params=[args.type, args.hiv_number])

        # get class count for ClassBalanceLoss
        args.class_count, args.expected_loss, _ = count_classes("../data/hivdb/{}_DataSet.fasta".format(args.hiv_type),
                                                                False, args.hiv_number)

        # load stratified folds previously created
        #   -> if your are missing the file 'hivdb_stratifiedFolds.pkl' (for HIV experiments), please run the file
        #      'hivdb_preparation.py' (for HIV experiments) first. This python script can be found in the data folder.
        with open('../data/hivdb/hivdb_stratifiedFolds.pkl', 'rb') as in_file:
            folds = pickle.load(in_file)
            folds = folds[args.hiv_type][args.hiv_name]
    else:
        ValueError('Unknown type of experiment')

    # iterate over each fold of the stratified cross-validation
    results = []
    for fold in folds:
        # initialize the CNN model
        model = Net(args.hiv_type)

        # initialize optimizer and loss function
        criterion = ClassBalanceLoss(args.class_count, 2, 'sigmoid', 0.99, 1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

        # create subsets of the data that represent the training and validation split of the current fold and make
        # these subsets accessable with a DataLoader
        data_train = data.Subset(data_all, fold[0])
        loader_train = data.DataLoader(data_train, batch_size=args.batch_size, shuffle=False)
        data_val = data.Subset(data_all, fold[1])
        loader_val = data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False)

        # train the model
        print('Start model training...')
        model.train(True)
        epoch_loss = None
        for epoch in range(args.epochs):
            # if the learning rate scheduler is 'ReduceLROnPlateau' and there is a current loss, the next
            # lr step needs the current loss as input
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                if epoch_loss is not None:
                    lr_scheduler.step(epoch_loss)

            # otherwise call the step() function of the learning rate scheduler
            else:
                lr_scheduler.step()

            # iterate over the training data
            running_loss = 0.0
            for inputs, labels, *_ in loader_train:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimizer
                outputs = model(inputs)
                loss = criterion(outputs, labels.argmax(1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(loader_train)
            if epoch % 25 == 24:  # print every 500 mini-batches
                print("    {}, loss: {}".format(epoch + 1, epoch_loss))

        # calculate validation performance of trained model
        model.train(False)
        n_samples = len(loader_val.dataset)
        batch_start = 0
        for i, (inputs, labels, *_) in enumerate(loader_val):
            # get the current batch size
            batch_size = inputs.shape[0]

            # do not keep track of the gradients
            with torch.no_grad():
                batch_out = model(inputs, True)

            # initialize tensor that holds the results of the forward propagation for each sample
            if i == 0:
                outputs = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = torch.Tensor(n_samples, labels.shape[-1]).type_as(labels)

            # update output and target_output tensor with the current results
            outputs[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = labels

            # continue with the next batch
            batch_start += batch_size

        # get performance matrices
        outputs.squeeze_(-1)
        res = compute_metrics(target_output, outputs)
        results.append(res.to_dict())

    # store the evaluation results of the current experiment
    filename = '/validation_results_{}_{}.pkl'.format(args.epochs, args.batch_size)
    with open(args.outdir + filename, 'wb') as out_file:
        pickle.dump(results, out_file, pickle.HIGHEST_PROTOCOL)


def evaluation(args):
    # evaluation needs pandas
    import pandas as pd

    # check if file exists
    try:
        filename = '/validation_results_{}_{}.pkl'.format(args.epochs, args.batch_size)
        with open(args.outdir + filename, 'rb') as in_file:
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

    # print the results
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print("Results for CNN trained for {} epochs with batch size {}\n".format(args.epochs, args.batch_size))
        print(df_res)


def main():
    # parse arguments
    args = load_args()

    # check if the evaluation parameter was set
    if args.eval:
        evaluation(args)

    # if not, perform normal training of a CNN model
    else:
        experiment(args)


if __name__ == '__main__':
    main()
