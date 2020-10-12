import os
import argparse

from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import numpy as np

from con import CON2, CONDataset, ClassBalanceLoss, kmer2dict, build_kmer_ref, Hook

from Bio import SeqIO


# MACROS

NAME = 'CON_experiment'
DATA_DIR = './data/'


# extend custom data handler for the used dataset
class CustomHandler(CONDataset):
    def __init__(self, filepath, kmer_size=3, drug='SQV', nb_classes=3, clean_set=True):
        if drug == 'SQV':
            self.drug_nb = 7
        elif drug == 'LPV':
            self.drug_nb = 8
        elif drug == 'DRV':
            self.drug_nb = 9
        elif drug == 'ATV':
            self.drug_nb = 10
        elif drug == 'NFV':
            self.drug_nb = 11
        elif drug == 'IDV':
            self.drug_nb = 12
        elif drug == 'FPV':
            self.drug_nb = 13
        elif drug == 'TPV':
            self.drug_nb = 14
        else:
            raise ValueError('The dataset does not contain any information about drug resilience of HIV' +
                             '\nagainst the specified drug!\n')

        if clean_set:
            aux_tup = (self.drug_nb, 'NA')
        else:
            aux_tup = None

        super(CustomHandler, self).__init__(filepath, kmer_size=kmer_size, alphabet='PROTEIN_AMBI',
                                            clean_set=aux_tup)

        self.nb_classes = nb_classes
        if nb_classes == 2:
            self.class_to_idx = {'H': 0, 'M': 0, 'L': 1}
            self.all_categories = ['res', 'no_res']
        else:
            self.class_to_idx = {'H': 0, 'M': 1, 'L': 2}
            self.all_categories = ['H', 'M', 'L']

    def __getitem__(self, idx):
        # get a sample using the parent __getitem__ function
        sample = super(CustomHandler, self).__getitem__(idx)

        # initialize label tensor
        if len(sample[1]) == 1:
            labels = torch.zeros(self.nb_classes)
        else:
            labels = torch.zeros(len(sample[1]), self.nb_classes)

        # iterate through all id strings and update the label tensor, accordingly
        for i, id_str in enumerate(sample[1]):
            try:
                aux_lab = id_str.split('|')[self.drug_nb]
                if aux_lab != 'NA':
                    if len(sample[1]) == 1:
                        labels[self.class_to_idx[aux_lab]] = 1.0
                    else:
                        labels[i, self.class_to_idx[aux_lab]] = 1.0

            except:
                continue

        # return the sample with updated label tensor
        sample = (sample[0], labels)
        return sample


def load_args():
    """
    Function to create an argument parser
    """
    parser = argparse.ArgumentParser(description="CON example experiment")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=4, metavar='M',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', dest="nb_epochs", type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument("--out-channels", dest="out_channels", metavar="m", default=[40, 128, 512, 1024], nargs='+',
                        type=int,
                        help="number of out channels for each oligo kernel and conv layer (default [40, 128, 512, 1024])")
    parser.add_argument("--strides", dest="strides", metavar="s", default=[1, 1, 1, 1], nargs='+', type=int,
                        help="stride value for each layer (default: [1, 1, 1, 1])")
    parser.add_argument("--paddings", dest="paddings", metavar="p", default=['SAME', 'SAME', 'SAME'], nargs="+",
                        type=str, help="padding values for each convolutional layer (default ['SAME', 'SAME', 'SAME'])")
    parser.add_argument("--kernel-sizes", dest="kernel_sizes", metavar="k", default=[5, 5, 9], nargs="+", type=int,
                        help="kernel sizes for oligo and convolutional layers (default [5, 5, 9])")
    parser.add_argument("--sigma", dest="sigma", default=4, type=float,
                        help="sigma for the oligo kernel layer (default: 4)")
    parser.add_argument("--scale", dest="scale", default=100, type=int,
                        help="scaling parameter for the oligo kernel layer (Default: 100)")
    parser.add_argument("--kmer", dest="kmer_size", default=1, type=int,
                        help="length of the k-mers used for the oligo kernel layer (default 3)")
    parser.add_argument("--num-classes", dest="num_classes", default=2, type=int,
                        help="number of classes in the prediction task")
    parser.add_argument("--kfold", dest="kfold", default=5, type=int, help="k-fold cross validation (default: 5)")
    parser.add_argument("--penalty", metavar="penal", dest="penalty", default='l2', type=str, choices=['l2', 'l1'],
                        help="regularization used in the last layer (default: l2)")
    parser.add_argument("--regularization", type=float, default=1e-6,
                        help="regularization parameter for sup CON")
    parser.add_argument("--preprocessor", type=str, default='standard_row', choices=['standard_row', 'standard_col'],
                        help="preprocessor for last layer of CON (default: standard_row)")
    parser.add_argument("--outdir", metavar="outdir", dest="outdir", default='output', type=str,
                        help="output path(default: '')")
    parser.add_argument("--use-cuda", action='store_true', default=True, help="use gpu (default: False)")
    parser.add_argument("--noise", type=float, default=0.0, help="perturbation percent")
    parser.add_argument("--file", dest="filepath", default="./data/test_dataset.fasta", type=str,
                        help="path to the file containing the dataset.")
    parser.add_argument("--extension", dest="extension", default="fasta", type=str,
                        help="extension of the file containing the dataset (default fasta)")
    parser.add_argument("--drug", dest="drug", default="SQV", type=str,
                        help="specifies the drug that will be used to classify virus resilience (default SQV)")

    # parse the arguments
    args = parser.parse_args()

    # GPU will only be used if available
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    # 'num-anchors', 'subsampling', and 'sigma' have to be of same length
    if not len(args.out_channels) == len(args.strides) == len(args.paddings) + 1 == len(args.kernel_sizes) + 1:
        raise ValueError('The size combination of out_channels, strides, paddings, and kernel_sizes is invalid!\n' +
                         'out_channels and strides have to have the same length while the length of paddings and ' +
                         'kernel_sizes have to be one less than the other two.')

    # number of layers is equal to length of the 'num-anchors' vector
    args.n_layer = len(args.out_channels)

    # set the random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # if an output directory is specified, create the dir structure to store the output of the current run
    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/{}".format(NAME)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/{}".format(args.drug)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/classes_{}".format(args.num_classes)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/kmer_{}".format(args.kmer_size)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/params_{}_{}".format(args.sigma, args.scale)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/anchors_{}'.format(args.out_channels[0])
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/layers_{}'.format(args.n_layer)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        args.outdir = outdir

    return args


def count_classes(filepath, verbose=False, drug=7, num_classes=3):
    class_count = [0] * num_classes
    nb_samples = 0

    with open(filepath, 'rU') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            aux_lab = record.id.split('|')[drug]
            if num_classes == 2:
                if aux_lab == 'H':
                    class_count[0] += 1
                elif aux_lab == 'M':
                    class_count[0] += 1
                elif aux_lab == 'L':
                    class_count[1] += 1
                else:
                    continue
            else:
                if aux_lab == 'H':
                    class_count[0] += 1
                elif aux_lab == 'M':
                    class_count[1] += 1
                elif aux_lab == 'L':
                    class_count[2] += 1
                else:
                    continue
            nb_samples += 1

    if verbose:
        print("number of samples in dataset: {}".format(nb_samples))
        if num_classes == 2:
            print("class balance: pos = {}, neg = {}".format(class_count[0] / nb_samples, class_count[1] / nb_samples))
        else:
            print("class balance: H = {}, M = {}, L = {}".format(class_count[0] / nb_samples, class_count[1] /
                                                                 nb_samples, class_count[2] / nb_samples))

    if num_classes == 2:
        expected_loss = -(class_count[0] / nb_samples) * np.log(0.33) - \
                        (class_count[1] / nb_samples) * np.log(0.33)
    else:
        expected_loss = -(class_count[0] / nb_samples) * np.log(0.33) - \
                        (class_count[1] / nb_samples) * np.log(0.33) -\
                        (class_count[2] / nb_samples) * np.log(0.33)

    if verbose:
        print("expected loss: {}".format(expected_loss))

    if num_classes == 2:
        rand_guess = (class_count[0] / nb_samples) * 0.5 + (class_count[1] / nb_samples) * 0.5
    else:
        rand_guess = (class_count[0] / nb_samples) * 0.33 + (class_count[1] / nb_samples) * 0.33 + \
                     (class_count[2] / nb_samples) * 0.33

    if verbose:
        print("expected accuracy with random guessing: {}".format(rand_guess))

    return class_count, expected_loss


def main():
    # set parameters for the test run
    args = load_args()
    args.alphabet = 'ARNDCQEGHILKMFPSTWYVXBZJUO'

    # create dictionary that maps kmers to index
    kmer_dict = kmer2dict(args.kmer_size, args.alphabet)

    # build tensor holding reference positions
    ref_pos = build_kmer_ref(args.filepath, args.extension, kmer_dict, args.kmer_size)

    # initialize con model
    model = CON2(out_channels_list=args.out_channels, ref_kmerPos=ref_pos, filter_sizes=args.kernel_sizes,
                 strides=args.strides, paddings=args.paddings, num_classes=args.num_classes,
                 kernel_args=[args.sigma, args.scale])

    # load data
    data_all = CustomHandler(args.filepath, kmer_size=args.kmer_size, drug=args.drug, nb_classes=args.num_classes,
                             clean_set=True)

    # Creating data indices for training and validation splits:
    validation_split = .2
    shuffle_dataset = True
    random_seed = args.seed
    dataset_size = len(data_all)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    args.train_indices, args.val_indices = indices[split:], indices[:split]

    # Creating PyTorch data samplers and loaders:
    data_train = Subset(data_all, args.train_indices)
    data_val = Subset(data_all, args.val_indices)

    loader_train = DataLoader(data_train, batch_size=args.batch_size)
    loader_val = DataLoader(data_val, batch_size=args.batch_size)

    # get distribution of classes for class balance loss
    args.class_count, args.expected_loss = count_classes(args.filepath, True, data_all.drug_nb, args.num_classes)

    # initialize optimizer and loss function
    if args.num_classes == 2:
        criterion = ClassBalanceLoss(args.class_count, args.num_classes, 'sigmoid', 0.99, 1.0)
    else:
        criterion = ClassBalanceLoss(args.class_count, args.num_classes, 'cross_entropy', 0.99, 1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

    # train model
    acc, loss = model.sup_train(loader_train, criterion, optimizer, lr_scheduler, val_loader=loader_val,
                                epochs=args.nb_epochs)

    # register a forward hook on the oligo kernel layer to inspect the kernel embedding
    hook_oligo = Hook(list(model._modules.items())[0][1])

    # iterate through data set and store the embeddings
    kernel_embeddings = []
    labels = []
    for phi, label, *_ in loader_train:
        phi.requires_grad = False
        data = torch.zeros(phi.size(0), 2, phi.size(-1))
        for i in range(phi.size(-1)):
            # project current position on the upper half of the unit circle
            x_circle = np.cos(((i + 1) / phi.size(-1)) * np.pi)
            y_circle = np.sin(((i + 1) / phi.size(-1)) * np.pi)

            # fill the input tensor
            data[:, 0, i] = x_circle
            data[:, 1, i] = y_circle

        # pass data through network to get the kernel embedding
        _ = model(data, phi)

        # store embeddings and labels
        aux_emb = []
        aux_lab = []
        for i in range(data.size(0)):
            aux_emb.append(hook_oligo.output[i, :, :].detach().numpy())
            aux_lab.append(label[i, :])
        kernel_embeddings.append(aux_emb)
        labels.append(aux_lab)

    # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
    # model
    torch.save({'args': args, 'state_dict': model.state_dict(), 'acc': acc, 'loss': loss,
                'kernel_embeddings': kernel_embeddings, 'labels': labels},
               args.outdir + "/CON_results_epochs" + str(args.nb_epochs) + ".pkl")

    try:
        # try to import pyplot
        import matplotlib.pyplot as plt

        # show the evolution of the acc and loss
        fig2, axs2 = plt.subplots(2, 2)
        axs2[0, 0].plot(acc['train'])
        axs2[0, 0].set_title("train accuracy")
        axs2[0, 0].set(xlabel='epoch', ylabel='accuracy')
        axs2[0, 1].plot(acc['val'])
        axs2[0, 1].set_title("val accuracy")
        axs2[0, 1].set(xlabel='epoch', ylabel='accuracy')
        axs2[1, 0].plot(loss['train'])
        axs2[1, 0].set_title("train loss")
        axs2[1, 0].set(xlabel='epoch', ylabel='loss')
        axs2[1, 1].plot(loss['val'])
        axs2[1, 1].set_title("val loss")
        axs2[1, 1].set(xlabel='epoch', ylabel='loss')
        # plt.show()
        plt.savefig(args.outdir + "/acc_loss.png")

        # show the position of the anchor points as a histogram
        anchor = (torch.acos(model.oligo.weight[:, 0]) / np.pi) * (ref_pos.size(1) - 1)
        anchor = anchor.detach().numpy()
        fig3 = plt.figure()
        plt.hist(anchor, bins=ref_pos.size(1))
        plt.xlabel('Position')
        plt.ylabel('# Anchor Points')
        plt.title('Distribution of anchor points')
        # plt.show()
        plt.savefig(args.outdir + "/anchor_positions.png")

    except:
        print("Cannot import matplotlib.pyplot")


if __name__ == '__main__':
    main()
