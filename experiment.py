################################################
# Example experiment using a CON model.        #
#                                              #
# Author: Jonas Ditz                           #
# Contact: ditz@informatik.uni-tuebingen.de    #
################################################

import os
import argparse

from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import gradcheck
import torch.optim as optim
import numpy as np

from con import (CON2, CONDataset, ClassBalanceLoss, kmer2dict, build_kmer_ref,
                 plot_grad_flow, Hook, anchors_to_motivs)

from Bio import SeqIO
from sklearn.metrics import roc_auc_score


# MACROS

# set to True to enable network debugging
DEBUGGING = False

# each Boolean value decides if the corresponding debugging step will be performed
DEBUG_STEPS = [True, True, True, True, False, False, True]


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
    parser.add_argument('--epochs', dest="nb_epochs", type=int, default=10, metavar='N',
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
    parser.add_argument("--kmer", dest="kmer_size", default=3, type=int,
                        help="length of the k-mers used for the oligo kernel layer (default 3)")
    parser.add_argument("--num-classes", dest="num_classes", default=3, type=int,
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
    parser.add_argument("--use-cuda", action='store_true', default=False, help="use gpu (default: False)")
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


def test_exp():
    # set parameters for the test run
    args = load_args()
    args.class_count, args.expected_loss = count_classes(args.filepath, True)
    args.alphabet = 'ARNDCQEGHILKMFPSTWYVXBZJUO'

    # create dictionary that maps kmers to index
    kmer_dict = kmer2dict(args.kmer_size, args.alphabet)

    # build tensor holding reference positions
    ref_pos = build_kmer_ref(args.filepath, args.extension, kmer_dict, args.kmer_size)

    # initialize con model
    #model = CON([40, 128], ref_pos, [3], [1, 3], num_classes=3, kernel_funcs=['exp', 'exp_chen'],
    #            kernel_args_list=[[1.25, 1], [0.5]], kernel_args_trainable=[False, False])
    #model = CON([40], ref_pos, [], [1], num_classes=3, kernel_funcs=['exp'],
    #            kernel_args_list=[[0.5, 1]], kernel_args_trainable=[False])
    #model = CON2(out_channels_list=[40, 32, 64, 128, 512, 1024], ref_kmerPos=ref_pos, filter_sizes=[3, 5, 5, 5, 9],
    #             strides=[1, 1, 1, 1, 1, 1], paddings=['SAME', 'SAME', 'SAME', 'SAME', 'SAME'], num_classes=3,
    #             kernel_args=[4, 10])
    #model = CON2(out_channels_list=[40], ref_kmerPos=ref_pos, filter_sizes=[], strides=[1], paddings=[], num_classes=3,
    #             kernel_args=[4, 10])
    #model = CON2(out_channels_list=[40, 100], ref_kmerPos=ref_pos, filter_sizes=[3], strides=[1, 1], paddings=['SAME'],
    #             num_classes=3, kernel_args=[4, 10])
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

    # initialize optimizer and loss function
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = ClassBalanceLoss(args.class_count, len(args.class_count), 'cross_entropy', 0.99, 1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

    # DEBUGGING START
    if DEBUGGING:

        # STEP 1: Print the model and trainable parameters
        if DEBUG_STEPS[0]:
            print("\n\n********* DEBUGGING STEP 1 *********\n    -> print the model and parameters\n")
            print(model)
            for name, parameter in model.named_parameters():
                print(name, parameter.shape, "required_grad = {}".format(parameter.requires_grad))

        # STEP 2: check the initial loss
        if DEBUG_STEPS[1]:
            print("\n\n********* DEBUGGING STEP 2 *********\n    -> check initial loss of the model\n" +
                  "    -> initial loss should be close to expected loss\n")
            model.initialize()
            running_loss = 0
            for phi, target, *_ in loader_train:
                phi.requires_grad = False
                data = torch.zeros(phi.size(0), 2, phi.size(-1))
                for i in range(phi.size(-1)):
                    # project current position on the upper half of the unit circle
                    x_circle = np.cos(((i+1)/phi.size(-1)) * np.pi)
                    y_circle = np.sin(((i+1)/phi.size(-1)) * np.pi)

                    # fill the input tensor
                    data[:, 0, i] = x_circle
                    data[:, 1, i] = y_circle

                out = model(data, phi)
                loss = criterion(out, target.argmax(1))
                running_loss += loss.item() * data.size(0)
            init_loss = running_loss / len(loader_train.dataset)
            print("\ninitial loss: {}; expected loss: {}".format(init_loss, args.expected_loss))

        # STEP 3: train model on a single data point and see if it can overfit
        if DEBUG_STEPS[2]:
            print("\n\n********* DEBUGGING STEP 3 *********\n    -> train model on a single datapoint\n" +
                  "    -> model has to be able to overfit with zero loss and validation accuracy at chance level\n")
            data_debug = Subset(data_all, [args.val_indices[0]])
            loader_debug = DataLoader(data_debug, batch_size=1)

            print("length of debug dataset: {}".format(len(loader_debug.dataset)))
            print("length of train dataset: {}".format(len(loader_train.dataset)))
            print("length of val dataset: {}\n".format(len(loader_val.dataset)))

            model.sup_train(loader_debug, criterion, optimizer, lr_scheduler, val_loader=loader_train, epochs=3)

        # STEP 4: visualize the gradient flow
        if DEBUG_STEPS[3]:
            print("\n\n********* DEBUGGING STEP 4 *********\n    -> visualize the gradient flow through the network\n" +
                  "    -> re-initialize the network to overwrite the gradients from the overfitting")
            model.initialize()
            for phi, target, *_ in loader_train:
                phi.requires_grad = False
                data = torch.zeros(phi.size(0), 2, phi.size(-1))
                for i in range(phi.size(-1)):
                    # project current position on the upper half of the unit circle
                    x_circle = np.cos(((i + 1) / phi.size(-1)) * np.pi)
                    y_circle = np.sin(((i + 1) / phi.size(-1)) * np.pi)

                    # fill the input tensor
                    data[:, 0, i] = x_circle
                    data[:, 1, i] = y_circle

                out = model(data, phi)
                loss = criterion(out, target.argmax(1))
                loss.backward()
                plot_grad_flow(model.named_parameters())
                break

        # STEP 5: visualizing the whole network with TensorBoard
        if DEBUG_STEPS[4]:
            print("\n\n********* DEBUGGING STEP 5 *********\n    -> visualize the network using TensorBoard\n" +
                  "    -> type 'tensorboard --logdir=runs' into the command line and navigate to " +
                  "https://localhost:6006 to inspect model\n")

            # import and initialize TensorBoard
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter('runs/debug')

            # add model to TensorBoard
            dataiter = iter(loader_train)
            viz_phi, *_ = dataiter.next()
            viz_phi.requires_grad = False
            viz_data = torch.zeros(viz_phi.size(0), 2, viz_phi.size(-1))
            for i in range(viz_phi.size(-1)):
                # project current position on the upper half of the unit circle
                x_circle = np.cos(((i + 1) / viz_phi.size(-1)) * np.pi)
                y_circle = np.sin(((i + 1) / viz_phi.size(-1)) * np.pi)

                # fill the input tensor
                viz_data[:, 0, i] = x_circle
                viz_data[:, 1, i] = y_circle
            writer.add_graph(model, (viz_data, viz_phi))
            writer.close()

        # STEP 6: gradient checking
        if DEBUG_STEPS[5]:
            print("\n\n********* DEBUGGING STEP 6 *********\n    -> perform gradient checking\n")

            # check the gradient of the convolutional oligo kernel layer
            print("gradient checking for convolutional oligo kernel layer...")
            in_tuple = (torch.randn(4, 2, ref_pos.size(1), dtype=torch.double, requires_grad=True),
                        torch.randn(4, len(kmer_dict), ref_pos.size(1), dtype=torch.double, requires_grad=False))
            test = gradcheck(model.oligo, in_tuple, eps=1e-6, atol=1e-4)
            print("Result: {}\n".format(test))

            # check the gradient of the loss function
            print("gradient checking for the loss function...")
            in_tuple = (torch.randn(4, 3, dtype=torch.double, requires_grad=True),
                        torch.randn(4, 3, dtype=torch.double, requires_grad=False).argmax(1))
            test = gradcheck(criterion, in_tuple, eps=1e-6, atol=1e-4)
            print("Result: {}\n".format(test))

        if DEBUG_STEPS[6]:
            print("\n\n********* DEBUGGING STEP 7 *********\n    -> visualize the oligo kernel embedding\n")

            import matplotlib.pyplot as plt

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

            # visualize embeddings as heatmaps
            fig, axs = plt.subplots(len(kernel_embeddings[0]), len(kernel_embeddings))
            for i in range(len(kernel_embeddings)):
                for j in range(len(kernel_embeddings[0])):
                    if len(kernel_embeddings[i]) != len(kernel_embeddings[0]):
                        continue
                    im = axs[j, i].imshow(kernel_embeddings[i][j], cmap='hot', interpolation=None, aspect='auto')
                    axs[j, i].set_title("target = %s" % str(labels[i][j]))
                    axs[j, i].set(xlabel='position', ylabel='anchor point')
                    axs[j, i].label_outer()
                    fig.colorbar(im, ax=axs[j, i])
            plt.show()

        return

    # DEBUGGING END

    # train model
    acc, loss = model.sup_train(loader_train, criterion, optimizer, lr_scheduler, val_loader=loader_val,
                                epochs=args.nb_epochs)

    # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
    # model
    torch.save({'args': args, 'state_dict': model.state_dict()},
               args.outdir + "/CON_k" + str(args.kmer_size) + "_epochs" + str(args.nb_epochs) + ".pkl")

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
                args.outdir + "/CON_k" + str(args.kmer_size) + "_epochs" + str(args.nb_epochs) + ".pkl")

    try:
        # try to import pyplot
        import matplotlib.pyplot as plt

        # visualize embeddings as heatmaps
        fig, axs = plt.subplots(len(kernel_embeddings[0]), len(kernel_embeddings))
        for i in range(len(kernel_embeddings)):
            for j in range(len(kernel_embeddings[0])):
                if len(kernel_embeddings[i]) != len(kernel_embeddings[0]):
                    continue
                im = axs[j, i].imshow(kernel_embeddings[i][j], cmap='hot', interpolation=None, aspect='auto')
                axs[j, i].set_title("%s" % str(labels[i][j]))
                axs[j, i].set(xlabel='position', ylabel='anchor point')
                axs[j, i].label_outer()
                fig.colorbar(im, ax=axs[j, i])
        #plt.show()
        plt.savefig(args.outdir + "/kernel_embedding.png")

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
        #plt.show()
        plt.savefig(args.outdir + "/acc_loss.png")

        # show the position of the anchor points as a histogram
        anchor = (torch.acos(model.oligo.weight[:, 0]) / np.pi) * (ref_pos.size(1) - 1)
        anchor = anchor.detach().numpy()
        fig3 = plt.figure()
        plt.hist(anchor, bins=20)
        plt.xlabel('Position')
        plt.ylabel('# Anchor Points')
        plt.title('Distribution of anchor points')
        #plt.show()
        plt.savefig(args.outdir + "/anchor_positions.png")

    except:
        print("Cannot import matplotlib.pyplot")


def count_classes(filepath, verbose=False, drug=7):
    class_count = [0, 0, 0]
    nb_samples = 0

    with open(filepath, 'rU') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            aux_lab = record.id.split('|')[drug]
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
        print("class balance: H = {}, M = {}, L = {}".format(class_count[0] / nb_samples, class_count[1] / nb_samples,
                                                             class_count[2] / nb_samples))

    expected_loss = -(class_count[0] / nb_samples) * np.log(0.33) - \
                    (class_count[1] / nb_samples) * np.log(0.33) -\
                    (class_count[2] / nb_samples) * np.log(0.33)

    if verbose:
        print("expected loss: {}".format(expected_loss))

    rand_guess = (class_count[0] / nb_samples) * 0.33 + (class_count[1] / nb_samples) * 0.33 + \
                 (class_count[2] / nb_samples) * 0.33

    if verbose:
        print("expected accuracy with random guessing: {}".format(rand_guess))

    return class_count, expected_loss


def test_dataset_handler(filepath, drug, nb_classes):
    # load dataset
    dataset = CustomHandler(filepath, drug=drug, nb_classes=nb_classes)

    # print dataset statistics
    count_classes(filepath, True, dataset.drug_nb)

    # check if the number of samples in the dataset is correct
    print('\nNumber of samples in dataset handler: {}'.format(len(dataset)))

    # interate over dataset and look at labels
    loader = DataLoader(dataset, 50)
    for phi, label, *_ in loader:
        print(label)
        break


def test_motiv(kmer_len, anchor, sigma):
    path = '/home/jonas/Documents/Research/OligoKernelNetwork/cluster_results/' +\
           'kmer_{}/params_{}.0_10/anchors_{}/layers_3/'.format(kmer_len, sigma, anchor)
    params = torch.load(path + 'CON_results__epochs500.pkl')
    args = params['args']
    state_dict = params['state_dict']

    # create dictionary that maps kmers to index
    kmer_dict = kmer2dict(args.kmer_size, args.alphabet)

    # build tensor holding reference positions
    ref_pos = build_kmer_ref('data/processed_PI_DataSet_sample_labels_clean.fasta', args.extension, kmer_dict,
                             args.kmer_size)

    # convert anchor points into sequence positions
    anchor = (torch.acos(state_dict['oligo.weight'][:, 0]) / np.pi) * (ref_pos.size(1) - 1)
    anchor = anchor.detach().numpy()

    # create motivs from the anchors
    anchors_to_motivs(anchor, ref_pos, kmer_dict, args.kmer_size, outdir=path+'motivs/')


def eval_results():
    # define macros
    KMER_SIZES = [1, 2]
    ANCHORS = [40, 65, 90]
    SIGMAS = [2, 4, 8, 16]
    DATAFILE = '/home/jonas/Development/CON/con/data/processed_PI_DataSet_sample_labels_clean.fasta'
    RESULTSDIR = '/home/jonas/Documents/Research/OligoKernelNetwork/cluster_results/'

    # iterate over all results and evaluate each of them
    for kmer_size in KMER_SIZES:
        for anchor in ANCHORS:
            for sigma in SIGMAS:
                print('Evaluating following parameter combination:\n    kmer_size={}, num_anchors={}, sigma={}'.format(
                    kmer_size, sigma, anchor))

                # load the results
                print('loading dataset... please hold...')
                results_dict = torch.load(RESULTSDIR +
                    'kmer_{}/params_{}.0_10/anchors_{}/layers_3/CON_results__epochs500.pkl'.format(kmer_size, sigma,
                                                                                                   anchor))
                args = results_dict['args']

                # access dataset
                data_all = CustomHandler(DATAFILE, kmer_size)
                loader = DataLoader(data_all, batch_size=args.batch_size)

                # initialize the model
                kmer_dict = kmer2dict(args.kmer_size, args.alphabet)
                ref_pos = build_kmer_ref(DATAFILE, args.extension, kmer_dict, args.kmer_size)
                model = CON2(out_channels_list=args.out_channels, ref_kmerPos=ref_pos, filter_sizes=args.kernel_sizes,
                             strides=args.strides, paddings=args.paddings, num_classes=args.num_classes,
                             kernel_args=[args.sigma, args.scale])

                # load the trained model state dictionary
                model.load_state_dict(results_dict['state_dict'])

                # store true label and predictions for each sample in the dataset
                print('calculating predictions of trained model... please hold...')
                pred_y, true_y = model.predict(loader, proba=True)

                # convert to numpy arrays
                pred_y = pred_y.detach().numpy()
                true_y = true_y.detach().numpy()
                true_y = np.nonzero(true_y)[1]

                print('\n==================================================\n' +
                      'Statistics of params combination [{}, {}, {}]\n'.format(kmer_size, sigma, anchor) +
                      '==================================================')

                # calculate the AU1P score
                aunp_score = roc_auc_score(true_y, pred_y, average="weighted", multi_class="ovr")
                print('\n    - AUNP score: {}'.format(aunp_score))

                # convert predicted probability into label
                pred_y_label = np.argmax(pred_y, axis=1)

                # calculate classification accuracy of each class separately
                print('\n    - Classification Accuracy:')
                class_names = ["H", "M", "L"]
                for i in range(args.num_classes):
                    # get indices of all samples belonging to the current class
                    class_indices = [j for j, e in enumerate(true_y) if e == i]

                    # calculate accuracy for current class
                    class_acc = sum(pred_y_label[class_indices] == i) / len(class_indices)

                    # print result
                    if i == 1:
                        print(pred_y_label[class_indices])
                    print('        * Classification accuracy of class {}: {}'.format(class_names[i], class_acc))


def main(filepath):
    print('main')


if __name__ == '__main__':
    #main('./data/test_dataset.fasta')
    count_classes('./data/processed_PI_DataSet_sample_labels.fasta', True, 14)
    #test_motiv(1, 65, 4)
    #test_exp()
    #test_dataset_handler('./data/processed_PI_DataSet_sample_labels.fasta', 'SQV', 2)
    #eval_results()
