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
from torch import nn, autograd
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from torch.autograd import gradcheck
import torch.optim as optim
import numpy as np

from con import (CON, CON2, CONDataset, ClassBalanceLoss, kmer2dict, build_kmer_ref, category_from_output,
                 compute_metrics, plot_grad_flow, Hook)# register_hooks

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from Bio import SeqIO


# MACROS

# set to True to enable network debugging
DEBUGGING = False

# each Boolean value decides if the corresponding debugging step will be performed
DEBUG_STEPS = [True, True, False, False, False, False, True]


name = 'example_experiment'
data_dir = './data/'


# extend custom data handler for the used dataset
class CustomHandler(CONDataset):
    def __init__(self, filepath):
        super(CustomHandler, self).__init__(filepath, alphabet='PROTEIN_AMBI')

        self.all_categories = ['H', 'M', 'L']

    def __getitem__(self, idx):
        # get a sample using the parent __getitem__ function
        sample = super(CustomHandler, self).__getitem__(idx)

        # initialize label tensor
        if len(sample[1]) == 1:
            labels = torch.zeros(3)
        else:
            labels = torch.zeros(len(sample[1]), 3)

        # iterate through all id strings and update the label tensor, accordingly
        for i, id_str in enumerate(sample[1]):
            try:
                aux_lab = id_str.split('|')[7]
                if aux_lab != 'NA':
                    if len(sample[1]) == 1:
                        labels[self.all_categories.index(aux_lab)] = 1.0
                    else:
                        labels[i, self.all_categories.index(aux_lab)] = 1.0

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
    parser.add_argument('--batch-size', type=int, default=128, metavar='M',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument("--num-anchors", dest="n_anchors", metavar="m", default=[64], nargs='+', type=int,
                        help="number of anchor points for each layer (default [64])")
    parser.add_argument("--subsamplings", dest="subsamplings", metavar="s", default=[1], nargs='+', type=int,
                        help="subsampling for each layer (default: [1])")
    parser.add_argument("--sigma", dest="kernel_params", default=[0.3], nargs='+', type=float,
                        help="sigma for each layer (default: [0.3])")
    parser.add_argument("--scale", dest="scale_param", default=100, type=int,
                        help="scaling parameter for the convolutional kernel layer (Default: 100)")
    parser.add_argument("--sampling-patches", dest="n_sampling_patches", default=250000, type=int,
                        help="number of sampled patches (default: 250000)")
    parser.add_argument("--kfold", dest="kfold", default=5, type=int, help="k-fold cross validation (default: 5)")
    parser.add_argument("--ntrials", dest="ntrials", default=1, type=int,
                        help="number of trials for training (default: 1)")
    parser.add_argument("--penalty", metavar="penal", dest="penalty", default='l2', type=str, choices=['l2', 'l1'],
                        help="regularization used in the last layer (default: l2)")
    parser.add_argument("--outdir", metavar="outdir", dest="outdir", default='output', type=str,
                        help="output path(default: '')")
    parser.add_argument("--regularization", type=float, default=1e-6, help="regularization parameter for CON model")
    parser.add_argument("--preprocessor", type=str, default='standard_row', choices=['standard_row', 'standard_col'],
                        help="preprocessor for CON (default: standard_row)")
    parser.add_argument("--use-cuda", action='store_true', default=False, help="use gpu (default: False)")
    parser.add_argument("--pooling", default='sum', choices=['mean', 'max', 'sum'], type=str,
                        help='specifies global pooling layer (default: sum)')
    parser.add_argument("--noise", type=float, default=0.0, help="perturbation percent")

    # parse the arguments
    args = parser.parse_args()

    # GPU will only be used if available
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    # 'num-anchors', 'subsampling', and 'sigma' have to be of same length
    if not len(args.n_anchors) == len(args.subsampling) == len(args.kernel_params):
        raise ValueError('Mismatching lengths!\nVectors given for the arguments --num-anchors, --subsampling, and '
                         '--sigma have to be of same length')

    # number of layers is equal to length of the 'num-anchors' vector
    args.n_layer = len(args.n_anchors)

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
        outdir = outdir + "/{}".format(name)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/noise_{}".format(args.noise)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/{}".format(args.pooling)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/{}_{}_{}'.format(
            args.n_layers, args.n_anchors, args.subsamplings)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/{}'.format(args.kernel_params)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        args.outdir = outdir

    return args


def test_exp():
    # set parameters for the test run
    #filepath = './data/processed_PI_DataSet_sample_labels_clean.fasta'
    filepath = './data/test_dataset.fasta'
    class_count, expected_loss = count_classes(filepath)
    extension = 'fasta'
    kmer_size = 3
    alphabet = 'ARNDCQEGHILKMFPSTWYVXBZJUO'
    nb_epochs = 10

    # create dictionary that maps kmers to index
    kmer_dict = kmer2dict(kmer_size, alphabet)

    # build tensor holding reference positions
    ref_pos = build_kmer_ref(filepath, extension, kmer_dict, kmer_size)

    # initialize con model
    #model = CON([40, 128], ref_pos, [3], [1, 3], num_classes=3, kernel_funcs=['exp', 'exp_chen'],
    #            kernel_args_list=[[1.25, 1], [0.5]], kernel_args_trainable=[False, False])
    #model = CON([40], ref_pos, [], [1], num_classes=3, kernel_funcs=['exp'],
    #            kernel_args_list=[[0.5, 1]], kernel_args_trainable=[False])
    #model = CON2(out_channels_list=[40, 32, 64, 128, 512, 1024], ref_kmerPos=ref_pos, filter_sizes=[3, 5, 5, 5, 10],
    #             strides=[1, 1, 1, 1, 1, 1], paddings=['SAME', 'SAME', 'SAME', 'SAME', 'SAME'], num_classes=3,
    #             kernel_args=[1.25, 1])
    model = CON2(out_channels_list=[40], ref_kmerPos=ref_pos, filter_sizes=[], strides=[1], paddings=[], num_classes=3,
                 kernel_args=[1.25, 1])

    # load data
    data_all = CustomHandler(filepath)

    # Creating data indices for training and validation splits:
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(data_all)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PyTorch data samplers and loaders:
    data_train = Subset(data_all, train_indices)
    data_val = Subset(data_all, val_indices)

    loader_train = DataLoader(data_train, batch_size=4)
    loader_val = DataLoader(data_val, batch_size=4)

    # initialize optimizer and loss function
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = ClassBalanceLoss(class_count, len(class_count), 'cross_entropy', 0.99, 1.0)
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
            print("\ninitial loss: {}; expected loss: {}".format(init_loss, expected_loss))

        # STEP 3: train model on a single data point and see if it can overfit
        if DEBUG_STEPS[2]:
            print("\n\n********* DEBUGGING STEP 3 *********\n    -> train model on a single datapoint\n" +
                  "    -> model has to be able to overfit with zero loss and validation accuracy at chance level\n")
            data_debug = Subset(data_all, [val_indices[0]])
            loader_debug = DataLoader(data_debug, batch_size=1)

            print("length of debug dataset: {}".format(len(loader_debug.dataset)))
            print("length of train dataset: {}".format(len(loader_train.dataset)))
            print("length of val dataset: {}\n".format(len(loader_val.dataset)))

            model.sup_train(loader_debug, criterion, optimizer, lr_scheduler, val_loader=loader_train, epochs=3)

        # STEP 4: visualize the gradient flow
        if DEBUG_STEPS[3]:
            print("\n\n********* DEBUGGING STEP 4 *********\n    -> visualize the gradient flow through the network\n")
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
            in_tuple = (torch.randn(4, 2, ref_pos.size(1), dtype=torch.double, requires_grad=False),
                        torch.randn(4, len(kmer_dict), ref_pos.size(1), dtype=torch.double, requires_grad=False))
            test = gradcheck(model.con_model[0], in_tuple, eps=1e-6, atol=1e-4)
            print("Result: {}\n".format(test))

            # check the gradient of the loss function
            print("gradient checking for the loss function...")
            in_tuple = (torch.randn(4, 3, dtype=torch.double, requires_grad=True),
                        torch.randn(4, 3, dtype=torch.double, requires_grad=False).argmax(1))
            test = gradcheck(criterion, in_tuple, eps=1e-6, atol=1e-4)
            print("Result: {}\n".format(test))

        if DEBUG_STEPS[6]:
            print("\n\n********* DEBUGGING STEP 7 *********\n    -> visualize the oligo kernel embedding\n")

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
            fig, axs = plt.subplots(len(kernel_embeddings), len(kernel_embeddings[0]))
            for i in range(len(kernel_embeddings)):
                for j in range(len(kernel_embeddings[0])):
                    im = axs[i, j].imshow(kernel_embeddings[i][j], cmap='hot', interpolation=None, aspect='auto')
                    axs[i, j].set_title("target = %s" % str(labels[i][j]))
                    axs[i, j].set(xlabel='position', ylabel='anchor point')
                    axs[i, j].label_outer()
                    fig.colorbar(im, ax=axs[i, j])
            plt.show()

        # register forward and backward hooks
        # hookF = [Hook(list(list(model._modules.items())[0][1])[0])]
        # hookF.append(Hook(list(list(model._modules.items())[0][1])[1]))
        # hookF.append(Hook(list(model._modules.items())[1][1]))
        # hookF.append(Hook(list(model._modules.items())[2][1]))
        # hookB = [Hook(list(list(model._modules.items())[0][1])[0], backward=True)]
        # hookB.append(Hook(list(list(model._modules.items())[0][1])[1], backward=True))
        # hookB.append(Hook(list(model._modules.items())[1][1], backward=True))
        # hookB.append(Hook(list(model._modules.items())[2][1], backward=True))

        #running_loss = 0.0
        #running_corrects = 0
        #aux_data = None
        #for data, target, *_ in loader_train:
            # with autograd.detect_anomaly():
            #     # perform one forward step
            #     out = model(data)
            #
            #     # register hooks on the out tensor
            #     #get_dot = register_hooks(out)
            #
            #     # get the dot and save it
            #     #dot = get_dot()
            #     #dot.save('tmp.dot')
            #
            #     #print(target.shape[1])
            #
            #     # backprop to get backward hooks
            #     #out.backward(target, retain_graph=True)
            #     loss = criterion(out, target.argmax(1))
            #     loss.backward()

            # simulate epochs for a single data point
            # for i in range(10):
            #     out = model(data)
            #
            #     pred = torch.zeros(out.shape)
            #     for j in range(out.shape[0]):
            #         pred[j, category_from_output(out[i, :])] = 1
            #
            #     loss = criterion(out, target.argmax(1))
            #     loss.backward()
            #     optimizer.step()
            #     model.normalize_()
            #     plot_grad_flow(model.named_parameters())
            #
            #     running_loss += loss.item() * data.size(0)
            #     running_corrects += torch.sum(torch.sum(pred == target.data, 1) ==
            #                                   torch.ones(pred.shape[0]) * pred.shape[1]).item()
            #     print(running_loss, running_corrects)
            #
            # return

            #out = model(data)
            #aux_data = (out, target)
            #pred = torch.zeros(out.shape)
            #for j in range(out.shape[0]):
            #    pred[j, category_from_output(out[j, :])] = 1

            #loss = criterion(out, target.argmax(1))

            #optimizer.zero_grad()
            #loss.backward()

            #aux = data.clone().detach().requires_grad_(True)
            #print("gradCheck: {}".format(gradcheck(model, (aux,))))

            #optimizer.step()
            #model.normalize_()
            #plot_grad_flow(model.named_parameters())

            #running_loss += loss.item() * data.size(0)
            #running_corrects += torch.sum(torch.sum(pred == target.data, 1) ==
            #                              torch.ones(pred.shape[0]) * pred.shape[1]).item()

            # print hooks
            # print()
            # print('***' * 4 + '  Forward Hooks Inputs & Outputs  ' + '***' * 4 + '\n')
            # for hook in hookF:
            #     try:
            #         print([x.shape for x in hook.input])#(hook.input)#.shape)
            #     except:
            #         print(hook.input[0].shape, hook.input[1:])
            #     print(hook.output.shape)#.shape)
            #     print('\n' + '---' * 27 + '\n')
            # print('\n')
            # print('***' * 4 + '  Backward Hooks Inputs & Outputs  ' + '***' * 4 + '\n')
            # for hook in hookB:
            #     print([x.shape for x in hook.input])#(hook.input)#.shape)
            #     print([x.shape for x in hook.output])#(hook.output)#.shape)
            #     print('\n' + '---' * 27 + '\n')
            #
            # # print the output of the CON layer as a heatmap
            # print('\nThe correct labels are: {}'.format(target))
            # items = [0, 1, 2, 3]
            # fig, axs = plt.subplots(2, 2)
            # for item, ax in zip(items, axs.ravel()):
            #     im = ax.imshow(hookF[0].output[item, :, :].detach().numpy(), cmap='hot', interpolation=None, aspect='auto')
            #     ax.set_title("target = %s" % str(target[item, :]))
            #     fig.colorbar(im, ax=ax)
            # plt.show()

        #def apply_fn(input, *params):
        #    return criterion(input, aux_data[1].argmax(1))
        #gradcheck(apply_fn, aux_data[0])

        #print("initial loss: {}".format(running_loss / len(loader_train.dataset)))
        #print("initial accuracy: {}".format(running_corrects / len(loader_train.dataset)))

        return

    # DEBUGGING END

    # train model
    model.sup_train(loader_train, criterion, optimizer, lr_scheduler, val_loader=loader_val, epochs=nb_epochs)

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
    fig, axs = plt.subplots(len(kernel_embeddings), len(kernel_embeddings[0]))
    for i in range(len(kernel_embeddings)):
        for j in range(len(kernel_embeddings[0])):
            im = axs[i, j].imshow(kernel_embeddings[i][j], cmap='hot', interpolation=None, aspect='auto')
            axs[i, j].set_title("target = %s" % str(labels[i][j]))
            fig.colorbar(im, ax=axs[i, j])
    plt.show()

    # save the model's state_dict to be able to perform inference and other stuff without the need of retraining the
    # model
    torch.save(model.state_dict(), "CON_k" + str(kmer_size) + "_epochs" + str(nb_epochs) + ".pt")

    # iterate through dataset
    #for i_batch, sample_batch in enumerate(loader):
    #    print('shape of batch {}: {}'.format(i_batch, sample_batch[0].shape))
    #    print('shape of target {}: {}'.format(i_batch, sample_batch[1].shape))
    #    print('')

    # perform "testing" using the validation data point (only for debugging purpose)
    #y_pred, y_true = model.predict(loader_val, proba=True)
    #scores = torch.sum(torch.sum(y_pred == y_true, 1) == torch.ones(y_pred.shape[0]) * y_true.shape[1]).item()

    # compute_metrics might only work for binary classification
    #scores = compute_metrics(y_pred, y_true)

    #print(scores)


def count_classes(filepath, verbose=False):
    class_count = [0, 0, 0]
    nb_samples = 0

    with open(filepath, 'rU') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            nb_samples += 1
            aux_lab = record.id.split('|')[7]
            if aux_lab == 'H':
                class_count[0] += 1
            elif aux_lab == 'M':
                class_count[1] += 1
            elif aux_lab == 'L':
                class_count[2] += 1

    if verbose:
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


def main(filepath):
    print('main')


if __name__ == '__main__':
    #main('./data/test_dataset.fasta')
    count_classes('./data/processed_PI_DataSet_sample_labels_clean.fasta', True)
    test_exp()
