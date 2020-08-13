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
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
import numpy as np

from con import CON, CONDataset, kmer2dict, build_kmer_ref, compute_metrics

from timeit import default_timer as timer
from graphviz import Digraph
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# MACROS
DEBUGGING = False


name = 'example_experiment'
data_dir = './data/'


#######################BAD GRAPH VISUALIZATION ROUTINE START#########################

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def register_hooks(var):
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


#######################BAD GRAPH VISUALIZATION ROUTINE END#########################


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
    filepath = './data/test_dataset.fasta'
    extension = 'fasta'
    kmer_size = 3
    alphabet = 'ARNDCQEGHILKMFPSTWYVXBZJUO'

    # create dictionary that maps kmers to index
    kmer_dict = kmer2dict(kmer_size, alphabet)

    # build tensor holding reference positions
    ref_pos = build_kmer_ref(filepath, extension, kmer_dict, kmer_size)

    # initialize con model
    model = CON([40, 128], ref_pos, [3], [1, 3], num_classes=3, kernel_funcs=['exp', 'exp_chen'],
                kernel_args_list=[[0.5, 1], [0.5]], kernel_args_trainable=[False, False])
    #model = CON([40], ref_pos, [], [1], num_classes=3, kernel_funcs=['exp'],
    #            kernel_args_list=[[0.5, 1]], kernel_args_trainable=[False])

    # load data
    data = CustomHandler(filepath)

    # Creating data indices for training and validation splits:
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PyTorch data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    loader_train = DataLoader(data, batch_size=4, sampler=train_sampler)
    loader_val = DataLoader(data, batch_size=4, sampler=val_sampler)

    # initialize optimizer and loss function
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.con_model.parameters(), lr=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

    # DEBUGGING START
    if DEBUGGING:
        # iterate over all parameter
        print(model)

        # register forward and backward hooks
        #hookF = [Hook(list(list(model._modules.items())[0][1])[0])]
        #hookF.append(Hook(list(list(model._modules.items())[0][1])[1]))
        #hookF.append(Hook(list(model._modules.items())[1][1]))
        #hookF.append(Hook(list(model._modules.items())[2][1]))
        #hookB = [Hook(list(list(model._modules.items())[0][1])[0], backward=True)]
        #hookB.append(Hook(list(list(model._modules.items())[0][1])[1], backward=True))
        #hookB.append(Hook(list(model._modules.items())[1][1], backward=True))
        #hookB.append(Hook(list(model._modules.items())[2][1], backward=True))

        for data, target, *_ in loader_train:
            # perform one forward step
            out = model(data)

            # register hooks on the out tensor
            #get_dot = register_hooks(out)

            # get the dot and save it
            #dot = get_dot()
            #dot.save('tmp.dot')

            print(target.shape[1])

            # backprop to get backward hooks
            #out.backward(target, retain_graph=True)
            loss = criterion(out, target.argmax(1))
            loss.backward()
            plot_grad_flow(model.named_parameters())

            # print hooks
            #print()
            #print('***' * 4 + '  Forward Hooks Inputs & Outputs  ' + '***' * 4 + '\n')
            #for hook in hookF:
            #    try:
            #        print([x.shape for x in hook.input])#(hook.input)#.shape)
            #    except:
            #        print(hook.input[0].shape, hook.input[1:])
            #    print(hook.output.shape)#.shape)
            #    print('\n' + '---' * 27 + '\n')
            #print('\n')
            #print('***' * 4 + '  Backward Hooks Inputs & Outputs  ' + '***' * 4 + '\n')
            #for hook in hookB:
            #    print([x.shape for x in hook.input])#(hook.input)#.shape)
            #    print([x.shape for x in hook.output])#(hook.output)#.shape)
            #    print('\n' + '---' * 27 + '\n')

            #break

        return

    # DEBUGGING END

    # train model
    model.sup_train(loader_train, criterion, optimizer, lr_scheduler, val_loader=loader_val, epochs=5)

    # iterate through dataset
    #for i_batch, sample_batch in enumerate(loader):
    #    print('shape of batch {}: {}'.format(i_batch, sample_batch[0].shape))
    #    print('shape of target {}: {}'.format(i_batch, sample_batch[1].shape))
    #    print('')

    # perform "testing" using the validation data point (only for debugging purpose)
    y_pred, y_true = model.predict(loader_val, proba=True)
    scores = sum(y_pred == y_true) / len(y_pred)

    # compute_metrics might only work for binary classification
    #scores = compute_metrics(y_pred, y_true)

    print(scores)


def main(filepath):
    print('main')


if __name__ == '__main__':
    #main('./data/test_dataset.fasta')
    test_exp()
