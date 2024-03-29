import argparse
import pickle
import random
import time

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold


def load_args():
    parser = argparse.ArgumentParser(
        description="Script to create synthetic DNA for prediction"
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=42,
        type=int,
        help="Set the random seed for the creation of the dataset.",
    )
    parser.add_argument(
        "--type",
        dest="type",
        default="data",
        type=str,
        choices=["data", "folds"],
        help="Select which function should be called. 'data' creates a synthetic dataset and 'folds' creates stratified folds for an existing dataset",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        default="./",
        type=str,
        help="Specify the location where the synthetic dataset will be stored.",
    )
    parser.add_argument(
        "--alphabet",
        dest="alphabet",
        default="ACGT",
        help="Alphabet used to create sequences.",
    )
    parser.add_argument(
        "--nbsamples",
        dest="nbsamples",
        default=500,
        type=int,
        help="Number of samples per class.",
    )
    parser.add_argument(
        "--seqlen",
        dest="seqlen",
        default=100,
        type=int,
        help="Length of the sequences within the synthetic dataset",
    )
    parser.add_argument(
        "--uncertainty-pos",
        dest="uncertainty_pos",
        default=5,
        type=int,
        help="The amount of sequence position that the occurance of the motifs can vary",
    )
    parser.add_argument(
        "--uncertainty-comp",
        dest="uncertainty_comp",
        default=2,
        type=int,
        help="Number of motif positions with compositional uncertainty.",
    )
    parser.add_argument(
        "--positive-pos",
        dest="positive_pos",
        default=80,
        type=int,
        help="Position where the positive motif is located around",
    )
    parser.add_argument(
        "--positive-motif",
        dest="positive_motif",
        default="AAGTC",
        type=str,
        help="Motif embedded into positive sequences.",
    )
    parser.add_argument(
        "--negative-pos",
        dest="negative_pos",
        default=20,
        type=int,
        help="Position where the negative motif is located around",
    )
    parser.add_argument(
        "--negative-motif",
        dest="negative_motif",
        default="TGAGT",
        type=str,
        help="Motif embedded into negative sequences.",
    )
    parser.add_argument(
        "--dataset",
        dest="path_to_dataset",
        default="default.fasta",
        type=str,
        help="Dataset for which the folds should be created. Currently, the function is implemented for fasta files. Only used if --type is set to 'folds'.",
    )
    parser.add_argument(
        "--kfold",
        dest="kfold",
        default=5,
        type=int,
        help="Number of stratified folds created. Only used if --type is set to 'folds'.",
    )

    args = parser.parse_args()

    # make sure that the number of motif positions with variability does not exceed
    # the length of motifs
    if args.uncertainty_comp > len(args.positive_motif) or args.uncertainty_comp > len(
        args.negative_motif
    ):
        raise ValueError(
            "Number of uncertain motif positions exceed the length of the motifs."
        )

    # set the random seed
    random.seed(args.seed)

    # define the possible positive motifs
    args.motifs_pos = [args.positive_motif]
    args.positions_pos = random.sample(
        range(len(args.positive_motif)), args.uncertainty_comp
    )
    for pos in args.positions_pos:
        replacement = random.choice(
            args.positive_motif.replace(args.positive_motif[pos], "")
        )
        aux_motif = (
            args.positive_motif[:pos] + replacement + args.positive_motif[pos + 1 :]
        )
        args.motifs_pos.append(aux_motif)

    # define the possible negative motifs
    args.motifs_neg = [args.negative_motif]
    args.positions_neg = random.sample(
        range(len(args.positive_motif)), args.uncertainty_comp
    )
    for pos in args.positions_neg:
        replacement = random.choice(
            args.negative_motif.replace(args.negative_motif[pos], "")
        )
        aux_motif = (
            args.negative_motif[:pos] + replacement + args.negative_motif[pos + 1 :]
        )
        args.motifs_neg.append(aux_motif)

    return args


def get_random_str(length, alphabet):
    """Function to return a random string of a defined length over a specified alphabet.
    """
    return "".join(random.choice(alphabet) for i in range(length))


def create_synthetic_data(args):
    """Function to create a synthetic dataset containing a positive and negative motif
    at specified locations with a defined degree of positional and compositional variability.
    """
    # create identifier for the current generated dataset
    data_id = int(time.time())

    # open the file that stores the synthetic dataset
    with open(f"{args.outdir}{data_id}_syntheticData.fasta", "w") as out_file:
        # create the positive samples
        for i in range(args.nbsamples):

            # create random sequence of specified length
            seq_pos = get_random_str(args.seqlen, args.alphabet)
            seq_neg = get_random_str(args.seqlen, args.alphabet)

            # determine position and compsition of the positive motif
            pos_pos = args.positive_pos + random.choice(
                list(range(args.uncertainty_pos * -1, args.uncertainty_pos + 1, 1))
            )
            pos_motif = random.choice(args.motifs_pos)

            # insert the positive motif
            seq_pos = (
                seq_pos[:pos_pos] + pos_motif + seq_pos[pos_pos + len(pos_motif) :]
            )

            # determine position and composition of the negative motif
            neg_pos = args.negative_pos + random.choice(
                list(range(args.uncertainty_pos * -1, args.uncertainty_pos + 1, 1))
            )
            neg_motif = random.choice(args.motifs_neg)

            # insert the motif
            seq_neg = (
                seq_neg[:neg_pos] + neg_motif + seq_neg[neg_pos + len(neg_motif) :]
            )

            # generate id string
            out_file.write(f">{i*2}_1\n{seq_pos}\n>{i*2 + 1}_0\n{seq_neg}\n")

    # store the arguments used in creating this dataset
    with open(f"{args.outdir}{data_id}_arguments.pkl", "wb") as out_file:
        pickle.dump(args, out_file)


def create_folds(args):
    """Function to create stratified cross-validation folds for an existing dataset. This is
    needed to train different methods on the dataset using the exact same folds.
    """

    # initalize sklearn's StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # parse the dataset
    tmp = list(SeqIO.parse(args.path_to_dataset, "fasta"))
    # get the sequences and labels
    seq = [i.seq for i in tmp]
    label = [int(i.id.split("_")[-1]) for i in tmp]

    # create np.array objects for the creation of stratified fold indices
    vec_indices = np.arange(len(seq))
    vec_label = np.array(label)

    # get train and validation indices for each fold
    aux_folds = []
    for _, split_idx in enumerate(skf.split(vec_indices, vec_label)):
        aux_folds.append((vec_indices[split_idx[0]], vec_indices[split_idx[1]]))

    # create the name of the folds file
    filename = args.path_to_dataset.split("/")[-1]
    filename = filename.split(".")[0]
    filename = filename + "_folds.pkl"

    # store the folds
    with open(args.outdir + filename, "wb") as out_file:
        pickle.dump(aux_folds, out_file, pickle.HIGHEST_PROTOCOL)


def main():
    args = load_args()

    if args.type == "data":
        create_synthetic_data(args)
    elif args.type == "folds":
        create_folds(args)


if __name__ == "__main__":
    main()
