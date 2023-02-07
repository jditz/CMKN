import argparse

import matplotlib.pyplot as plt
import numpy as np

from cmkn import get_weight_distribution


def load_args():
    parser = argparse.ArgumentParser(description="Script to perform CMKN analysis")
    parser.add_argument(
        "--data",
        dest="data",
        default="NFV",
        type=str,
        help="Select the dataset for which the analysis should be performed",
    )
    parser.add_argument(
        "--run",
        dest="run",
        required=True,
        help="Select the run for which the analysis should be performed",
    )
    parser.add_argument(
        "--type",
        dest="type",
        default="robustness_pos",
        choice=["robustness_pos"],
        help="Select the analysis that should be performed",
    )
    parser.add_argument(
        "--positions",
        dest="pos",
        nargs="+",
        type=int,
        help="Position for which the analyis should be performed",
    )
    parser.add_argument(
        "--modelargs",
        required=True,
        nargs="+",
        type=str,
        help="Parameters of the model that will be used for the analysis.",
    )
    parser.add_argument(
        "--folds",
        dest="folds",
        type=int,
        default=5,
        help="Number of folds used for model training.",
    )
    parser.add_argument(
        "--classes",
        dest="classes",
        type=int,
        default=2,
        help="Number of classes used in the experiment.",
    )
    parser.add_argument(
        "--layers",
        dest="layers",
        nargs="+",
        type=str,
        default=["classifier", "fc"],
        help="List with the names of the model's layers",
    )

    args = parser.parse_args()

    # automatically assign other values dependent on user choices
    if args.data in ["ATV", "DRV", "FPV", "IDV", "LPV", "NFV", "SQV", "TPV"]:
        args.data_type = "HIV"
        args.drug_type = "PI"
        args.seq_len = 99
    elif args.data in ["3TC", "ABC", "AZT", "D4T", "DDI", "TDF"]:
        args.data_type = "HIV"
        args.drug_type = "NRTI"
        args.seq_len = 240
    elif args.data in ["EFV", "NVP", "ETR", "RPV"]:
        args.data_type = "HIV"
        args.drug_type = "NNRTI"
        args.seq_len = 240
    elif args.data == "NN269_acceptor":
        args.data = args.data.split("_")[1]
        args.data_type = "NN269"
        args.seq_len = 90
    elif args.data == "NN269_donor":
        args.data = args.data.split("_")[1]
        args.data_type = "NN269"
        args.seq_len = 15
    elif args.data == "DGSPLICER_acceptor":
        args.data = args.data.split("_")[1]
        args.data_type = "DGSPLICER"
        args.seq_len = 36
    elif args.data == "DGSPLICER_donor":
        args.data = args.data.split("_")[1]
        args.data_type = "DGSPLICER"
        args.seq_len = 18
    else:
        raise ValueError(f"Unknown dataset selected: {args.data}")

    # the models that will be analyzed with this script had seven parameters
    #   -> ATTENTION: change the following lines if your models (i.e. file path) are different
    if len(args.modelargs) != 7:
        raise ValueError(f"Unsupported number of model parameters: {args.modelargs}")

    # set up the path to the stored results
    #   -> ATTENTION: Please change the path appropriately on your system
    args.filepath = (
        f"/home/jonas/Documents/Research/MotifKernelNetwork/cluster_results/"
        f"{args.data_type}/{args.run}/{args.data}/"
        f"classes_{args.modelargs[0]}/kmer_{args.modelargs[1]}/params_{args.modelargs[2]}_"
        f"{args.modelargs[3]}_{args.modelargs[4]}/anchors_{args.modelargs[5]}/layers_{args.modelargs[6]}/"
    )

    # store the file name
    #   -> ATTENTION: Please change the filename appropriatly
    args.filename = "CON_result_epochs200_fold{}.pkl"

    return args


def robustness_pos(args):
    """Function to analyize the robustness of the positional interpretability.
    """

    # iterate over all folds and store the weights
    weights = []
    for i in range(args.folds):
        _, aux_weights = get_weight_distribution(
            model_path=args.filepath + args.filename.format(i),
            num_classes=args.classes,
            seq_len=args.seq_len,
            layers=args.layers,
        )
        weights.append(aux_weights)

    # plot derivation of weights over folds as boxplot
    plt.figure()
    for i, pos in enumerate(args.pos):
        weights_positive = []
        weights_negative = []
        for j in range(args.folds):
            weights_positive.append(aux_weights[j][1, pos])
            weights_negative.append(aux_weights[j][0, pos])

        plt.boxplot(
            np.array(weights_positive),
            positions=2 * i - 0.4,
            sym="",
            widths=0.6,
            c="r",
        )

        plt.boxplot(
            np.array(weights_negative),
            positions=2 * i + 0.4,
            sym="",
            widths=0.6,
            c="b",
        )

    plt.show()


def main():
    """Main function of the analysis script.
    """

    # read in command line arguments
    args = load_args()

    # perfom the requested analysis
    if args.type == "robustness_pos":
        robustness_pos(args)


if __name__ == "__main__":
    main()
