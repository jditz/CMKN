import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cmkn import get_learned_motif, get_weight_distribution

# MACROS
COLORS = [
    "black",
    "blue",
    "yellow",
    "green",
    "red",
    "brown",
    "cyan",
    "darkblue",
    "darkorange",
    "limegreen",
    "steelblue",
    "darkgrey",
    "salmon",
    "deeppink",
    "lawngreen",
    "orange",
    "yellowgreen",
    "cornflowerblue",
    "darkkhaki",
    "darkred",
    "darkgreen",
    "turquoise",
    "darkviolet",
    "beige",
    "crimson",
    "silver",
    "gold",
    "forestgreen",
]


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
    elif args.data == "synthetic":
        args.data_type = "synthetic"
        args.seq_len = 100
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
    args.filename = "CMKN_results_fold{}.pkl"

    return args


def set_box_color(bp, colors, lw=2.5, facecolor=True):
    """Auxiliary function to set colors in a box plot
    """
    for i, c in enumerate(colors):
        if c == "darkorange" and facecolor:
            plt.setp(bp["boxes"][i], facecolor="orange", edgecolor=c, linewidth=lw)
        elif c == "darkblue" and facecolor:
            plt.setp(bp["boxes"][i], facecolor="steelblue", edgecolor=c, linewidth=lw)
        elif not facecolor:
            try:
                plt.setp(bp["boxes"][i], color=c, linewidth=lw)
            except AttributeError as err:
                print(f"AttributeError: {err}")
        plt.setp(bp["medians"][i], color="firebrick", linewidth=lw)
        plt.setp(bp["whiskers"][i * 2], color=c, linewidth=lw)
        plt.setp(bp["whiskers"][i * 2 + 1], color=c, linewidth=lw)
        plt.setp(bp["caps"][i * 2], color=c, linewidth=lw)
        plt.setp(bp["caps"][i * 2 + 1], color=c, linewidth=lw)


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

    # get the weights at requested positions
    weights_positive = []
    weights_negative = []
    for i, pos in enumerate(args.pos):
        weights_positive.append([])
        weights_negative.append([])
        for j in range(args.folds):
            weights_positive[i].append(weights[j][1, pos])
            weights_negative[i].append(weights[j][0, pos])

    # plot derivation of weights over folds as boxplot
    plt.figure()
    plt.title(args.data)

    bp_pos = plt.boxplot(
        [np.array(i) for i in weights_positive],
        positions=np.arange(len(args.pos)) * 2 - 0.4,
        sym="",
        widths=0.6,
        patch_artist=True,
    )
    set_box_color(bp_pos, ["darkorange"] * len(args.pos))

    bp_neg = plt.boxplot(
        [np.array(i) for i in weights_negative],
        positions=np.arange(len(args.pos)) * 2 + 0.4,
        sym="",
        widths=0.6,
        patch_artist=True,
    )
    set_box_color(bp_neg, ["darkblue"] * len(args.pos))

    plt.xticks(ticks=np.arange(len(args.pos)) * 2, labels=args.pos)
    plt.ylim((0, 0.06))

    plt.show()


def robustness_motif(args):
    """Function to analyze the robustness of motifs.
    """

    # get the learned motifs for each fold
    motifs = []
    alphabet_size = 0
    for i in range(args.folds):
        aux_motifs = get_learned_motif(
            model_path=args.filepath + args.filename.format(i),
            positions=args.pos,
            num_classes=args.classes,
            seq_len=args.seq_len,
            layers=args.layers,
            viz=False,
        )
        motifs.append(aux_motifs)
        alphabet_size = aux_motifs[0].shape[1]

    # rearrange the learned motif contribution for plotting
    motif_res = []
    motif_sus = []
    for i, _ in enumerate(args.pos):
        motif_res.append([])
        motif_sus.append([])
        for j in range(alphabet_size):
            motif_res.append([])
            motif_sus.append([])

            for k in range(args.folds):
                motif_res[-1].append(motifs[k][1][i, j].item())
                motif_sus[-1].append(motifs[k][0][i, j].item())

    # plot the results
    summary = False
    if summary:
        plt.figure()
        plt.title(args.data)
        bp_res = plt.boxplot(
            motif_res, flierprops=dict(markerfacecolor="k", marker="D", markersize=1),
        )
        set_box_color(
            bp_res,
            colors=COLORS[: alphabet_size + 1] * len(args.pos),
            lw=1,
            facecolor=False,
        )
        plt.xticks(
            ticks=[
                i * alphabet_size + (alphabet_size / 2) for i in range(len(args.pos))
            ],
            labels=args.pos,
        )
        plt.ylim((0, 0.1))
        plt.show()

        plt.figure()
        plt.title(args.data)
        bp_sus = plt.boxplot(
            motif_sus, flierprops=dict(markerfacecolor="k", marker="D", markersize=1),
        )
        set_box_color(
            bp_sus,
            colors=COLORS[: alphabet_size + 1] * len(args.pos),
            lw=1,
            facecolor=False,
        )
        plt.xticks(
            ticks=[
                i * (alphabet_size + 1) + (alphabet_size / 2)
                for i in range(len(args.pos))
            ],
            labels=args.pos,
        )
        plt.ylim((0, 0.1))
        plt.show()

        plt.figure()
        plt.bar(
            x=np.arange(alphabet_size),
            height=[1] * alphabet_size,
            color=COLORS[1 : alphabet_size + 1],
            tick_label=[
                "A",
                "R",
                "N",
                "D",
                "C",
                "Q",
                "E",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
                "X",
                "B",
                "Z",
                "J",
                "U",
                "O",
            ],
        )
        plt.show()

    else:
        lims = [0.0175, 0.055, 0.02, 0.035, 0.012, 0.0175, 0.03, 0.03, 0.055, 0.025]
        for i in range(len(args.pos)):
            fig, axs = plt.subplots(1, 2)
            bp_res = axs[0].boxplot(
                motif_res[i * (alphabet_size + 1) : (i + 1) * (alphabet_size + 1)],
                flierprops=dict(markerfacecolor="k", marker="D", markersize=1),
            )
            bp_sus = axs[1].boxplot(
                motif_sus[i * (alphabet_size + 1) : (i + 1) * (alphabet_size + 1)],
                flierprops=dict(markerfacecolor="k", marker="D", markersize=1),
            )
            set_box_color(
                bp_res, colors=COLORS[: alphabet_size + 1], lw=1, facecolor=False
            )
            set_box_color(
                bp_sus, colors=COLORS[: alphabet_size + 1], lw=1, facecolor=False
            )
            axs[0].set_ylim(0, lims[i])
            axs[1].set_ylim(0, lims[i])
            plt.title(str(args.pos[i]))
            plt.show()


def cv_performance(args):
    """Function to assess the cross-validation performance of a model.
    """
    results = []
    for i in range(args.folds):

        # load the dictionary containing the validation performance
        res_dict = torch.load(args.filepath + args.filename.format(i))
        val_perf = res_dict["val_performance"]

        # transform dictionary into pandas dataframe
        val_perf = pd.DataFrame.from_dict(val_perf)
        results.append(val_perf)

    # concatenate results of all folds
    df_res = pd.concat(results, axis=1)
    df_res["mean"] = df_res.mean(axis=1)
    df_res["std"] = df_res[:-1].std(axis=1)

    # print results
    print("===============================")
    print(f"Results for {args.data}")
    print("===============================\n")

    print("Parameters:")
    print(f"    motif length  : {args.modelargs[1]}")
    print(f"    sigma         : {args.modelargs[2]}")
    print(f"    beta (kernel) : {args.modelargs[3]}")
    print(f"    alpha (kernel): {args.modelargs[4]}")
    print(f"    anchors       : {args.modelargs[5]}")
    print(f"    layers        : {args.modelargs[6]}\n")

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None,
    ):
        print(df_res)


def main():
    """Main function of the analysis script.
    """

    # read in command line arguments
    args = load_args()

    # perfom the requested analysis
    if args.type == "robustness_pos":
        robustness_pos(args)
    elif args.type == "robustness_motif":
        robustness_motif(args)
    elif args.type == "cvperformance":
        cv_performance(args)


if __name__ == "__main__":
    main()
