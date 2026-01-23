__author__ = "Jason C. Klima"


import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from typing import Optional

from src.plotting.config import rc_params


def main(
    output_path: str,
    simulation_records_in_scorefile: bool = True,
    set_xlim: bool = True,
    set_ylim: bool = True,
    legend_fontsize: Optional[int] = None,
) -> None:
    """Plot PyRosettaCluster usage example #1 results."""
    scorefile = Path(output_path) / "scores.json"
    records = {}
    with scorefile.open("r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if simulation_records_in_scorefile:
                    output_file = d["metadata"]["output_file"]
                    scores = {output_file: d["scores"]}
                else:
                    scores = d
                records.update(scores)

    # Setup DataFrame
    df = (
        pd.DataFrame.from_dict(records, orient="index")
        .reset_index()
        .rename(columns={"index": "output_file"})
    )

    # Plot
    mpl.rcParams.update(rc_params)
    dpi = 600
    fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
    cbar_magnitude = 9
    cbar_scale = pow(10, cbar_magnitude)
    norm = TwoSlopeNorm(
        vmin=-(2 ** 32 / 2) / cbar_scale,
        vcenter=0.0,
        vmax=((2 ** 32 / 2) - 1) / cbar_scale,
    )
    x = "rmsd_all_heavy"
    y = "total_score"
    c = "seed"
    s = 25
    df.plot.scatter(
        x=x,
        y=y,
        c=df[c] / cbar_scale,
        s=s,
        cmap="RdYlBu",
        norm=norm,
        edgecolor="k",
        ax=ax,
    )
    if set_xlim:
        x_min = max(np.floor(df[x].min()) - 1, 0)
        ax.set_xlim(left=x_min)
    if set_ylim:
        y_min = np.floor(df[y].min()) - 1
        ax.set_ylim(bottom=y_min)
    # Adjust axes labels
    label_fontsize = 12
    ax.set_xlabel("Heavy Atom RMSD (Å)", fontsize=label_fontsize)
    ax.set_ylabel(r"Total Score ($\mathtt{beta\_jan25}$)", fontsize=label_fontsize)
    cbar = ax.collections[0].colorbar
    cbar.set_label(f"Seed ($\\times 10^{cbar_magnitude}$)", fontsize=label_fontsize)
    # Adjust axes tick marks
    tick_fontsize = 10
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(3))
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.5))
    # Set title
    ax.set_title("Heavy Atom RMSD vs. Total Score", fontsize=14)
    # Adjust Legend
    idx_min = df[y].idxmin()
    x_min = df.loc[idx_min, x]
    y_min = df.loc[idx_min, y]
    c_min = df.loc[idx_min, c] / cbar_scale
    idx_max = df[y].idxmax()
    x_max = df.loc[idx_max, x]
    y_max = df.loc[idx_max, y]
    c_max = df.loc[idx_max, c] / cbar_scale
    markersize_small = 7
    markersize_large = 10
    idx_min_color = plt.cm.RdBu(norm(c_min))
    idx_max_color = plt.cm.RdBu(norm(c_max))
    ax.scatter(x_min, y_min, marker="o", s=s * 3, edgecolor="k", color=idx_min_color, lw=1, zorder=5)
    ax.scatter(x_max, y_max, marker="s", s=s * 3, edgecolor="k", color=idx_max_color, lw=1, zorder=5)
    ax.scatter(x_min, y_min, marker="+", s=s * 7, color="k", lw=2, zorder=5)
    legend_handles = [
        # Lowest energy decoy (Decoy-1)
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=idx_min_color,
            markeredgecolor="k",
            markeredgewidth=1,
            markersize=markersize_large,
            label="Decoy-1",
        ),
        # Highest energy decoy (Decoy-N)
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            markerfacecolor=idx_max_color,
            markeredgecolor="k",
            markeredgewidth=1,
            markersize=markersize_large,
            label=f"Decoy-{df.shape[0]}",
        ),
        # Decoys
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="lightgray",
            markeredgecolor="k",
            markeredgewidth=1,
            markersize=markersize_small,
            label="Decoys",
        ),
        # Reproduced decoy (Decoy-N+1)
        ax.scatter(
            [], [],
            marker="+",
            s=120,
            color="k",
            lw=2,
            label=f"Decoy-{df.shape[0] + 1}",
        ),
    ]
    if legend_fontsize is None:
        legend_fontsize = tick_fontsize
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=legend_fontsize,
    )
    # Save
    fig.tight_layout()
    fig.savefig(
        Path(output_path) / "rmsd_total_score_seed_scatter_plot.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot PyRosettaCluster usage example #1 results.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The PyRosettaCluster simulation output directory.",
    )
    parser.add_argument(
        "--no-simulation_records_in_scorefile",
        dest="simulation_records_in_scorefile",
        action="store_false",
        help="Simulation records are not in the PyRosettaCluster output scorefile.",
    )
    parser.add_argument(
        "--no-set_xlim",
        dest="set_xlim",
        action="store_false",
        help="Do not set the x-axis limits.",
    )
    parser.add_argument(
        "--no-set_ylim",
        dest="set_ylim",
        action="store_false",
        help="Do not set the y-axis limits.",
    )
    parser.add_argument(
        "--legend_fontsize",
        type=int,
        required=False,
        default=None,
        help="Set the legend fontsize.",
    )
    parser.set_defaults(
        simulation_records_in_scorefile=True,
        set_xlim=True,
        set_ylim=True,
    )
    args = parser.parse_args()
    main(
        args.output_path,
        simulation_records_in_scorefile=args.simulation_records_in_scorefile,
        set_xlim=args.set_xlim,
        set_ylim=args.set_ylim,
        legend_fontsize=args.legend_fontsize,
    )
