__author__ = "Jason C. Klima"


import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MultipleLocator
from pathlib import Path

from src.plotting.config import rc_params


def main(
    output_path: str,
    simulation_records_in_scorefile: bool = True,
    set_xlim: bool = True,
    set_ylim: bool = False,
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
    df.plot.scatter(
        x=x,
        y=y,
        c=df[c] / cbar_scale,
        cmap="RdBu",
        norm=norm,
        s=25,
        edgecolor="k",
        ax=ax,
    )
    if set_xlim:
        x_min = max(np.floor(df[x].min()) - 1, 0)
        x_max = np.ceil(df[x].max()) + 1
        ax.set_xlim(x_min, x_max)
    if set_ylim:
        y_min = np.floor(df[y].min()) - 1
        y_max = np.ceil(df[y].max()) + 1
        ax.set_ylim(y_min, y_max)
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
        "--set_ylim",
        dest="set_ylim",
        action="store_true",
        help="Set the y-axis limits.",
    )
    parser.set_defaults(
        simulation_records_in_scorefile=True,
        set_xlim=True,
        set_ylim=False,
    )
    args = parser.parse_args()
    main(
        args.output_path,
        simulation_records_in_scorefile=args.simulation_records_in_scorefile,
        set_xlim=args.set_xlim,
        set_ylim=args.set_ylim,
    )
