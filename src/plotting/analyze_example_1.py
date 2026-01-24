__author__ = "Jason C. Klima"


import argparse

from pathlib import Path

from src.utils import get_dataframe


def main(original_scorefile: Path, scorefxn: str) -> None:
    """Print info about the lowest and highest energy decoys."""
    df = get_dataframe(original_scorefile)
    x = "rmsd_all_heavy"
    if scorefxn == "beta_jan25":
        y = "total_score"
    elif scorefxn == "ref2015":
        y = "total_score_ref2015"
    else:
        raise ValueError(f"Scorefunction is not supported: '{scorefxn}'")
    c = "seed"
    idx_min = df[y].idxmin()
    x_min = df.loc[idx_min, x]
    y_min = df.loc[idx_min, y]
    c_min = df.loc[idx_min, c]
    print(f"Lowest energy decoy ({x}={x_min}; {y}={y_min}; {c}={int(c_min)}):", df.loc[idx_min, "output_file"],)
    idx_max = df[y].idxmax()
    x_max = df.loc[idx_max, x]
    y_max = df.loc[idx_max, y]
    c_max = df.loc[idx_max, c]
    print(f"Highest energy decoy ({x}={x_max}, {y}={y_max}, {c}={int(c_max)}):", df.loc[idx_max, "output_file"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze PyRosettaCluster usage example #1 results.",
    )
    parser.add_argument(
        "--original_scorefile",
        type=Path,
        required=True,
        help="The original PyRosettaCluster simulation output scorefile (i.e., a 'scores.json' file).",
    )
    parser.add_argument(
        "--scorefxn",
        type=str,
        required=False,
        default="beta_jan25",
        help="The scorefunction value to plot.",
    )
    args = parser.parse_args()
    main(args.original_scorefile, args.scorefxn)
