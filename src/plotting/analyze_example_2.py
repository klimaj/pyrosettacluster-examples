__author__ = "Jason C. Klima"


import argparse

from pathlib import Path

from src.utils import get_dataframe_from_pickle


def main(original_scorefile: Path) -> None:
    """Print info about the lowest scRMSD decoy."""
    df = get_dataframe_from_pickle(original_scorefile)
    v = (
        df
        .loc[df["protocol_number"].eq(5)]
        .sort_values("bb_rmsd", ascending=True)
        .reset_index(drop=True)
        .iloc[0] # Top ranked design
    )
    bb_rmsd = v["bb_rmsd"]
    total_score = v["total_score"]
    protocol_number = v["protocol_number"]
    decoy_ids = v["decoy_ids"]
    seeds = v["seeds"]
    output_file = Path(v["output_file"]).with_suffix(".b64_pose")
    metrics_str = "; ".join(
        [
            f"bb_rmsd={bb_rmsd}",
            f"total_score={total_score}",
            f"protocol_number={protocol_number}",
            f"decoy_ids={decoy_ids}",
            f"seeds={seeds}",
        ]
    )
    print(f"Lowest scRMSD decoy ({metrics_str}): {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze PyRosettaCluster usage example #2 results.",
    )
    parser.add_argument(
        "--original_scorefile",
        type=Path,
        required=True,
        help="The original PyRosettaCluster simulation output pickled `pandas.DataFrame` scorefile (i.e., a 'scores.bz2' file).",
    )
    args = parser.parse_args()
    main(args.original_scorefile)
