__author__ = "Jason C. Klima"


import argparse

from pathlib import Path

from src.utils import get_bb_rmsd_nosuper, get_dataframe_from_pickle


def main(original_decoy: Path, reproduce_decoy: Path) -> None:
    """Print info about the lowest scRMSD decoy."""
    bb_rmsd = get_bb_rmsd_nosuper(str(original_decoy), str(reproduce_decoy))
    print(bb_rmsd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze PyRosettaCluster usage example #2 results.",
    )
    parser.add_argument(
        "--original_decoy",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reproduce_decoy",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    main(args.original_decoy, args.reproduce_decoy)
