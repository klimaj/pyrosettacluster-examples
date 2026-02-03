__author__ = "Jason C. Klima"


import argparse
import json
import pandas as pd

from pathlib import Path
from typing import List

from src.utils import (
    get_bb_rmsd_nosuper,
    get_dataframe_from_pickle,
    get_sequence_percent_identity,
)


def expand_decoy_ids(df: pd.DataFrame, column="decoy_ids", prefix="decoy_id_protocol_number") -> pd.DataFrame:
    df_decoy_id = pd.DataFrame(df[column].tolist(), index=df.index).add_prefix(f"{prefix}_").astype("Int64")

    return df.join(df_decoy_id)


def get_decoy_ids(df: pd.DataFrame, protocol_number: int) -> List[int]:
    return (
        df
        .loc[df["protocol_number"].eq(protocol_number)]
        .sort_values("bb_rmsd", ascending=True)
        .reset_index(drop=True)
        .iloc[0, df.columns.get_loc("decoy_ids")] # Top ranked design
    )


def assert_one_row(df: pd.DataFrame) -> pd.DataFrame:
    assert len(df) == 1, f"The `pd.DataFrame` object has more than one row: {len(df)}"
    return df


def main(original_scorefile: Path, reproduce_scorefile: Path, protocol_number: int = 5) -> None:
    """Save info about the lowest scRMSD decoy."""
    df1 = expand_decoy_ids(get_dataframe_from_pickle(original_scorefile))
    df2 = expand_decoy_ids(get_dataframe_from_pickle(reproduce_scorefile))
    decoy_ids_1 = get_decoy_ids(df1, protocol_number)
    decoy_ids_2 = get_decoy_ids(df2, protocol_number)
    assert decoy_ids_1 == decoy_ids_2, f"Decoy IDs are not identical: {decoy_ids_1} != {decoy_ids_2}"

    protocol_number_data = {}
    for protocol_number, decoy_id in enumerate(decoy_ids_1):
        target_decoy_ids = decoy_ids_1[: (protocol_number + 1)]
        v1 = (
            df1
            .loc[
                (df1[f"decoy_id_protocol_number_{protocol_number}"].eq(decoy_id))
                & (df1["decoy_ids"].apply(lambda x: x == target_decoy_ids))
            ]
            .pipe(assert_one_row) # Fail if >1 task was run
            .iloc[0]
        )
        v2 = (
            df2
            .loc[
                (df2[f"decoy_id_protocol_number_{protocol_number}"].eq(decoy_id))
                & (df2["decoy_ids"].apply(lambda x: x == target_decoy_ids))
            ]
            .pipe(assert_one_row) # Fail if >1 task was run
            .iloc[0]
        )
        # Compute backbone heavy atom RMSD
        original_decoy = Path(v1["output_file"]).with_suffix(".b64_pose")
        reproduce_decoy = Path(v2["output_file"]).with_suffix(".b64_pose")
        bb_rmsd = get_bb_rmsd_nosuper(str(original_decoy), str(reproduce_decoy))
        # Compute sequence percent identity
        original_sequence = v1["sequence"]
        reproduce_sequence = v2["sequence"]
        sequence_percent_identity = get_sequence_percent_identity(original_sequence, reproduce_sequence)
        # Compute delta total_score
        original_total_score = v1["total_score"]
        reproduce_total_score = v2["total_score"]
        delta_total_score = float(reproduce_total_score - original_total_score)
        # Cache scores
        protocol_number_data[protocol_number] = {
            "bb_rmsd": bb_rmsd,
            "sequence_percent_identity": sequence_percent_identity,
            "delta_total_score": delta_total_score,
        }
    # Print data
    print("Results:", protocol_number_data)
    # Save data
    output_json_file = reproduce_scorefile.parent / "original_vs_reproduce_comparison_example-2.json"
    with output_json_file.open("w") as f:
        json.dump(protocol_number_data, f)
    print(f"Wrote: '{output_json_file}'")


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
    parser.add_argument(
        "--reproduce_scorefile",
        type=Path,
        required=True,
        help="The reproduced PyRosettaCluster simulation output pickled `pandas.DataFrame` scorefile (i.e., a 'scores.bz2' file).",
    )
    parser.add_argument(
        "--protocol_number",
        type=int,
        required=False,
        default=5,
        help="The final protocol number to analyze.",
    )
    args = parser.parse_args()
    main(
        args.original_scorefile,
        args.reproduce_scorefile,
        protocol_number=args.protocol_number,
    )
