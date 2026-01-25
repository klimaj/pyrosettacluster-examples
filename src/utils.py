__author__ = "Jason C. Klima"


import json
import pandas as pd
import pyrosetta
import pyrosetta.distributed.io as io
import time

from functools import wraps
from pathlib import Path
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Callable, TypeVar, cast


T = TypeVar("T", bound=Callable[..., Any])


def write_resfile(packed_pose: PackedPose, output_path: str) -> Path:
    """
    Write a resfile to the provided output directory.

    Args:
        packed_pose: A required `PackedPose` object containing the sequence for the resfile.
        output_path: A required `str` object representing the output directory.

    Returns:
        A `Path` object for the resfile.
    """
    seq = packed_pose.pose.sequence()
    resfile = (Path(output_path) / "resfile").resolve()
    resfile.parent.mkdir(parents=True, exist_ok=True)
    with resfile.open("w") as f:
        f.write("start\n")
        for i, s in enumerate(seq, start=1):
            f.write(f"{i} A PIKAA {s}\n")

    return resfile


def write_blueprint(packed_pose: PackedPose, output_path: str) -> Path:
    """
    Write a blueprint file to the provided output directory.

    Args:
        packed_pose: A required `PackedPose` object containing the sequence/structure for the blueprint file.
        output_path: A required `str` object representing the output directory.

    Returns:
        A `Path` object for the blueprint file.
    """
    pose = packed_pose.pose
    seq = pose.sequence()
    ss = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()
    abegos = "".join(pyrosetta.rosetta.core.sequence.get_abego(pose))
    assert len(seq) == len(ss) == len(abegos), "Sequence/secondary structure/ABEGO size mismatch."
    space = " " * 4
    blueprint = (Path(output_path) / "blueprint").resolve()
    blueprint.parent.mkdir(parents=True, exist_ok=True)
    with blueprint.open("w") as f:
        for i in range(len(seq)):
            f.write(f"0{space}{seq[i]}{space}{ss[i]}{abegos[i]}{space}R\n")

    return blueprint


def timeit(func: T) -> T:
    """
    Decorator that prints the runtime of a function after it finishes.

    Args:
        func: A required callable to be timed.

    Returns:
        A callable with the same function signature and return type as `func`.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function that times `func` and prints its runtime."""
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"The `{func.__name__}` function finished in {dt:.3f} seconds.")

        return result

    return cast(T, wrapper)


def get_dataframe(scorefile: Path) -> pd.DataFrame:
    """
    Return a `pandas.DataFrame` object from a JSON-formatted scorefile.

    Args:
        scorefile: A required `Path` object to the JSON-formatted scorefile.

    Returns:
        A `pandas.DataFrame` object.
    """
    if not scorefile.name.endswith(".json"):
        raise ValueError(f"Scorefile must end with '.json'. Received: '{scorefile}")

    records = {}
    with scorefile.open("r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if all(x in d for x in ("instance", "metadata", "scores")):
                    output_file = d["metadata"]["output_file"]
                    scores = {output_file: d["scores"]}
                else:
                    scores = d
                records.update(scores)

    return (
        pd.DataFrame.from_dict(records, orient="index")
        .reset_index()
        .rename(columns={"index": "output_file"})
    )


def atom_array_to_packed_pose(atom_array: "AtomArray") -> PackedPose:
    from biotite.structure.io.pdb import PDBFile
    from io import StringIO

    buffer = StringIO()
    pdb = PDBFile()
    pdb.set_structure(atom_array)
    pdb.write(buffer)
    pdbstring = buffer.getvalue()
    packed_pose = io.pose_from_pdbstring(pdbstring)

    return packed_pose
