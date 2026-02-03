__author__ = "Jason C. Klima"


import hashlib
import json
import os
import pandas as pd
import pyrosetta
import pyrosetta.distributed.io as io
import time

from functools import wraps
from pathlib import Path
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Callable, Optional, TypeVar, cast


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


def get_dataframe_from_pickle(scorefile: Path) -> pd.DataFrame:
    """
    Return a `pandas.DataFrame` object from a pickled `pandas.DataFrame`-formatted scorefile.

    Args:
        scorefile: A required `Path` object to the `pandas.DataFrame`-formatted scorefile.

    Returns:
        A `pandas.DataFrame` object.
    """
    df = (
        pd.read_pickle(scorefile, compression="infer")
        .reset_index(drop=False)
        .rename(columns={"index": "output_file"})
    )
    if set(df.columns) == {"output_file", "scores", "metadata", "instance"}:
        scores_df = df["scores"].apply(pd.Series)
        instance_df = df["instance"].apply(pd.Series)
        df = pd.concat([df[["output_file"]], instance_df[["decoy_ids", "seeds"]], scores_df], axis=1)

    return df


def pyrosetta_to_torch_seed(pyrosetta_seed: int) -> int:
    """
    Scale an input PyRosetta seed to the Torch seed proper range.
    PyRosetta seed range: [-(2 ** 31), (2 ** 31) - 1]
    Torch seed range: [0, (2 ** 32) - 1]

    Args:
        pyrosetta_seed: An `int` object representing the PyRosetta seed.

    Returns:
        An `int` object representing the Torch seed.
    """
    return pyrosetta_seed + (2 ** 31)


def atom_array_to_packed_pose(atom_array: "AtomArray") -> PackedPose:
    """
    Convert a biotite `AtomArray` object to a PyRosetta `PackedPose`
    object in memory.

    Args:
        atom_array: A biotite `AtomArray` object.

    Returns:
        A `PackedPose` object.
    """
    from biotite.structure.io.pdb import PDBFile
    from io import StringIO

    buffer = StringIO()
    pdb = PDBFile()
    pdb.set_structure(atom_array)
    pdb.write(buffer)
    pdbstring = buffer.getvalue()
    packed_pose = io.pose_from_pdbstring(pdbstring)

    return packed_pose


def print_protocol_info(**kwargs: Any) -> None:
    """
    Print user-provided PyRosetta protocol and CUDA info during runtime.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        None
    """
    import torch

    protocol_name = kwargs["PyRosettaCluster_protocol_name"]
    protocol_number = kwargs["PyRosettaCluster_protocol_number"]
    seed = kwargs["PyRosettaCluster_seed"]
    client_repr = kwargs["PyRosettaCluster_client_repr"]
    cuda_is_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_is_available else None
    print(
        "Running --",
        f"Protocol name: '{protocol_name}';",
        f"Protocol number: {protocol_number};",
        f"Protocol seed: {seed};",
        f"Client: '{client_repr}';",
        f"CUDA is available: {cuda_is_available};",
        f"CUDA device count: {cuda_device_count};",
        f"CUDA device name: {cuda_device_name};",
        sep=" ",
    )


def get_sha256_digest(checkpoint_file: Path, size=1024 * 1024, verbose=False) -> str:
    """
    Generate a SHA256 digest of a binary checkpoint file.

    Args:
        checkpoint_file: A required `Path` object for which to generate the SHA256 digest.

    Keyword Args:
        size: An `int` object representing the chuck size in bytes to use for iterating
            over the checkpoint file.
            Default: 1024 * 1024
        verbose: A `bool` object specifying whether or not to print the result.

    Returns:
        A `str` object representing the SHA256 digest.
    """
    h = hashlib.sha256()
    with checkpoint_file.open("rb") as f:
        while data := f.read(size):
            h.update(data)
    digest = h.hexdigest()
    if verbose:
        print(f"Generated SHA256 digest for checkpoint file '{checkpoint_file}': {digest}")

    return digest


def get_bb_rmsd_nosuper(file1: str, file2: str, flags: Optional[str] = None) -> float:
    """
    Return the backbone heavy atom root-mean-squared deviation (RMSD) without superposition
    between two input structure files. If PyRosetta is not yet initialized, then PyRosetta
    will first be initialized with the optionally input PyRosetta initialization flags, 
    otherwise "-mute all" flags are used.

    Args:
        file1: A `str` object representing the first structure file path.
        file2: A `str` object representing the second structure file path.

    Keyword Args:
        flags: An optional `str` object representing PyRosetta initialization
            options to use if PyRosetta is not already initialized.
            Default: ""

    Returns:
        A `float` object representing the backbone heavy atom RMSD.
    """
    from pyrosetta.rosetta.core.scoring import (
        rms_at_corresponding_atoms_no_super,
        setup_matching_protein_backbone_heavy_atoms,
    )
    from pyrosetta.rosetta.std import map_core_id_AtomID_core_id_AtomID

    for file in (file1, file2):
        if not isinstance(file, str) or not os.path.isfile(file):
            raise ValueError(f"The input file must be a `str` object and exist: '{file}'.")
    if flags and not isinstance(flags, str):
        raise ValueError(
            "The 'flags' keyword argument parameter must be an instance of `str`. "
            f"Received: {type(flags)}"
        )
    if not pyrosetta.rosetta.basic.was_init_called():
        extra_options = flags if flags else "-mute all"
        pyrosetta.init(options="", extra_options=extra_options, silent=True)

    pose1 = io.pose_from_file(file1).pose
    pose2 = io.pose_from_file(file2).pose
    atom_id_map = map_core_id_AtomID_core_id_AtomID()
    setup_matching_protein_backbone_heavy_atoms(pose1=pose1, pose2=pose2, atom_id_map=atom_id_map)
    bb_rmsd = rms_at_corresponding_atoms_no_super(mod_pose=pose1, ref_pose=pose2, atom_id_map=atom_id_map)

    return bb_rmsd


def get_sequence_percent_identity(seq1: str, seq2: str) -> float:
    """
    Return the percent sequence identity between two sequences.

    Args:
        seq1: A `str` object representing the first sequence.
        seq2: A `str` object representing the second sequence.

    Returns:
        A `float` object representing the percent sequence identity.
    """
    assert len(seq1) == len(seq2), f"Input sequences must be equal length: {len(seq1)} != {len(seq2)}"
    identical = sum(res1 == res2 for res1, res2 in zip(seq1, seq2))
    total = len(seq1)

    return (identical / total) * 100
