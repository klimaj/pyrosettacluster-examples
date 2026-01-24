__author__ = "Jason C. Klima"


import argparse
import hashlib
import os
import pyrosetta
import pyrosetta.distributed.io as io
import struct
import tempfile

from dask.distributed import Client, LocalCluster
from pathlib import Path
from pyrosetta import Pose
from pyrosetta.distributed.cluster import PyRosettaCluster
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.toolbox.rcsb import load_from_rcsb
from typing import Any, Dict, Generator

from src.protocols.pyrosetta import blueprintbdr


PDB_CODE: str = "1L2Y"
PDB_CODE_SHA256: str = "5d1bbb545a312dfff1ae1e64b6d8addecb2f561ddc4011aeb5bee9d1dfcd4438"
PACKED_POSE_SHA256: str = "510d68790d6778fbca3a9c97752fde011a4c9a2debaebcc99d8f5938931fca1f"


def initialize_pyrosetta() -> None:
    """
    Initialize PyRosetta on the client.

    Returns:
        None
    """
    pyrosetta.init("-run:constant_seed 1 -multithreading:total_threads 1")


def get_input_packed_pose() -> PackedPose:
    """
    Return a `PackedPose` object from a downloaded multimodel PDB accession number.

    Raises:
        `AssertionError` if the resulting `PackedPose` object from the downloaded
        PDB file is not expected.

    Returns:
        The first model in the multimodel PDB accession number as a `PackedPose` object.
    """

    def get_pdb_code_digest(pdb_filename: Path) -> str:
        """
        Return a SHA256 digest of the input `Path` object for a PDB file.

        Args:
            pdb_filename: The input `Path` object to digest.

        Returns:
            A `str` object representing the `Path` object's SHA256 digest.
        """
        with pdb_filename.open("r") as f:
            return hashlib.sha256(f.read().encode("utf-8")).hexdigest()

    def get_pose_digest(pose: Pose) -> str:
        """
        Return a SHA256 digest of residue numbers, atom numbers, atom names,
        and double precision atomic coordinates of the `Pose` object.

        Args:
            pose: The input `Pose` object to digest.

        Returns:
            A `str` object representing the `Pose` object's SHA256 digest.
        """
        h = hashlib.sha256()
        for res in range(1, pose.size() + 1):
            residue = pose.residue(res)
            h.update(struct.pack("!i", res))
            h.update(residue.name().strip().encode("utf-8"))
            for atom in range(1, residue.natoms() + 1):
                h.update(residue.atom_name(atom).strip().encode("utf-8"))
                xyz = residue.atom(atom).xyz()
                for axis in "xyz":
                    h.update(struct.pack("!d", getattr(xyz, axis)))

        return h.hexdigest()

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdb_filename = Path(tmp_dir) / f"{PDB_CODE}.pdb"
        load_from_rcsb(PDB_CODE, pdb_filename=pdb_filename)
        pdb_code_digest = get_pdb_code_digest(pdb_filename)
        if pdb_code_digest != PDB_CODE_SHA256:
            raise AssertionError("The PDB file SHA256 digest is not expected.")
        packed_pose = next(iter(io.poses_from_multimodel_pdb(pdb_filename)))
    if get_pose_digest(packed_pose.pose) != PACKED_POSE_SHA256:
        raise AssertionError("The PackedPose SHA256 digest is not expected.")

    return packed_pose


def create_tasks(num_tasks: int) -> Generator[Dict[str, Any], None, None]:
    """
    Create tasks for a PyRosettaCluster simulation.

    Args:
        num_tasks: A required `int` object representing the number of tasks to generate.

    Yields:
        An output `dict` object representing a task.
    """
    for _ in range(num_tasks):
        yield {
            "options": {
                "beta_nov16": "1",
            },
            "extra_options": {
                "multithreading:total_threads": "1",
                "linmem_ig": "10",
                "no_optH": "0",
                "flip_HNQ": "0",
                "no_his_his_pairE": "1",
                "run:preserve_header": "1",
                "nblist_autoupdate": "1",
                "write_all_connect_info": "1",
                "connect_info_cutoff": "3.0",
            },
            "set_logging_handler": "logging",
        }


def main(
    output_path: str,
    scratch_dir: str,
    num_tasks: int,
) -> None:
    """Run the PyRosettaCluster example #1 simulation."""
    # Initialize PyRosetta
    initialize_pyrosetta()

    # Setup the simulation inputs
    input_packed_pose = get_input_packed_pose()

    # Set the number of workers dynamically
    n_workers = os.cpu_count()
    print(f"Spinning up {n_workers} dask workers.")

    # Run the simulation
    with LocalCluster(
        n_workers=n_workers,
        threads_per_worker=2,
        memory_limit=f"{12.7 / n_workers:.2f}GB",
        scheduler_port=8786,
        dashboard_address=":8787",
        resources={"CPU": 1},
    ) as cluster, Client(cluster) as client:
        protocols = [blueprintbdr]
        num_protocols = len(protocols)
        PyRosettaCluster(
            tasks=create_tasks(num_tasks),
            input_packed_pose=input_packed_pose,
            client=client,
            scratch_dir=scratch_dir,
            output_path=output_path,
            project_name="pyrosettacluster-examples",
            simulation_name="example-1",
            simulation_records_in_scorefile=True,
            output_init_file=None,
            author=__author__,
            license=(
                f"Copyright (c) {__author__}. "
                "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND)"
            ),
        ).distribute(
            protocols=protocols,
            clients_indices=[0] * num_protocols,
            priorities=list(range(num_protocols)),
            resources=[{"CPU": 1}] * num_protocols,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyRosettaCluster usage example #1.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The PyRosettaCluster simulation output directory.",
    )
    parser.add_argument(
        "--scratch_dir",
        type=str,
        required=True,
        help="The PyRosettaCluster simulation scratch directory.",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=50,
        required=False,
        help="The number of tasks in the PyRosettaCluster simulation.",
    )
    args = parser.parse_args()
    main(
        args.output_path,
        args.scratch_dir,
        args.num_tasks,
    )
