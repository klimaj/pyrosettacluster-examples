__author__ = "Jason C. Klima"


import argparse
import os
import pyrosetta
import pyrosetta.distributed.io as io
import tempfile

from dask.distributed import Client, LocalCluster
from pathlib import Path
from pyrosetta.distributed.cluster import PyRosettaCluster
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.toolbox.rcsb import load_from_rcsb
from typing import Any, Dict, Generator

from src.protocols.pyrosetta import blueprintbdr


PDB_CODE: str = "1L2Y"


def initialize_pyrosetta() -> None:
    """
    Initialize PyRosetta on the client.

    Returns:
        None
    """
    pyrosetta.init("-run:constant_seed 1")


def get_input_packed_pose() -> PackedPose:
    """
    Return a `PackedPose` object from a multimodel PDB accession number.

    Returns:
        The first model in the multimodel PDB accession number as a `PackedPose` object.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdb_filename = str(Path(tmp_dir) / f"{PDB_CODE}.pdb")
        load_from_rcsb(PDB_CODE, pdb_filename=pdb_filename)
        packed_pose = next(iter(io.poses_from_multimodel_pdb(pdb_filename)))

    return packed_pose


def create_tasks(num_tasks: int) -> Generator[Dict[str, Any], None, None]:
    """
    Create tasks for a PyRosettaCluster simulation.

    Args:
        num_tasks: A required `int` object representing the number of tasks to generate.

    Yields:
        An output `dict` object representing a task.
    """
    for task_id in range(num_tasks):
        yield {
            "options": {
                "beta_jan25": "1",
                "score:count_pair_hybrid": "0",
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
            "task_id": task_id,
        }


def main(
    output_path: str,
    scratch_dir: str,
    num_tasks: int = 100,
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
        threads_per_worker=1,
        memory_limit=f"{12.6 / n_workers:.1f}GB",
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
    args = parser.parse_args()
    main(args.output_path, args.scratch_dir)
