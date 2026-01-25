__author__ = "Jason C. Klima"


import argparse
import os
import subprocess
import pyrosetta
import pyrosetta.distributed.io as io

from dask.distributed import Client, LocalCluster
from pathlib import Path
from pyrosetta.distributed.cluster import PyRosettaCluster
from typing import Any, Dict, Generator

from src.protocols.foundry import solublempnn, rfd3


def initialize_pyrosetta() -> None:
    """
    Initialize PyRosetta on the client.

    Returns:
        None
    """
    pyrosetta.init("-run:constant_seed 1 -multithreading:total_threads 1")
    pyrosetta.secure_unpickle.add_secure_package("pandas")
    pyrosetta.secure_unpickle.add_secure_package("biotite")


def download_checkpoints() -> None:
    subprocess.run(
        ["foundry", "install", "rfd3", "solublempnn"],
        check=True,
    )


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
        }


def main(
    output_path: str,
    scratch_dir: str,
    num_tasks: int,
) -> None:
    """Run the PyRosettaCluster example #2 simulation."""
    # Initialize PyRosetta
    initialize_pyrosetta()
    # Download Foundry checkpoints
    download_checkpoints()

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
        protocols = [rfd3, solublempnn]
        num_protocols = len(protocols)
        PyRosettaCluster(
            tasks=create_tasks(num_tasks),
            input_packed_pose=None,
            client=client,
            scratch_dir=scratch_dir,
            output_path=output_path,
            seeds=[111] * len(protocols),
            decoy_ids=[0] * len(protocols),
            project_name="pyrosettacluster-examples",
            simulation_name="example-2",
            simulation_records_in_scorefile=False,
            filter_results=True,
            output_init_file=None,
            compression=True,
            compressed=False,
            output_decoy_types=[".pdb"],
            output_scorefile_types=[".json", ".bz2"],
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
        description="PyRosettaCluster usage example #2.",
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
