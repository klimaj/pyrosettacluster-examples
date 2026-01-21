__author__ = "Jason C. Klima"


import argparse
import os
import pyrosetta
import pyrosetta.distributed.io as io

from dask.distributed import Client, LocalCluster
from pyrosetta.distributed.cluster import PyRosettaCluster
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Dict, Generator

from src.protocols.pyrosetta import test_protocol


def create_tasks(num_tasks: int) -> Generator[Dict[str, Any], None, None]:
    """
    Create tasks with a 'task_id' key.
    
    Args:
        num_tasks: A required `int` object representing the number of tasks to generate.

    Yields:
        The output `dict` objects representing the tasks.
    """
    for i in range(num_tasks):
        yield {
            "options": {"multithreading:total_threads": "1"},
            "set_logging_handler": "logging",
            "task_id": i
        }


def get_input_packed_pose() -> PackedPose:
    """Return the input `PackedPose` object."""
    return io.pose_from_sequence("TEST")


def main(
    output_path: str,
    num_tasks: int = 5,
    num_protocols: int = 3,
) -> None:
    """Run the PyRosettaCluster example #1 simulation."""
    pyrosetta.init("-run:constant_seed 1")

    with LocalCluster(
        n_workers=os.cpu_count(),
        threads_per_worker=1,
        memory_limit="4GB",
        scheduler_port=8786,
        dashboard_address=":8787",
    ) as cluster, Client(cluster) as client:
        PyRosettaCluster(
            tasks=create_tasks(num_tasks),
            input_packed_pose=get_input_packed_pose(),
            client=client,
            output_path=output_path,
        ).distribute(
            protocols=[test_protocol] * num_protocols,
            clients_indices=None,
            priorities=None,
            resources=None,
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
    args = parser.parse_args()
    main(args.output_path)
