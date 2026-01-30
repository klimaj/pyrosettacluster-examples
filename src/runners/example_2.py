__author__ = "Jason C. Klima"


import argparse
import os
import subprocess
import pyrosetta

from collections.abc import Callable
from dask.distributed import Client, LocalCluster
from pyrosetta.distributed.cluster import PyRosettaCluster
from typing import Any, Dict, Generator, Union

from src.protocols.foundry import proteinmpnn, rf3, rfd3
from src.protocols.pyrosetta import cart_min, compute_rmsd, cst_cart_min_poly_gly


def initialize_pyrosetta() -> None:
    """
    Initialize PyRosetta on the client.

    Returns:
        None
    """
    pyrosetta.init("-run:constant_seed 1 -multithreading:total_threads 1")
    pyrosetta.secure_unpickle.add_secure_package("pandas")


def download_checkpoints() -> None:
    """Download Foundry model weights"""
    subprocess.run(
        ["foundry", "install", "rfd3", "proteinmpnn", "rf3"],
        check=True,
    )


def create_tasks(num_tasks: int, gpu: bool) -> Generator[Dict[str, Any], None, None]:
    """
    Create tasks for a PyRosettaCluster simulation that uses a Dask `LocalCluster` instance.

    Args:
        num_tasks: A required `int` object representing the number of tasks to generate.
        gpu: A required `bool` object specifying whether or not to use GPU resources.

    Yields:
        An output `dict` object representing a task.
    """
    import torch

    if not isinstance(num_tasks, int):
        raise ValueError(
            f"The 'num_tasks' keyword argument parameter must be of type `int`. Received: {type(num_tasks)}"
        )
    if not isinstance(gpu, bool):
        raise ValueError(
            f"The 'gpu' keyword argument parameter must be of type `bool`. Received: {type(gpu)}"
        )

    cuda_visible_devices = ",".join(map(str, range(torch.cuda.device_count()))) if gpu and torch.cuda.is_available() else ""
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
            # RFdiffusion-3 parameters
            "rfd3": {
                "length": "20-30",
                "diffusion_batch_size": 2,
                "n_batches": 1,
            },
            # ProteinMPNN parameters
            "proteinmpnn": {
                "temperature": 0.1,
                "batch_size": 2,
                "number_of_batches": 1,
            },
            # RoseTTAFold-3 parameters
            "rf3": {
                "diffusion_batch_size": 2,
                "n_recycles": 5,
                "num_steps": 50,
            },
            # Protocol-specific parameters
            "mpnn_packed_pose_key": "mpnn_packed_pose",
            "cuda_visible_devices": cuda_visible_devices, 
        }


class Resources:
    """Manage compute resources for PyRosettaCluster example #2 simulation."""
    _gpu_enabled_protocols: set[Callable[..., Any]] = {proteinmpnn, rf3, rfd3}
    _cpu_resource: Dict[str, Union[float, int]] = {"CPU": 1}
    _gpu_resource: Dict[str, Union[float, int]] = {"GPU": 1}

    def __init__(self, *protocols: Callable[..., Any], gpu: bool = False) -> None:
        for protocol in protocols:
            if not callable(protocol):
                raise ValueError(
                    f"The protocol '{protocol!r}' is not a callable or generator object. Received: {type(protocol)}"
                )
        if not isinstance(gpu, bool):
            raise ValueError(
                f"The 'gpu' keyword argument parameter must be of type `bool`. Received: {type(gpu)}"
            )
        self.protocols: tuple[Callable[..., Any]] = protocols
        self.gpu: bool = gpu

    def get(self) -> list[Dict[str, Union[float, int]]]:
        """
        Get resources for the `PyRosettaCluster.distribute(resources=...)` keyword argument parameter.

        Returns:
            A `list` object of `dict` objects representing protocol-specific abstract resource constraints.
        """
        return [
            Resources._gpu_resource.copy()
            if self.gpu and protocol in Resources._gpu_enabled_protocols
            else Resources._cpu_resource.copy()
            for protocol in self.protocols
        ]


def main(
    output_path: str,
    scratch_dir: str,
    num_tasks: int,
    gpu: bool,
) -> None:
    """Run the PyRosettaCluster example #2 simulation."""
    # Initialize PyRosetta
    initialize_pyrosetta()
    # Download Foundry checkpoints
    download_checkpoints()

    # Set the number of workers dynamically
    n_workers = 1
    print(f"Spinning up {n_workers} dask workers.")

    # Setup client resources
    resources = {}
    resources.update(Resources._cpu_resource)
    if gpu:
        resources.update(Resources._gpu_resource)

    # Run the simulation
    with LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=f"{12.7 / n_workers:.2f}GB",
        scheduler_port=8786,
        dashboard_address=":8787",
        resources=resources,
    ) as cluster, Client(cluster) as client:
        # Setup protocols
        protocols = [rfd3, cst_cart_min_poly_gly, proteinmpnn, rf3, cart_min, compute_rmsd]
        num_protocols = len(protocols)
        PyRosettaCluster(
            tasks=create_tasks(num_tasks, gpu),
            input_packed_pose=None,
            client=client,
            scratch_dir=scratch_dir,
            output_path=output_path,
            project_name="pyrosettacluster-examples",
            simulation_name=f"example-2-gpu-{int(gpu)}",
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
            resources=Resources(*protocols, gpu=gpu).get(),
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
        default=1,
        required=False,
        help="The number of tasks in the PyRosettaCluster simulation.",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        help="Run the PyRosettaCluster simulation with GPUs enabled.",
    )
    parser.set_defaults(gpu=False)
    args = parser.parse_args()
    main(
        args.output_path,
        args.scratch_dir,
        args.num_tasks,
        args.gpu,
    )
