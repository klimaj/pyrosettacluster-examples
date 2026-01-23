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


PDB_CODE = "1L2Y"


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


def get_input_packed_pose(pdb_code: str) -> PackedPose:
    """
    Return a `PackedPose` object from a multimodel PDB accession number.
    
    Args:
        pdb_code: A `str` object representing a multimodel PDB accession number.

    Returns:
        The first model in the multimodel PDB accession number as a `PackedPose` object.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdb_filename = str(Path(tmp_dir) / f"{pdb_code}.pdb")
        load_from_rcsb(pdb_code, pdb_filename=pdb_filename)
        packed_pose = next(iter(io.poses_from_multimodel_pdb(pdb_filename)))

    return packed_pose


def create_tasks(xml_str: str, num_tasks: int) -> Generator[Dict[str, Any], None, None]:
    """
    Create tasks for a PyRosettaCluster simulation.

    Args:
        xml_str: A required `str` object representing a RosettaScripts XML string.
        num_tasks: A required `int` object representing the number of tasks to generate.

    Yields:
        An output `dict` object representing a task.
    """
    for task_id in range(num_tasks):
        yield {
            "extra_options": {
                "multithreading:total_threads": "1",
            },
            "set_logging_handler": "logging",
            "xml_str": xml_str,
            "task_id": task_id,
        }


def main(
    output_path: str,
    scratch_dir: str,
    num_tasks: int = 20,
) -> None:
    """Run the PyRosettaCluster example #1 simulation."""
    # Initialize PyRosetta
    pyrosetta.init("-run:constant_seed 1")

    # Setup the simulation inputs
    input_packed_pose = get_input_packed_pose(PDB_CODE)
    resfile = write_resfile(input_packed_pose, output_path)
    blueprint = write_blueprint(input_packed_pose, output_path)
    xml_file = Path(__file__).parent.parent / "rosetta_scripts" / "blueprintbdr.xml"
    xml_str = xml_file.read_text().format(
        resfile=resfile,
        blueprint=blueprint,
    )

    # Set the number of workers dynamically
    n_workers = os.cpu_count()
    print(f"Spinning up {n_workers} dask workers.")

    # Run the simulation
    with LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit="7GB",
        scheduler_port=8786,
        dashboard_address=":8787",
        resources={"CPU": 1},
    ) as cluster, Client(cluster) as client:
        protocols = [blueprintbdr]
        num_protocols = len(protocols)
        PyRosettaCluster(
            tasks=create_tasks(xml_str, num_tasks),
            input_packed_pose=input_packed_pose,
            client=client,
            scratch_dir=scratch_dir,
            output_path=output_path,
            project_name="pyrosettacluster-examples",
            simulation_name="example-1",
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
