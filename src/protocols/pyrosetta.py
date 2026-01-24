__author__ = "Jason C. Klima"


from pyrosetta.distributed.cluster import requires_packed_pose
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Dict, Optional, Tuple

from src.utils import (
    timeit,
    write_blueprint,
    write_resfile,
)


def test_protocol(
    packed_pose: PackedPose, **kwargs: Any
) -> Tuple[PackedPose, Dict[str, Any]]:
    """
    Test protocol for PyRosettaCluster.

    Args:
        packed_pose: An optional input `PackedPose` object.

    Keyword Arguments:
        task_id: A required `int` object representing the task number.

    Returns:
        A `tuple` object containing the input `PackedPose` object
        and input keyword arguments.
    """
    client_repr = kwargs["PyRosettaCluster_client_repr"]
    protocol_name = kwargs["PyRosettaCluster_protocol_name"]
    protocol_number = kwargs["PyRosettaCluster_protocol_number"]
    seed = kwargs["PyRosettaCluster_seed"]
    task_id = kwargs["task_id"]

    print(
        f"Task ID: {task_id};",
        f"Protocol name: '{protocol_name}';",
        f"Protocol number: {protocol_number};",
        f"Protocol seed: {seed};",
        f"Client: '{client_repr}';",
        sep=" ",
    )

    return packed_pose, kwargs


@timeit
@requires_packed_pose
def blueprintbdr(
    packed_pose: PackedPose, **kwargs: Any
) -> Optional[PackedPose]:
    """
    A PyRosetta protocol that runs the `BluePrintBDR` mover followed by structure-based scoring.

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A `PackedPose` object with cached scores (total score, RMSD values, and PyRosetta seed),
        or `None` if the `SingleoutputRosettaScriptsTask` execution fails or 'total_score' is >= 0.
    """
    import logging
    import pyrosetta
    import pyrosetta.distributed.io as io
    import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts

    from pathlib import Path
    from pyrosetta.rosetta.core.scoring import (
        all_atom_rmsd_nosuper,
        bb_rmsd,
        bb_rmsd_including_O,
        CA_rmsd,
    )

    protocol_name = kwargs["PyRosettaCluster_protocol_name"]
    protocol_number = kwargs["PyRosettaCluster_protocol_number"]
    seed = kwargs["PyRosettaCluster_seed"]
    client_repr = kwargs["PyRosettaCluster_client_repr"]
    print(
        "Running --",
        f"Protocol name: '{protocol_name}';",
        f"Protocol number: {protocol_number};",
        f"Protocol seed: {seed};",
        f"Client: '{client_repr}';",
        sep=" ",
    )

    # Cache reference PackedPose
    ref_pose = packed_pose.pose.clone()

    # Setup RosettaScripts XML string
    tmp_path = kwargs["PyRosettaCluster_tmp_path"]
    resfile = write_resfile(packed_pose, tmp_path)
    blueprint = write_blueprint(packed_pose, tmp_path)
    xml_file = Path(__file__).parent.parent / "rosetta_scripts" / "blueprintbdr.xml"
    xml_str = xml_file.read_text().format(
        resfile=resfile,
        blueprint=blueprint,
    )

    # Run RosettaScripts
    try:
        src_pose = rosetta_scripts.SingleoutputRosettaScriptsTask(xml_str)(packed_pose).pose
    except Exception as ex:
        logging.error(f"{type(ex).__name__}: Failed to run `SingleoutputRosettaScriptsTask`. {ex}")
        return None

    # Filter
    if src_pose.cache["total_score"] >= 0.0:
        return None

    # Score
    scorefxn = pyrosetta.create_score_function("ref2015")
    total_score = scorefxn(src_pose.clone())
    total_score_res = total_score / src_pose.size()
    src_pose.cache["total_score_ref2015"] = total_score
    src_pose.cache["total_score_res_ref2015"] = total_score_res

    # Superimpose result onto reference
    superimpose_mover = pyrosetta.rosetta.protocols.simple_moves.SuperimposeMover()
    superimpose_mover.set_ca_only(False)
    superimpose_mover.set_reference_pose(ref_pose)
    superimpose_mover.set_target_range(start=1, end=ref_pose.size())
    superimpose_mover.apply(src_pose)

    # Compute RMSD metrics
    src_pose.cache.update(
        all_atom_rmsd_nosuper=all_atom_rmsd_nosuper(ref_pose, src_pose),
        bb_rmsd=bb_rmsd(ref_pose, src_pose),
        bb_rmsd_including_O=bb_rmsd_including_O(ref_pose, src_pose),
        ca_rmsd=CA_rmsd(ref_pose, src_pose),
    )

    # Cache seed
    src_pose.cache["seed"] = float(seed)

    return io.to_packed(src_pose)
