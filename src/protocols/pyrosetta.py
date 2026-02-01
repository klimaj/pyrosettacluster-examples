__author__ = "Jason C. Klima"


from pathlib import Path
from pyrosetta.distributed.cluster import requires_packed_pose
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Dict, Optional, Tuple

from src.utils import (
    timeit,
    print_protocol_info,
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


def run_xml_file(packed_pose: PackedPose, xml_file: Path) -> PackedPose:
    """
    The the provided RosettaScripts XML file on the provided `PackedPose` object.

    Args:
        packed_pose: A required input `PackedPose` object.
        xml_file: A required RosettaScripts XML file path or `Path` object.

    Returns:
        A `PackedPose` object.
    """
    import pyrosetta.distributed.io as io

    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    xml_obj = XmlObjects.create_from_file(str(xml_file)).get_mover("ParsedProtocol")
    pose = packed_pose.pose
    xml_obj.apply(pose)

    return io.to_packed(pose)


@timeit
@requires_packed_pose
def cst_cart_min_poly_gly(
    packed_pose: PackedPose, **kwargs: Any
) -> PackedPose:
    """
    A PyRosetta protocol that converts the input `PackedPose` object to poly-glycine,
    and minimizes with C-alpha coordinate constraints.

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A `PackedPose` object.
    """
    from pathlib import Path

    # Print runtime info
    print_protocol_info(**kwargs)
    # Run RosettaScripts
    xml_file = Path(__file__).parent.parent / "rosetta_scripts" / "cst_cart_min_poly_gly.xml"

    return run_xml_file(packed_pose, xml_file)


@timeit
@requires_packed_pose
def cart_min(
    packed_pose: PackedPose, **kwargs: Any
) -> PackedPose:
    """
    A PyRosetta protocol that performs Cartesian minimization on the input `PackedPose` object.

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A `PackedPose` object.
    """
    from pathlib import Path

    # Print runtime info
    print_protocol_info(**kwargs)
    # Run RosettaScripts
    xml_file = Path(__file__).parent.parent / "rosetta_scripts" / "cart_min.xml"

    return run_xml_file(packed_pose, xml_file)


@timeit
@requires_packed_pose
def compute_rmsd(
    packed_pose: PackedPose, **kwargs: Any
) -> PackedPose:
    """
    A PyRosetta protocol that performs C-alpha superposition and computes the backbone
    heavy atom root-mean-squared deviation (RMSD) between the input `PackedPose` and a
    reference `PackedPose` object given by the 'mpnn_packed_pose_key' keyword argument
    parameter in the input `PackedPose.pose.cache` dictionary, which is cleared from
    the `PackedPose.pose.cache` dictionary at the end of the protocol.

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        mpnn_packed_pose_key: A required key name for the ProteinMPNN `PackedPose` object.
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A `PackedPose` object.
    """
    import pyrosetta
    import pyrosetta.distributed.io as io

    # Print runtime info
    print_protocol_info(**kwargs)
    # Setup protocol
    src_pose = packed_pose.pose
    ref_pose = packed_pose.pose.cache[kwargs["mpnn_packed_pose_key"]].pose
    # Superimpose input onto reference
    superimpose_mover = pyrosetta.rosetta.protocols.simple_moves.SuperimposeMover()
    superimpose_mover.set_ca_only(True)
    superimpose_mover.set_reference_pose(ref_pose)
    superimpose_mover.set_target_range(start=1, end=ref_pose.size())
    superimpose_mover.apply(src_pose)
    # Compute RMSD
    bb_rmsd = pyrosetta.rosetta.core.scoring.bb_rmsd_including_O(src_pose, ref_pose)
    # Cache RMSD & sequence
    packed_pose = packed_pose.update_scores(
        bb_rmsd=bb_rmsd,
        sequence=packed_pose.pose.sequence(),
    )
    # Clear reference `PackedPose` object from `Pose.cache`
    pose = packed_pose.pose
    pose.cache.pop(kwargs["mpnn_packed_pose_key"])
    packed_pose = io.to_packed(pose)

    return packed_pose
