__author__ = "Jason C. Klima"


from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Dict, Tuple


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
    import pyrosetta

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
