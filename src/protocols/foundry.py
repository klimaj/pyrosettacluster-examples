__author__ = "Jason C. Klima"


from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Optional

from src.utils import timeit


@timeit
def rfd3(packed_pose: PackedPose, **kwargs: Any) -> Optional[PackedPose]:
    """
    A PyRosetta protocol that runs the RFdiffusion-3. 

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A `PackedPose` object.
    """
    import pyrosetta

    from lightning.fabric import seed_everything
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    if packed_pose is not None:
        raise ValueError(
            f"The 'packed_pose' argument parameter must be `None`. Received: {type(packed_pose)}"
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
    # Setup seed
    torch_seed = seed + (2 ** 31) # PyRosetta seed range: [-(2 ** 31) -> (2 ** 31) - 1]
    seed_everything(torch_seed) # Torch seed range: [0 -> (2 ** 32) - 1]
    # Configure RFD3
    config = RFD3InferenceConfig(
        specification={
            "length": 20,
        },
        diffusion_batch_size=2,
    )
    # Initialize RFD3 inference engine
    model = RFD3InferenceEngine(**config)
    # Run RFD3
    outputs = model.run(
        inputs=None,
        out_dir=None,
        n_batches=1,
    )
    print(outputs)

    return packed_pose
