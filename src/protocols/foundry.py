__author__ = "Jason C. Klima"


import pyrosetta.distributed.io as io

from pathlib import Path
from pyrosetta.distributed.cluster import requires_packed_pose
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Optional

from src.utils import (
    atom_array_to_packed_pose,
    pyrosetta_to_torch_seed,
    timeit,
)


@timeit
def rfd3(packed_pose: PackedPose, **kwargs: Any) -> Optional[PackedPose]:
    """
    A PyRosetta protocol that runs the RFdiffusion-3. 

    Args:
        packed_pose: A required `None` object.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A `PackedPose` object.
    """
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # Disable GPU for determinism
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import pyrosetta
    pyrosetta.secure_unpickle.add_secure_package("pandas")
    pyrosetta.secure_unpickle.add_secure_package("biotite")

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
    cuda_is_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_is_available else None
    print(
        "Running --",
        f"Protocol name: '{protocol_name}';",
        f"Protocol number: {protocol_number};",
        f"Protocol seed: {seed};",
        f"Client: '{client_repr}';",
        f"CUDA is available: {cuda_is_available};",
        f"CUDA device count: {cuda_device_count};",
        f"CUDA device name: {cuda_device_name};",
        sep=" ",
    )
    # Setup seed
    seed_everything(pyrosetta_to_torch_seed(seed))
    # Configure RFD3
    config = RFD3InferenceConfig(
        specification={
            "length": 20,
        },
        diffusion_batch_size=1,
    )
    # Initialize RFD3 inference engine
    model = RFD3InferenceEngine(**config)
    # Run RFD3
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        results = model.run(
            inputs=None,
            out_dir=None,
            n_batches=1,
        )
    packed_poses = []
    for _example_id, rfd3_outputs in results.items():
        for rfd3_output in rfd3_outputs:
            atom_array = rfd3_output.atom_array
            metadata = rfd3_output.metadata
            packed_pose = atom_array_to_packed_pose(atom_array)
            packed_pose = packed_pose.update_scores(
                # rdf3_atom_array=atom_array,
                rfd3_output_metadata=metadata,
            )
            packed_poses.append(packed_pose)

    return packed_poses


@timeit
@requires_packed_pose
def proteinmpnn(packed_pose: PackedPose, **kwargs: Any) -> Optional[PackedPose]:
    """
    A PyRosetta protocol that runs the ProteinMPNN.

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A `PackedPose` object.
    """
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # Disable GPU for determinism
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import pyrosetta
    pyrosetta.secure_unpickle.add_secure_package("pandas")
    pyrosetta.secure_unpickle.add_secure_package("biotite")

    from lightning.fabric import seed_everything
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine

    protocol_name = kwargs["PyRosettaCluster_protocol_name"]
    protocol_number = kwargs["PyRosettaCluster_protocol_number"]
    seed = kwargs["PyRosettaCluster_seed"]
    client_repr = kwargs["PyRosettaCluster_client_repr"]
    cuda_is_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_is_available else None
    print(
        "Running --",
        f"Protocol name: '{protocol_name}';",
        f"Protocol number: {protocol_number};",
        f"Protocol seed: {seed};",
        f"Client: '{client_repr}';",
        f"CUDA is available: {cuda_is_available};",
        f"CUDA device count: {cuda_device_count};",
        f"CUDA device name: {cuda_device_name};",
        sep=" ",
    )
    # Setup seed
    torch_seed = pyrosetta_to_torch_seed(seed)
    seed_everything(torch_seed)
    # Configure MPNNInferenceEngine
    config = {
        "model_type": "protein_mpnn",
        "is_legacy_weights": True,
        "out_directory": None,
        "write_structures": False,
        "write_fasta": False,
    }
    # Configure per-input inference
    structure_path = Path(kwargs["PyRosettaCluster_tmp_path"]) / "tmp.pdb"
    structure_path.write_text(io.to_pdbstring(packed_pose))
    input_dicts = [
        {
            "structure_path": str(structure_path),
            "batch_size": 8,
            "number_of_batches": 1,
            "temperature": 0.05,
            "omit": ["CYS", "UNK"],
            "structure_noise": 0.0,
            "decode_type": "auto_regressive",
            "causality_pattern": "auto_regressive",
            "remove_waters": True,
            "seed": torch_seed,
        }
    ]
    # Run ProteinMPNN
    model = MPNNInferenceEngine(**config)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        results = model.run(input_dicts=input_dicts)
    # Parse results
    packed_poses = []
    for i, mpnn_output in enumerate(results):
        atom_array = mpnn_output.atom_array
        input_dict = mpnn_output.input_dict
        output_dict = mpnn_output.output_dict
        packed_pose = atom_array_to_packed_pose(atom_array)
        packed_pose = packed_pose.update_scores(
            mpnn_atom_array=atom_array,
            mpnn_input_dict=input_dict,
            mpnn_output_dict=output_dict,
        )
        print(f"MPNN result {i} sequence:", packed_pose.pose.sequence())
        print(f"MPNN result {i} input_dict:", input_dict)
        print(f"MPNN result {i} output_dict:", output_dict)
        packed_poses.append(packed_pose)

    return packed_poses
