__author__ = "Jason C. Klima"


from pathlib import Path
from pyrosetta.distributed.cluster import requires_packed_pose
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, List, Optional

from src.utils import (
    atom_array_to_packed_pose,
    print_protocol_info,
    pyrosetta_to_torch_seed,
    timeit,
)


@timeit
def rfd3(packed_pose: PackedPose, **kwargs: Any) -> List[PackedPose]:
    """
    A PyRosetta protocol that runs RFdiffusion-3.

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
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from lightning.fabric import seed_everything
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    if packed_pose is not None:
        raise ValueError(
            f"The 'packed_pose' argument parameter must be `None`. Received: {type(packed_pose)}"
        )
    # Print runtime info
    print_protocol_info(**kwargs)
    # Setup seed
    seed_everything(pyrosetta_to_torch_seed(kwargs["PyRosettaCluster_seed"]))
    # Configure RFD3
    config = RFD3InferenceConfig(
        specification={
            "length": kwargs["rfd3_length"],
        },
        diffusion_batch_size=kwargs["rfd3_diffusion_batch_size"],
    )
    # Initialize RFD3 inference engine
    model = RFD3InferenceEngine(**config)
    # Run RFD3
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        results = model.run(
            inputs=None,
            out_dir=None,
            n_batches=1,
        )
    packed_poses = []
    for _example_id, rfd3_outputs in results.items():
        for rfd3_output in rfd3_outputs:
            packed_pose = atom_array_to_packed_pose(rfd3_output.atom_array)
            packed_pose = packed_pose.update_scores(
                rfd3_output_metadata=rfd3_output.metadata,
            )
            packed_poses.append(packed_pose)

    return packed_poses


@timeit
@requires_packed_pose
def proteinmpnn(packed_pose: PackedPose, **kwargs: Any) -> List[PackedPose]:
    """
    A PyRosetta protocol that runs ProteinMPNN.

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        mpnn_packed_pose_key: A required key name for the ProteinMPNN `PackedPose` object.
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A list of `PackedPose` objects.
    """
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # Disable GPU for determinism
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import pyrosetta.distributed.io as io

    from lightning.fabric import seed_everything
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine

    # Print runtime info
    print_protocol_info(**kwargs)
    # Setup seed
    torch_seed = pyrosetta_to_torch_seed(kwargs["PyRosettaCluster_seed"])
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
            "batch_size": kwargs["mpnn_batch_size"],
            "number_of_batches": kwargs["mpnn_number_of_batches"],
            "temperature": kwargs["mpnn_temperature"],
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
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        results = model.run(input_dicts=input_dicts)
    # Parse results
    packed_poses = []
    for mpnn_output in results:
        _packed_pose = atom_array_to_packed_pose(mpnn_output.atom_array)
        _packed_pose = _packed_pose.update_scores(
            {kwargs["mpnn_packed_pose_key"]: packed_pose.clone()},
            mpnn_input_dict=mpnn_output.input_dict,
            mpnn_output_dict=mpnn_output.output_dict,
            mpnn_pdbstring=io.to_pdbstring(packed_pose),
        )
        packed_poses.append(_packed_pose)

    return packed_poses


@timeit
@requires_packed_pose
def rf3(packed_pose: PackedPose, **kwargs: Any) -> PackedPose:
    """
    A PyRosetta protocol that runs RoseTTAFold-3.

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
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import pyrosetta
    import pyrosetta.distributed.io as io
    import toolz

    from lightning.fabric import seed_everything
    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.inference import InferenceInput

    # Print runtime info
    print_protocol_info(**kwargs)
    # Setup seed
    torch_seed = pyrosetta_to_torch_seed(kwargs["PyRosettaCluster_seed"])
    seed_everything(torch_seed)
    # Initialize RF3 inference engine
    engine = RF3InferenceEngine(
        n_recycles=kwargs["rf3_n_recycles"],
        diffusion_batch_size=kwargs["rf3_diffusion_batch_size"],
        num_steps=kwargs["rf3_num_steps"],
        template_noise_scale=1e-5,
        raise_if_missing_msa_for_protein_of_length_n=None,
        compress_outputs=False,
        early_stopping_plddt_threshold=None,
        metrics_cfg="default",
        ckpt_path="rf3",
        seed=torch_seed,
        num_nodes=1,
        devices_per_node=1,
        verbose=True,
    )
    # Dump temporary .pdb file
    tmp_path = Path(kwargs["PyRosettaCluster_tmp_path"])
    tmp_pdb_file = tmp_path / "tmp.pdb"
    io.dump_pdb(packed_pose, str(tmp_pdb_file))
    # Setup RF3 inference inputs
    example_id = "rf3_example_id"
    inputs = InferenceInput.from_cif_path(
        path=tmp_pdb_file,
        example_id=example_id,
        template_selection=None,
        ground_truth_conformer_selection=None,
    )
    # Run RF3 inference
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        results = engine.run(
            inputs=inputs,
            out_dir=None,
            dump_predictions=False,
            dump_trajectories=False,
            one_model_per_file=True,
            annotate_b_factor_with_plddt=True,
            sharding_pattern=None,
            skip_existing=False,
            template_selection=None,
            ground_truth_conformer_selection=None,
            cyclic_chains=[],
        )
    rf3_output = results[example_id][0] # Top ranked prediction
    rf3_packed_pose = atom_array_to_packed_pose(rf3_output.atom_array)

    _reserved = pyrosetta.Pose().cache._reserved
    rf3_packed_pose = rf3_packed_pose.update_scores(
        toolz.keyfilter(lambda k: k not in _reserved, packed_pose.pose.cache),
        toolz.keymap(
            lambda k: f"rf3_{k}",
            toolz.merge(rf3_output.confidences, rf3_output.summary_confidences)
        ),
        rf3_example_id=rf3_output.example_id,
        rf3_sample_idx=rf3_output.sample_idx,
        rf3_seed=rf3_output.seed,
    )

    return rf3_packed_pose
