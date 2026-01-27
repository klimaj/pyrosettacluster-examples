__author__ = "Jason C. Klima"


from pathlib import Path
from pyrosetta.distributed.cluster import requires_packed_pose
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import Any, Optional

from src.utils import (
    atom_array_to_packed_pose,
    print_protocol_info,
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
    # Print runtime info
    print_protocol_info(**kwargs)
    # Setup seed
    seed_everything(pyrosetta_to_torch_seed(kwargs["PyRosettaCluster_seed"]))
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
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
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
    A PyRosetta protocol that runs ProteinMPNN.

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A list of `PackedPose` objects.
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
    import pyrosetta.distributed.io as io
    pyrosetta.secure_unpickle.add_secure_package("pandas")
    pyrosetta.secure_unpickle.add_secure_package("biotite")

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
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
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


@timeit
@requires_packed_pose
def rf3(packed_pose: PackedPose, **kwargs: Any) -> Optional[PackedPose]:
    """
    A PyRosetta protocol that runs RoseTTAFold-3.

    Args:
        packed_pose: A required input `PackedPose` object.

    Keyword Args:
        PyRosettaCluster_*: Default `PyRosettaCluster` keyword arguments.

    Returns:
        A list of `PackedPose` object.
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
    import pyrosetta.distributed.io as io
    pyrosetta.secure_unpickle.add_secure_package("pandas")
    pyrosetta.secure_unpickle.add_secure_package("biotite")

    from io import StringIO
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
        n_recycles=5,
        diffusion_batch_size=5,
        num_steps=50,
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
    inputs = InferenceInput.from_cif_path(
        path=tmp_pdb_file,
        example_id=None,
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
    packed_poses = []
    for _example_id, rf3_outputs in results.items():
        print("Example ID:", _example_id)
        for i, rf3_output in enumerate(rf3_outputs):
            example_id = rf3_output.example_id
            atom_array = rf3_output.atom_array
            summary_confidences = rf3_output.summary_confidences
            confidences = rf3_output.confidences
            sample_idx =rf3_output.sample_idx
            seed = rf3_output.seed
            packed_pose = atom_array_to_packed_pose(atom_array)
            print(f"RF3 output {i}:", example_id)
            print(f"RF3 output {i}:", atom_array)
            print(f"RF3 output {i}:", summary_confidences)
            print(f"RF3 output {i}:", confidences)
            print(f"RF3 output {i}:", sample_idx)
            print(f"RF3 output {i}:", seed)
            packed_poses.append(packed_pose)

    return packed_poses
