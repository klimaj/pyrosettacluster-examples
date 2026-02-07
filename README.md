# PyRosettaCluster Usage Examples

## Jupyter Notebooks
Run the original simulation, then reproduce an output decoy from it, for any example:

| Example | Original | Reproduce |
|:---:|:---:|:---:|
| #1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klimaj/pyrosettacluster-examples/blob/main/notebooks/run_example_1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klimaj/pyrosettacluster-examples/blob/main/notebooks/reproduce_example_1.ipynb) |
| #2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klimaj/pyrosettacluster-examples/blob/main/notebooks/run_example_2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klimaj/pyrosettacluster-examples/blob/main/notebooks/reproduce_example_2.ipynb) |

## Results

### Example #1:
#### CPU-only simulations:
- Original:
  - `./results/example_1/original/example-1_ee4e3f706805477b9fd81c8ff3516949.pdb.bz2`: Original lowest energy output decoy.
  - `./results/example_1/original/example-1_1a5392a54bad4a86a6ee9721393ef357.pdb.bz2`: Original highest energy output decoy.
- Reproduced:
  - `./results/example_1/reproduce/example-1-reproduce_2ccbfc4774024e0bb1e3aebe6bd9fd63.pdb.bz2`: Reproduced lowest energy output decoy.

### Example #2:

#### CPU-only simulations:
- Original:
  - `./results/example_2/cpu/original/example-2-gpu-0_d906fd986bc14c029cff3e39159e1850.pdb`: Original lowest scRMSD output decoy (PDB format).
  - `./results/example_2/cpu/original/example-2-gpu-0_d906fd986bc14c029cff3e39159e1850.b64_pose`: Original lowest scRMSD output decoy (Base64-encoded pickled Pose format).
- Reproduced:
  - `./results/example_2/cpu/reproduce/example-2-gpu-0_a4d83e975a4c4fdb872c7bf2a6478fe6.pdb`: Reproduced lowest scRMSD output decoy (PDB format).
  - `./results/example_2/cpu/reproduce/example-2-gpu-0_a4d83e975a4c4fdb872c7bf2a6478fe6.b64_pose`: Reproduced lowest scRMSD output decoy (Base64-encoded pickled Pose format).
- Analysis:
  - `./results/example_2/cpu/original_vs_reproduce_comparison_example-2.json`: Output decoy comparison analysis for original vs. reproduced lowest scRMSD output decoys.

#### GPU-enabled simulations:
- Original:
  - `./results/example_2/gpu/original/example-2-gpu-1_45d23491d678412ca4e40dd92ba0ef7e.pdb`: Original lowest scRMSD output decoy (PDB format).
  - `./results/example_2/gpu/original/example-2-gpu-1_45d23491d678412ca4e40dd92ba0ef7e.b64_pose`: Original lowest scRMSD output decoy (Base64-encoded pickled Pose format).
- Reproduced:
  - `./results/example_2/gpu/reproduce/example-2-gpu-1_8e4bed62169c4909a2166a25621741ec.pdb`: Reproduced lowest scRMSD output decoy (PDB format).
  - `./results/example_2/gpu/reproduce/example-2-gpu-1_8e4bed62169c4909a2166a25621741ec.b64_pose`: Reproduced lowest scRMSD output decoy (Base64-encoded pickled Pose format).
- Analysis:
  - `./results/example_2/gpu/original_vs_reproduce_comparison_example-2.json`: Output decoy comparison analysis for original vs. reproduced lowest scRMSD output decoys.

## License
Copyright © 2026 Jason C. Klima. All rights reserved.
