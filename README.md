[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18157982.svg)](https://doi.org/10.5281/zenodo.19828564)

# _PyRosettaCluster_ Usage Examples

Python code, Jupyter notebooks, _RosettaScripts_ scripts, PyMOL scripts, runtime environment configurations, and output data for "Example #1" and "Example #2" in the preprint publication:

**Jason C. Klima. PyRosettaCluster: a Python framework for scalable and reproducible bio-macromolecular modeling and design. ChemRxiv. 01 May 2026.
DOI: https://doi.org/10.26434/chemrxiv.15002628/v1**

## Jupyter Notebooks
Run the original simulation, then reproduce an output decoy from it, for any example:

| Example | Original | Reproduce |
|:---:|:---:|:---:|
| #1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klimaj/pyrosettacluster-examples/blob/main/notebooks/run_example_1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klimaj/pyrosettacluster-examples/blob/main/notebooks/reproduce_example_1.ipynb) |
| #2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klimaj/pyrosettacluster-examples/blob/main/notebooks/run_example_2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klimaj/pyrosettacluster-examples/blob/main/notebooks/reproduce_example_2.ipynb) |

## Results

### Example #1:
##### Pixi version `0.63.2` was used to create the environment for the original simulation and to recreate the environment for the reproduction simulation.
#### CPU-only simulations:
- Original:
  - [`./results/example_1/original/example-1_ee4e3f706805477b9fd81c8ff3516949.pdb`](results/example_1/original/example-1_ee4e3f706805477b9fd81c8ff3516949.pdb): Original lowest energy output decoy.
  - [`./results/example_1/original/example-1_1a5392a54bad4a86a6ee9721393ef357.pdb`](results/example_1/original/example-1_1a5392a54bad4a86a6ee9721393ef357.pdb): Original highest energy output decoy.
- Reproduced:
  - [`./results/example_1/reproduce/example-1-reproduce_2ccbfc4774024e0bb1e3aebe6bd9fd63.pdb`](results/example_1/reproduce/example-1-reproduce_2ccbfc4774024e0bb1e3aebe6bd9fd63.pdb): Reproduced lowest energy output decoy.

### Example #2:
##### Pixi version `0.63.2` was used to create the environments for the original simulations and to recreate the environments for the reproduction simulations.
#### CPU-only simulations:
- Original:
  - [`./results/example_2/cpu/original/example-2-gpu-0_d906fd986bc14c029cff3e39159e1850.pdb`](results/example_2/cpu/original/example-2-gpu-0_d906fd986bc14c029cff3e39159e1850.pdb): Original lowest scRMSD output decoy.
- Reproduced:
  - [`./results/example_2/cpu/reproduce/example-2-gpu-0_03bc22afcb3b4d2e9d0ad73857b8520c.pdb`](results/example_2/cpu/reproduce/example-2-gpu-0_03bc22afcb3b4d2e9d0ad73857b8520c.pdb): Reproduced lowest scRMSD output decoy.
- Analysis:
  - [`./results/example_2/cpu/original_vs_reproduce_comparison_example-2.json`](results/example_2/cpu/original_vs_reproduce_comparison_example-2.json): Output decoy comparison analysis for original vs. reproduced lowest scRMSD output decoys.

#### GPU-enabled simulations:
- Original:
  - [`./results/example_2/gpu/original/example-2-gpu-1_45d23491d678412ca4e40dd92ba0ef7e.pdb`](results/example_2/gpu/original/example-2-gpu-1_45d23491d678412ca4e40dd92ba0ef7e.pdb): Original lowest scRMSD output decoy.
- Reproduced:
  - [`./results/example_2/gpu/reproduce/example-2-gpu-1_06521badd4f2475b9835723a9cd63373.pdb`](results/example_2/gpu/reproduce/example-2-gpu-1_06521badd4f2475b9835723a9cd63373.pdb): Reproduced lowest scRMSD output decoy.
- Analysis:
  - [`./results/example_2/gpu/original_vs_reproduce_comparison_example-2.json`](results/example_2/gpu/original_vs_reproduce_comparison_example-2.json): Output decoy comparison analysis for original vs. reproduced lowest scRMSD output decoys.

## License
Copyright © 2026 Jason C. Klima. This work is dual-licensed under the following:
  - MIT license for Python code, Jupyter notebooks, _RosettaScripts_ scripts, PyMOL scripts, runtime environment configurations, and JavaScript Object Notation (JSON) files.
  - Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license for Protein Data Bank (PDB) files.

## Citation

If you use this material, please cite the Zenodo software archive:

```
@software{klima_2026_19828564,
  author       = {Klima, Jason C.},
  title        = {PyRosettaCluster: a Python framework for scalable and reproducible bio-macromolecular modeling and design: Supplementary materials},
  month        = {may},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19828564},
  url          = {https://doi.org/10.5281/zenodo.19828564},
}
```

If you reference the associated preprint publication, please cite the ChemRxiv article:

```
@article{klima_2026_15002628,
  author       = {Klima, Jason C.},
  title        = {PyRosettaCluster: a Python framework for scalable and reproducible bio-macromolecular modeling and design},
  journal      = {ChemRxiv},
  volume       = {2026},
  number       = {0501},
  pages        = {},
  year         = {2026},
  doi          = {10.26434/chemrxiv.15002628/v1},
  url          = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15002628/v1},
  eprint       = {https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15002628/v1},
  abstract     = {The Rosetta software suite offers a set of powerful tools for bio-macromolecular modeling and design capable of high-resolution protein structure refinement and generation of de novo designed proteins with sub-angstrom atomic accuracy. Leveraging these tools at scale requires both immense computational resources and domain expertise, without which may lead to results that are not scientifically reproducible by independent investigators, in part due to challenges in communicating all of the necessary runtime configurations and system requirements. To address these reproducibility challenges for large-scale  simulation, we developed PyRosettaCluster, a versatile Python-based framework integrated into PyRosetta software to automatically capture the simulation data required to accurately reproduce a decoy and directly encode it into every output decoy file. PyRosettaCluster abstracts away complicated task execution orchestration details for distributed  workflows, effectively parallelizing modular, user-defined PyRosetta protocols parameterized by user-defined tasks using the lightweight Dask library. PyRosettaCluster supports integration of emerging third-party software applications into PyRosetta protocols, enabling inference of modern graphics processing unit (GPU)-accelerated artificial intelligence (AI) models for bio-macromolecular modeling and design tasks, provided they preserve deterministic runtime behavior for reproducibility purposes. Integration with the Dask-Jobqueue library enables job scheduling on high-performance computing (HPC) or elastic cloud compute infrastructure with arbitrary compute resources. Overall, this multipurpose tool provides a basis for scalable and reproducible bio-macromolecular modeling and design workflows in PyRosetta software. PyRosettaCluster was merged into the Rosetta software suite codebase in September 2020, and its source code has been publicly accessible since March 2024.},
}
```
