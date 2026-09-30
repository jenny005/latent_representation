# POD vs. Autoencoder: A Falsifiable Test of Representation Faithfulness

Code accompanying the paper:

> J. Zhan, J. Chen. *How to Choose a Surrogate? A Falsifiable Test of Representation Faithfulness.* Submitted to *Journal of Computational Physics*, 2026.

The study compares a linear latent representation (proper orthogonal decomposition, POD) with a nonlinear one (convolutional autoencoder, AE) at matched latent dimension (n = 8, 16, 32, 64) on single-phase Darcy-flow pressure fields. Two experiments share the same PDE, solver, mesh, initial condition, and protocol, and differ only in whether the production well is fixed (Experiment A) or randomly located (Experiment B).

## Repository structure

```
solver/darcyFoam/          OpenFOAM v2212 solver source (compiled to the darcyFoam binary)
case_template/system/      Base case: blockMeshDict (40 x 40 x 20 mesh), fvSchemes, fvSolution
data_generation/           Per-sample case generation
  generate_samples_v3_fixedwell.py    Experiment A (fixed well)
  generate_samples_v3_reconfirm.py    Experiment B (variable well)
extraction/                OpenFOAM output -> per-sample .npy arrays
analysis/
  pod_vs_ae_eta.py         POD vs. AE faithfulness error (eta) on held-out wells
  plot_singular_values.py  Normalized POD singular-value spectra
slurm/                     Batch scripts used on the LONI cluster
results/                   Reported numbers (Table 1)
```

The generation scripts write `controlDict`, `0/`, and `constant/` for each sample from the base case.

## Requirements

- OpenFOAM v2212
- Python 3 with the packages in `requirements.txt`
- A GPU is recommended for autoencoder training

## Reproducing the results

1. Compile the solver: `cd solver/darcyFoam && wmake`
2. Generate and solve the ensembles (1,000 simulations each): `sbatch slurm/darcyFoam_v3_fixedwell.sbatch` (Experiment A) and `sbatch slurm/darcyFoam_v3_reconfirm.sbatch` (Experiment B). Edit the path variables at the top of each script for your system.
3. Extract pressure fields: `python extraction/extract_fields_v3.py`
4. Compute eta: `sbatch slurm/ch3_fixedwell_sweep.sbatch` (A) and `sbatch slurm/ch3_full_sweep.sbatch` (B). Both use a well-level split of 850 training / 150 held-out wells at seed 0.
5. Singular-value spectra: `python analysis/plot_singular_values.py`

## Expected results

Relative reconstruction error eta on held-out wells (paper, Table 1):

| n  | A: POD  | A: AE   | B: POD  | B: AE   |
|----|---------|---------|---------|---------|
| 8  | 0.00220 | 0.00217 | 0.06062 | 0.00657 |
| 16 | 0.00173 | 0.00203 | 0.03877 | 0.00588 |
| 32 | 0.00139 | 0.00175 | 0.02205 | 0.00504 |
| 64 | 0.00107 | 0.00158 | 0.01131 | 0.00462 |

Autoencoder results may vary slightly across hardware and library versions.

## Data

The full pressure snapshots (about 12.8 GB per experiment in float32) are not stored here; they can be regenerated with the scripts above.

## Acknowledgements

Computing resources were provided by the Louisiana Optical Network Infrastructure (LONI) under allocation `loni_latent_pde`.

## Contact

Jing Zhan, Louisiana State University, jzhan2@lsu.edu
