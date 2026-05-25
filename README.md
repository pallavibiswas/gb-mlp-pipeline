# GB-MLP Pipeline

End-to-end atomistic machine learning pipeline for grain-boundary structure denoising, ACE descriptor extraction, and MLP-based structural classification on HPC infrastructure.

---

## Research Context

Developed in the **MicroMechanics of Deformation (MoD) Research Group** at Rutgers University under the supervision of **Dr. Ryan Sills**.

This project focuses on atomistic machine learning workflows for grain-boundary structure analysis, denoising, and classification using graph neural networks and ACE descriptors on HPC systems.

---

## Overview

This pipeline implements a full end-to-end workflow for:

1. Generating synthetic noisy atomic structures
2. Training a Graph Neural Network denoiser
3. Denoising perturbed grain-boundary structures
4. Extracting ACE/PACE descriptors via LAMMPS
5. Performing PCA dimensionality reduction
6. Training MLP classifiers for structure discrimination
7. Evaluating denoising and classification performance

The pipeline was developed for distinguishing between **Kite** and **L** grain-boundary structures using atomistic simulation data.

---

## Pipeline

```
Clean Structures
        ↓
Synthetic Noise Generation
        ↓
Graph Neural Network Denoising
        ↓
Denoised Atomic Structures
        ↓
LAMMPS ACE/PACE Feature Extraction
        ↓
ACE Descriptor Matrix
        ↓
PCA Dimensionality Reduction
        ↓
MLP Classification
        ↓
Evaluation + Visualization
```

---

## Repository Structure

```
scripts/        Final working pipeline scripts
slurm/          SLURM batch job scripts
ace_utils/      OVITO/LAMMPS helper utilities
legacy/         Older experimental/reference scripts
results/        Example plots and evaluation outputs
docs/           Documentation assets
data/sample/    Small sample files only
```

---

## Core Pipeline Scripts

### 1. Synthetic Structure Generation — `generate_structural_noise.py`

Creates synthetic noisy grain-boundary structures by adding Gaussian perturbations to atomic coordinates.

- Multiple noise levels (σ = 0.01, 0.02, 0.03, 0.05)
- Multiple random replicates
- ASE-based structure handling
- `.extxyz` export

---

### 2. Denoiser Training — `train_kitel.py`

Trains a Graph Neural Network denoiser using the Graphite framework.

Outputs: Lightning checkpoints, training logs, and the learned denoising model.

---

### 3. Denoising Inference — `denoise_kitel.py`

Applies the trained denoiser to synthetic noisy structures and outputs denoised `.extxyz` files.

---

### 4. Denoising Evaluation

**`evaluate_denoising.py`** — Compares original, noisy, and denoised structures using mean and RMS displacement error.

**`plot_denoising_results.py`** — Generates denoising comparison visualizations.

---

### 5. ACE/PACE Descriptor Extraction

**`batch_denoised_to_ace_direct.sh`** — Batch processes all denoised structures through OVITO and LAMMPS to extract PACE descriptors, then archives results.

**`test_one_ace_direct.sh`** — Single-file ACE extraction for debugging and testing.

---

### 6. PCA + MLP Classification — `run_mlp_on_denoised_ace.py`

- Assembles ACE dataset and standardizes features
- Applies PCA dimensionality reduction
- Trains and evaluates MLP classifier
- Generates confusion matrix

**Final performance:** Test Accuracy = **1.0000** | PCA components retained = **3**

---

## Example Workflow

```bash
# Step 1 — Generate synthetic structures
python scripts/generate_structural_noise.py

# Step 2 — Train denoiser
sbatch slurm/run_train_kitel.slurm

# Step 3 — Run denoising
sbatch slurm/run_denoise_kitel.slurm

# Step 4 — Evaluate denoising
sbatch slurm/run_evaluate_denoising.slurm

# Step 5 — Extract ACE features
sbatch slurm/run_batch_denoised_to_ace_direct.slurm

# Step 6 — Run PCA + MLP classification
sbatch slurm/run_mlp_on_denoised_ace.slurm
```

---

## Results

**Denoising:** Significant reduction in displacement noise with preservation of grain-boundary structural features.

**Classification:** Perfect separation between Kite and L structures; ACE descriptor space reduced to 3 principal components via PCA.

---

## HPC Environment

Developed on Rutgers HPC infrastructure (Soemaster, Amarel).

| Category | Tools |
|---|---|
| Languages | Python |
| Atomistic Simulation | ASE, LAMMPS, OVITO |
| ML Frameworks | PyTorch, PyTorch Lightning, Scikit-learn |
| Job Scheduling | SLURM |

---

## Legacy Scripts

The `legacy/` directory contains earlier experiments, feature-space synthetic augmentation tests, deprecated ACE workflows, and older classification experiments. These are preserved for reproducibility but are not part of the final pipeline.

---

## Notes

Large datasets, checkpoints, and generated outputs are excluded from this repository via `.gitignore`. This repo focuses on reproducibility, pipeline organization, HPC workflow automation, and scientific ML methodology.

---

## Author

**Pallavi Biswas**  
Undergraduate Researcher — MicroMechanics of Deformation (MoD) Research Group  
Rutgers University · Computer Science + Data Science  

**Research Advisor:** Dr. Ryan Sills
