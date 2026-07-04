# GB Kite–L Transition Classification Pipeline

End-to-end atomistic machine learning pipeline for detecting Kite-to-L grain-boundary transitions using denoising, ACE descriptor extraction, and supervised classification on Rutgers HPC infrastructure.

---

## Research Context

Developed in the **MicroMechanics of Deformation (MoD) Research Group** at Rutgers University under the supervision of **Dr. Ryan Sills**.
This project analyzes real grain-boundary simulation data to identify when a structure transitions from a **Kite-like** state to an **L-like** state across finite-temperature molecular dynamics trajectories.

---

## Pipeline Overview
```text
Raw GB simulation files
        ↓
Denoising
        ↓
ACE/PACE descriptor extraction
        ↓
Structure-level feature construction
        ↓
Kite/L classifier training
        ↓
Prediction on finite-temperature trajectories
        ↓
Transition timing and probability analysis
```

---

## Data

The training data consists of labeled 0 K reference structures:

Class Number of structures
Kite 6
L 45
Total 51

The testing data consists of finite-temperature trajectories at:

150 K, 300 K, 450 K, 600 K, 750 K, 900 K

Across all temperatures, the test set contains 375 denoised trajectory frames.

---

## Model

The current pipeline uses a supervised binary Logistic Regression classifier trained on ACE descriptor features.

Each atomic structure is first represented by per-atom ACE descriptors. These are then aggregated into a structure-level feature vector using descriptor means and standard deviations.

The classification pipeline is:

SimpleImputer
→ StandardScaler
→ LogisticRegression

The classifier predicts:

* Kite/L class label,
* probability of Kite,
* probability of L,
* confidence score.

---

## Core Scripts

scripts/denoise_gb_data.py
scripts/batch_gb_data_train_to_ace_direct.sh
scripts/batch_gb_data_test_to_ace_direct.sh
scripts/train_predict_gb_data_classifier.py

---

## SLURM Jobs

slurm/run_denoise_gb_data_train.slurm
slurm/run_denoise_gb_data_test.slurm
slurm/run_ace_gb_data_train.slurm
slurm/run_ace_gb_data_test.slurm
slurm/run_gb_data_classifier.slurm

---

## Example Workflow

```bash
# Step 1 — Denoise training structures
sbatch slurm/run_denoise_gb_data_train.slurm
# Step 2 — Denoise test trajectories
sbatch slurm/run_denoise_gb_data_test.slurm
# Step 3 — Extract ACE descriptors for training data
sbatch slurm/run_ace_gb_data_train.slurm
# Step 4 — Extract ACE descriptors for test data
sbatch slurm/run_ace_gb_data_test.slurm
# Step 5 — Train classifier and predict transitions
sbatch slurm/run_gb_data_classifier.slurm
```

---

## Results Summary

The trained classifier was applied to finite-temperature trajectories to identify Kite-to-L transitions.

Temperature Transition
150 K No transition observed
300 K 120000 → 123000
450 K 111000 → 114000
600 K 105000 → 108000
750 K 96000 → 99000
900 K 87000 → 90000

The transition occurs earlier at higher temperatures, supporting the interpretation of thermally accelerated Kite-to-L grain-boundary transformation.

---

## Generated Outputs

The pipeline generates:

gb_data_train_predictions.csv
gb_data_test_predictions.csv
transition_summary_by_temperature.csv
gb_data_classifier_summary.txt
predicted_class_heatmap.png
prob_Kite_heatmap.png
prob_Kite_vs_timestep_by_temperature.png
prob_L_vs_timestep_by_temperature.png

Generated outputs, simulation files, ACE output folders, model files, plots, and .extxyz files are excluded from GitHub through .gitignore.

---

## Repository Policy

This repository stores the reusable code pipeline only. Large simulation data and generated artifacts are kept outside version control.

---

## Author

Pallavi Biswas
Undergraduate Researcher — MicroMechanics of Deformation (MoD) Research Group
Rutgers University · Computer Science + Data Science

Research Advisor: Dr. Ryan Sills
