#!/bin/bash
#SBATCH --job-name=gb_classifier
#SBATCH --account=sills_2
#SBATCH --partition=SOE_sills
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/volume/NFS/pb685/project_inventory/repo_draft/logs/%x_%j.out
#SBATCH --error=/volume/NFS/pb685/project_inventory/repo_draft/logs/%x_%j.err

set -euo pipefail
cd /volume/NFS/pb685/project_inventory/repo_draft

echo "Classifier started on $(hostname) at $(date)"

python3 -u scripts/train_predict_gb_data_classifier.py

echo "Classifier finished at $(date)"
