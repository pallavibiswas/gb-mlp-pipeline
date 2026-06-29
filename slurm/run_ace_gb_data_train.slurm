#!/bin/bash
#SBATCH --job-name=ace_gb_train
#SBATCH --account=sills_2
#SBATCH --partition=SOE_sills
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/volume/NFS/pb685/project_inventory/repo_draft/logs/%x_%j.out
#SBATCH --error=/volume/NFS/pb685/project_inventory/repo_draft/logs/%x_%j.err

set -euo pipefail
cd /volume/NFS/pb685/project_inventory/repo_draft

echo "ACE train started on $(hostname) at $(date)"

rm -rf /volume/NFS/pb685/ACE/runs2/gb_data_original_train_ace

bash scripts/batch_gb_data_train_to_ace_direct.sh

echo "Final train ACE count:"
find /volume/NFS/pb685/ACE/runs2/gb_data_original_train_ace \
  -name "pace_row_output.txt" | wc -l

echo "ACE train finished at $(date)"
