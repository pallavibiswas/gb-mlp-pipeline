#!/bin/bash
#SBATCH --job-name=denoise_gb_train
#SBATCH --account=sills_2
#SBATCH --partition=SOE_sills
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/volume/NFS/pb685/project_inventory/repo_draft/logs/%x_%j.out
#SBATCH --error=/volume/NFS/pb685/project_inventory/repo_draft/logs/%x_%j.err

set -euo pipefail

cd /volume/NFS/pb685/project_inventory/repo_draft

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Help Python find the Graphite denoiser modules
export PYTHONPATH="$HOME/graphite:$HOME/graphite/notebooks/denoiser/lit:${PYTHONPATH:-}"

echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "Python: $(which python3)"

rm -rf /volume/NFS/pb685/MD_GB_sims/gb_data_original_train_denoised

python3 -u scripts/denoise_gb_data.py \
  --input-root /volume/NFS/pb685/MD_GB_sims/gb_data/train/GBALTemp0_1 \
  --output-root /volume/NFS/pb685/MD_GB_sims/gb_data_original_train_denoised \
  --pattern "dump.gb.*" \
  --force-symbol H \
  --overwrite

echo "Final denoised count:"
find /volume/NFS/pb685/MD_GB_sims/gb_data_original_train_denoised \
  -name "*.denoised_.extxyz" | wc -l

echo "Job finished at $(date)"
