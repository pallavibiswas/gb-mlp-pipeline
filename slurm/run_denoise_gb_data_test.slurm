#!/bin/bash
#SBATCH --job-name=denoise_gb_test
#SBATCH --account=sills_2
#SBATCH --partition=SOE_sills
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=/volume/NFS/pb685/project_inventory/repo_draft/logs/%x_%j.out
#SBATCH --error=/volume/NFS/pb685/project_inventory/repo_draft/logs/%x_%j.err

set -euo pipefail

cd /volume/NFS/pb685/project_inventory/repo_draft

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$HOME/graphite:$HOME/graphite/notebooks/denoiser/lit:${PYTHONPATH:-}"

echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "Python: $(which python3)"

rm -rf /volume/NFS/pb685/MD_GB_sims/gb_data_test_denoised

python3 -u scripts/denoise_gb_data.py \
  --input-root /volume/NFS/pb685/MD_GB_sims/gb_data/test \
  --output-root /volume/NFS/pb685/MD_GB_sims/gb_data_test_denoised \
  --pattern "dump.gbstep*.cfg" \
  --force-symbol H \
  --overwrite

echo "Final denoised count:"
find /volume/NFS/pb685/MD_GB_sims/gb_data_test_denoised \
  -name "*.denoised_.extxyz" | wc -l

echo "Job finished at $(date)"
