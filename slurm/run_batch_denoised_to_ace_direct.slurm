#!/bin/bash
#SBATCH --job-name=ace_batch_direct
#SBATCH --account=sills
#SBATCH --partition=SOE_sills
#SBATCH --output=ace_batch_direct_%j.out
#SBATCH --error=ace_batch_direct_%j.err
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

cd /volume/NFS/pb685/graphite
export PATH=$HOME/.local/bin:$PATH

bash ~/graphite/batch_denoised_to_ace_direct.sh
