#!/bin/bash
#SBATCH --job-name=mlp_denoised_ace
#SBATCH --account=sills
#SBATCH --partition=SOE_sills
#SBATCH --output=mlp_denoised_ace_%j.out
#SBATCH --error=mlp_denoised_ace_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

cd /volume/NFS/pb685/graphite
export PATH=$HOME/.local/bin:$PATH

python3 -u run_mlp_on_denoised_ace.py
