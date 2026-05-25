#!/bin/bash
#SBATCH --job-name=evaluate_denoising
#SBATCH --account=sills
#SBATCH --partition=SOE_sills
#SBATCH --output=evaluate_denoising_%j.out
#SBATCH --error=evaluate_denoising_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

cd /volume/NFS/pb685/graphite

export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=""

python3 -u evaluate_denoising.py

