#!/bin/bash
#SBATCH --job-name=denoise_kitel
#SBATCH --account=sills
#SBATCH --partition=SOE_sills
#SBATCH --gres=gpu:1
#SBATCH --output=denoise_kitel_%j.out
#SBATCH --error=denoise_kitel_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /volume/NFS/pb685/graphite

export PATH=$HOME/.local/bin:$PATH

python3 -u -c "import torch; print('cuda?', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python3 -u denoise_kitel.py
