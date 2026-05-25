import os
import re

# Define paths
base_dir = os.path.abspath(".")
data_dir = os.path.join(base_dir, "sample_data")
template_input = os.path.join(base_dir, "in.template")
output_dir = os.path.join(base_dir, "runs2")
potential_file = os.path.join(base_dir, "output_potential.yace")
lmp_exec = os.path.join(base_dir, "lmp")

os.makedirs(output_dir, exist_ok=True)

# List all .data files
data_files = sorted([
    os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir)
    for f in filenames if f.endswith(".data")
])

print(f"Found {len(data_files)} data files.")

for data_file in data_files:
    print(f"Processing file: {data_file}")
    
    filename = os.path.basename(data_file)
    
    # ?????? T ? seed
    match = re.search(r'T(\d+)_seed(\d+)', filename)
    if match:
        T = match.group(1)
        seed = match.group(2)
        print(f"  Extracted T={T}, seed={seed}")
    else:
        print(f"  Filename format not recognized for: {filename}, skipping...")
        continue  # ??????
    
    run_dir = os.path.join(output_dir, f"run_T{T}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Created run directory: {run_dir}")

    # Modify input script (replace placeholder with actual data filename)
    with open(template_input) as fin:
        text = fin.read().replace("structure.data", "input.data")
    with open(os.path.join(run_dir, "in.run"), "w") as fout:
        fout.write(text)
    print(f"  Wrote modified input script to in.run")

    # Write SLURM job script
    job_script = os.path.join(run_dir, "job.slurm")
    with open(job_script, "w") as f:
        f.write(f"""#!/bin/bash -l
#SBATCH --job-name=run_T{T}_seed{seed}
#SBATCH --partition=SOE_main
#SBATCH -t 02:00:00
#SBATCH --mem=10000
#SBATCH -n 10
#SBATCH -c 1

# Move to working directory
cd {run_dir}

# Copy necessary files
cp {data_file} ./input.data
cp {template_input} ./in.template
cp {potential_file} ./
cp {lmp_exec} ./lmp_exec

# Load MPI module
module load mpi/openmpi_4.1.4-gnu

# Run LAMMPS
# mpirun -mca btl '^openib,tcp' -x UCX_NET_DEVICES=mlx5_0:1 ./lmp_exec -in in.run
mpirun --mca btl vader,self -host $(hostname) ./lmp_exec -in in.run
# -host $(hostname) pins all ranks to this node, so there’s no risk of cross-node placement.
""")
    print(f"  Wrote SLURM script to job.slurm")

    # Submit job
    os.chdir(run_dir)
    submit_status = os.system("sbatch job.slurm")
    if submit_status == 0:
        print(f"  Submitted job for T={T} seed={seed}")
    else:
        print(f"  Failed to submit job for T={T} seed={seed}")
