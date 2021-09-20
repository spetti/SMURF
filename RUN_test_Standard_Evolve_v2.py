#! /usr/bin/python
import subprocess
import os
from subprocess import call

date = f"9_20"

n_min = 5
n_max = 7
out_dir = "results_v2"

for choice in ['EvolveMRF_0.05_highest_MM_yes_0.3_0.95', 'EvolveMRF_0.05_highest_MM_no_0.3_0.9','Gremlin_0.05_64_G4_no_0.0_0.0']:
    
    mode, lr, batch_size, iters, gradual, msa_frac, msa_memory = choice.split('_')
    ext=f"{date}_{mode}_{gradual}_{n_min}_{n_max}"
    batchfileName = f"/tmp/{ext}"
    batchfile = open(batchfileName, "w")
    batchfile.write("#!/bin/bash \n")
    batchfile.write("#SBATCH -c 8 \n")
    #batchfile.write("#SBATCH -N 2 \n")
    batchfile.write("#SBATCH -t 60:00 \n")
    batchfile.write("#SBATCH -p eddy_gpu \n")
    batchfile.write("#SBATCH --mem=32000 \n")
    batchfile.write("#SBATCH --gres=gpu:1 \n")
    batchfile.write("#SBATCH --nice=100000000 \n")
    batchfile.write(f"#SBATCH -o ./slurm/out.{ext} \n")
    batchfile.write(f"#SBATCH -e ./slurm/err.{ext} \n")
    #batchfile.write("#SBATCH --mail-type=END \n")
    #batchfile.write("#SBATCH --exclude holy7b0910 \n")

    batchfile.write("source activate esm \n")
    batchfile.write("module load CUDA/10.0.130 \n")

        #this line changed

    commandstring=f"python test_Standard_Evolve_v2.py {date} {lr} {batch_size} {iters} {gradual} {msa_frac} {msa_memory} {mode} {n_min} {n_max} {out_dir}"

    batchfile.write(commandstring)
    batchfile.write("\n")
    batchfile.close()

    sbatchstring =["sbatch", batchfileName]
    call(sbatchstring)
    os.remove(batchfileName)
