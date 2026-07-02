#!/bin/bash
# ----------------SLURM Parameters----------------
# You may need to change the details depending on your cluster configuration. 
# Check with your system administrator if you are unsure.
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --nodes=1
# Define how long the job will run d-hh:mm:ss
#SBATCH --time=01:00:00
# Get email notification when job finishes or fails
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=END,FAIL
# Give your job a name, so you can recognize it in the queue overview
#SBATCH -J apptainer-workflow
# ----------------Load Modules--------------------
# ----------------Commands------------------------
make all
