## Using this package on a Slurm-based system

The two batch files here are examples of how to run the full workflow on a Slurm-based high-performance computing system.

- `sbatch apptainer-workflow.slurm` will build an apptainer container and run the workflow using that container.  This is the preferred method if apptainer is available.
- `sbatch local-workflow.slurm` will run the workflow using the local package and python packages installed using `uv`.

You may need to modify the batch files to match the configuration of your system.