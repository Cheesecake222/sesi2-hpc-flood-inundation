#!/bin/bash
# Uji strong scaling di satu node: 1,2,4,...,32 proses. Sesuaikan header seperti run_flood.sbatch.
#SBATCH --job-name=floodsim-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --partition=compute         # <- ganti
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
module purge
module load python mpi              # <- ganti
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Masalah cukup besar agar komputasi mengalahkan komunikasi (ubah sesuai memori).
python scripts/benchmark_scaling.py --config configs/demo.toml \
    --procs 1 2 4 8 16 32 --launcher mpirun --repeat 2 \
    --set terrain.ny=640 --set terrain.nx=960 --set run.duration_h=0.5 \
    --out outputs/scaling_${SLURM_JOB_ID}
