#!/bin/bash
# Template SLURM untuk menjalankan floodsim dengan MPI.
# SESUAIKAN: partition, account, modul, dan lingkungan Python dengan klaster Anda.
#
#   sbatch slurm/run_flood.sbatch
#
#SBATCH --job-name=floodsim
#SBATCH --nodes=1
#SBATCH --ntasks=32                 # jumlah proses MPI (<= jumlah baris grid)
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=compute         # <- ganti
##SBATCH --account=PROJECT_ID       # <- aktifkan bila diwajibkan
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# --- lingkungan (contoh; nama modul berbeda-beda di tiap klaster) ---
module purge
module load python mpi              # <- ganti sesuai `module avail`
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt mpi4py   # sekali saja, di node login

export OMP_NUM_THREADS=1            # satu thread per proses MPI; NumPy tidak berebut core
export OPENBLAS_NUM_THREADS=1

CONFIG=${CONFIG:-configs/demo.toml}
OUT=outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}

srun python -m floodsim run --config "$CONFIG" --backend mpi --outdir "$OUT" \
     --set run.snapshot_interval_min=0

echo "Selesai. Gambar bisa dibuat di node login:  python -m floodsim plot --run $OUT --no-gif"
