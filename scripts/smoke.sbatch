#!/usr/bin/env bash
#SBATCH --job-name=puro-smoke
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#
# Smoke test: import check, then 30 steps of a tiny model in BF16, then the
# same in blockwise FP8 (to learn whether that recipe runs on H100 at all).
set -uo pipefail
source ~/venvs/puro/env.sh
cd ~/projects/puro

echo "=== node $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
python import_test.py 2>&1 | grep -v -i warning

DATA=~/storage/puro/packed/fwedu_smoke
export TRAIN_DATA_PATH=$DATA/train VALID_DATA_PATH=$DATA/valid
export NUM_LAYERS=8 HIDDEN=512 GLOBAL_BATCH=8 MICRO_BATCH=2
export TRAIN_SAMPLES=240 WARMUP_SAMPLES=80 EVAL_INTERVAL=10 NGPU=1

for prec in bf16 fp8; do
  echo "=== smoke $prec"
  RUN_DIR=~/projects/puro/runs/smoke-$prec-$SLURM_JOB_ID PRECISION=$prec \
    bash scripts/run_proxy.sh
  echo "=== smoke $prec exit code $?"
done
