#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=mcs.gpu.q
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

# Usage: sbatch run_experiments.sh <SWEEP> <DATASET> <FILEPATH> <SEED>
#
# SWEEP: backbone | data-pct | prefix-length
# DATASET: bpi2012 | bpi2017 | bpi2015_2 | bpi2020_dd

SWEEP=$1
DATASET=$2
FILEPATH=$3
SEED=$4
DIR="results"

echo "========================================================"
echo "  Sweep   : $SWEEP"
echo "  Dataset : $DATASET"
echo "  Seed    : $SEED"
echo "========================================================"

mkdir -p "$DIR" logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

case "$SWEEP" in
"backbone"|"data-pct"|"prefix-length")
    ;;
*)
    echo "[ERROR] Unknown sweep: $SWEEP"
    echo "        Choose from: backbone | data-pct | prefix-length"
    exit 1 ;;
esac

LOG_FILE="logs/experiments_${SWEEP}_${DATASET}_seed${SEED}.log"
OUT_FILE="${DIR}/${DATASET}_${SWEEP//-/_}.csv"

python experiments.py "$SWEEP" \
    --dataset  "$DATASET" \
    --filepath "$FILEPATH" \
    --seed     "$SEED" \
    --out      "$OUT_FILE" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "DONE: $SWEEP | $DATASET | Seed $SEED"