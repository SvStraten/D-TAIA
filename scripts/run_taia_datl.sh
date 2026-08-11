#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=mcs.gpu.q
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

# Usage: sbatch run_taia_datl.sh <VARIANT> <DATASET> <FILEPATH> [EPOCHS] [SEEDS] [BACKBONE]
#
# VARIANT: DTAIA | no_datl | no_domain_id | no_faiss | no_taia | FTLLM | MTRNN
# DATASET: bpi2012 | bpi2017 | bpi2015_2 | bpi2020_dd
# EPOCHS  overrides the per-variant default below if given explicitly.
# SEEDS   space-separated seed list, default "1 2 3 4 5". Pass as a single
#         quoted argument, e.g. sbatch run_taia_datl.sh DTAIA bpi2015_2 data.xes "" "1 2 3 4 5"
# BACKBONE: tinyllm (default) | qwen | llama. Ignored for MTRNN (LSTM backbone,
#         no HF model involved).
#
# One job == one (VARIANT, DATASET, BACKBONE) experiment. All seeds for that
# experiment run sequentially inside this single job so the whole set shares
# one allocation, rather than spawning a separate job per seed.

VARIANT=$1
DATASET=$2
FILEPATH=$3
EPOCHS_ARG=$4
SEEDS_ARG=$5
BACKBONE_ARG=$6
DIR="results"
CLEAN_CSV="data_functions/clean_data/${DATASET}_engineered.csv"
SEEDS=${SEEDS_ARG:-"1 2 3 4 5"}
BACKBONE=${BACKBONE_ARG:-"tinyllm"}

case "$BACKBONE" in
"tinyllm") HF_MODEL_NAME="arnir0/Tiny-LLM" ;;
"qwen")    HF_MODEL_NAME="Qwen/Qwen2.5-0.5B" ;;
"llama")   HF_MODEL_NAME="meta-llama/Llama-3.2-1B" ;;
*)
    echo "[ERROR] Unknown backbone: $BACKBONE (choose from: tinyllm | qwen | llama)"
    exit 1 ;;
esac

# Uniform epoch budget for every variant. Previously FTLLM/MTRNN used
# smaller defaults (10/25) and DTAIA used 20; now everyone gets 60 so
# early stopping (see HP below) actually has room to matter and every
# method is compared under the same training budget.
DEFAULT_EPOCHS=60
EPOCHS=${EPOCHS_ARG:-$DEFAULT_EPOCHS}

echo "========================================================"
echo "  Variant  : $VARIANT"
echo "  Dataset  : $DATASET"
echo "  Backbone : $BACKBONE ($HF_MODEL_NAME)"
echo "  Seeds    : $SEEDS"
echo "  Epochs   : $EPOCHS"
echo "========================================================"

mkdir -p "$DIR" logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

SKIP_FLAG=""
if [ -f "$CLEAN_CSV" ]; then
    echo "Found cached $CLEAN_CSV, skipping XES re-parse"
    SKIP_FLAG="--skip-data-prep"
elif [ ! -f "$FILEPATH" ]; then
    echo "[ERROR] Neither $CLEAN_CSV nor $FILEPATH exist"
    exit 1
fi

# ---------------------------------------------------------------
# Single fixed hyperparameter set, applied uniformly across every
# dataset and every method. Replaces the old per-(dataset, variant)
# tuned table after diagnostics showed the previous DTAIA settings
# (finetune-lr=5e-4, early-stopping-patience=10 default) causing
# val_loss to diverge from epoch 1 while train_acc was still rising --
# i.e. the model was cut off before it had a real chance to learn.
#
# finetune-lr=2e-4          : config.py default; 5e-4 was too aggressive
# early-stopping-patience=30: was defaulting to 10, cutting training short
# lstm-hidden-dim/num-layers: only consumed by MTRNN, harmless otherwise
# ---------------------------------------------------------------
HP="--batch-size 32 --finetune-lr 2e-4 --dropout 0.30 --loss-alpha 1.0 --triplet-margin 1.0 --lora-r 8 --lora-alpha 16 --lora-dropout 0.15 --early-stopping-patience 30 --lstm-hidden-dim 128 --lstm-num-layers 2"

case "$VARIANT" in
"DTAIA")         EXTRA_FLAGS="" ;;
"no_datl")       EXTRA_FLAGS="--no-datl" ;;
"no_domain_id")  EXTRA_FLAGS="--no-domain-id" ;;
"no_faiss")      EXTRA_FLAGS="--no-faiss" ;;
"no_taia")       EXTRA_FLAGS="--no-taia" ;;
"FTLLM")         EXTRA_FLAGS="--no-datl --no-taia --no-faiss --oyamada-input" ;;
"MTRNN")         EXTRA_FLAGS="--backbone-lstm --no-datl --no-taia" ;;
*)
    echo "[ERROR] Unknown variant: $VARIANT"
    echo "        Choose from: DTAIA | no_datl | no_domain_id | no_faiss | no_taia | FTLLM | MTRNN"
    exit 1 ;;
esac

# MTRNN has no HF backbone (it's an LSTM) -- --hf-model-name would be a no-op,
# so only pass it, and only include it in filenames, for HF-backed variants.
BACKBONE_FLAG=""
BACKBONE_FILE_TAG=""
if [ "$VARIANT" != "MTRNN" ]; then
    BACKBONE_FLAG="--hf-model-name $HF_MODEL_NAME"
    BACKBONE_FILE_TAG="_${BACKBONE}"
elif [ "$BACKBONE_ARG" != "" ] && [ "$BACKBONE" != "tinyllm" ]; then
    echo "[WARN] --backbone $BACKBONE requested for MTRNN, which has no HF backbone -- ignoring"
fi

echo "  Extra flags: $EXTRA_FLAGS"
echo "  Hyperparams: $HP"
echo ""

OVERALL_EXIT=0

for SEED in $SEEDS; do
    LOG_FILE="logs/${VARIANT}_${DATASET}${BACKBONE_FILE_TAG}_seed${SEED}.log"
    echo "--------------------------------------------------------"
    echo "  Running seed $SEED"
    echo "--------------------------------------------------------"

    python -m d_taia.pipeline \
        --dataset    "$DATASET" \
        --filepath   "$FILEPATH" \
        --seed       "$SEED" \
        --finetune-epochs "$EPOCHS" \
        $EXTRA_FLAGS \
        $BACKBONE_FLAG \
        $HP \
        $SKIP_FLAG \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Seed $SEED failed with exit code $EXIT_CODE"
        OVERALL_EXIT=$EXIT_CODE
    else
        echo "DONE: $VARIANT | $DATASET | Seed $SEED"
    fi
done

if [ $OVERALL_EXIT -ne 0 ]; then
    echo "[ERROR] One or more seeds failed for $VARIANT | $DATASET (last exit code $OVERALL_EXIT)"
    exit $OVERALL_EXIT
fi

echo "DONE: $VARIANT | $DATASET | Seeds: $SEEDS"