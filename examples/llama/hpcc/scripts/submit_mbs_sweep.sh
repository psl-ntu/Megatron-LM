#!/bin/bash
# MBS sweep on 1 node — measures whether larger MBS improves MFU.
# Keeps GRAD_ACCUMU_STEP fixed at 16 so GBS = MBS * 8 * 16 scales with MBS.
# Compare TFLOP/s/GPU across MBS values; MFU = TFLOP/s/GPU / 989.
#
# Usage: bash submit_mbs_sweep.sh

PBS_SCRIPT=examples/llama/hpcc/train_llama_hpcc.pbs
NODES=1
GPUS_PER_NODE=8
GRAD_ACCUMU_STEP=16   # fixed — isolates MBS effect on MFU

for DTYPE in bf16 fp8; do
    for MBS in 1 2 4 8; do
        GBS=$(( MBS * GPUS_PER_NODE * NODES * GRAD_ACCUMU_STEP ))
        TRAIN_SAMPLES=$(( GBS * 10 ))   # 10 iterations

        # Skip if would OOM: FP8 MBS=8 estimated ~170GB > 141GB
        if [[ "$DTYPE" == "fp8" && "$MBS" -ge 8 ]]; then
            echo "Skipping $DTYPE MBS=$MBS (estimated OOM)"
            continue
        fi
        # BF16 MBS=8 estimated ~125GB — keep but flag
        if [[ "$DTYPE" == "bf16" && "$MBS" -ge 8 ]]; then
            echo "Warning: $DTYPE MBS=$MBS may be tight (~125GB estimated), submitting anyway"
        fi

        echo "Submitting: dtype=$DTYPE MBS=$MBS GBS=$GBS grad_accum=$GRAD_ACCUMU_STEP"
        qsub -l select=${NODES}:ngpus=${GPUS_PER_NODE} \
             -v DTYPE=${DTYPE},MICRO_BATCH_SIZE=${MBS},GLOBAL_BATCH_SIZE=${GBS},TRAIN_SAMPLES=${TRAIN_SAMPLES} \
             ${PBS_SCRIPT}
    done
done
