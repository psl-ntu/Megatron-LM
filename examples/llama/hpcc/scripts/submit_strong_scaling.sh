#!/bin/bash
# Strong scaling: GBS fixed at 1024 (4.2M tokens), grad_accum shrinks as nodes grow.
#   BF16: MBS=2  → grad_accum = 1024 / (2 * DP)
#   FP8:  MBS=4  → grad_accum = 1024 / (4 * DP)
#
# Usage: bash submit_strong_scaling.sh

PBS_SCRIPT=examples/llama/hpcc/train_llama_hpcc.pbs
GPUS_PER_NODE=8
GBS=1024
TRAIN_SAMPLES=$(( GBS * 10 ))   # 10 iterations

declare -A MBS=([bf16]=2 [fp8]=4)

for DTYPE in bf16 fp8; do
    _MBS=${MBS[$DTYPE]}
    for NODES in 1 2 4 8 16; do
        echo "Submitting: dtype=$DTYPE nodes=$NODES MBS=$_MBS GBS=$GBS"
        qsub -l select=${NODES}:ngpus=${GPUS_PER_NODE} \
             -v DTYPE=${DTYPE},MICRO_BATCH_SIZE=${_MBS},GLOBAL_BATCH_SIZE=${GBS},TRAIN_SAMPLES=${TRAIN_SAMPLES} \
             ${PBS_SCRIPT}
    done
done
