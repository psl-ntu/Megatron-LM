#!/bin/bash
# Weak scaling: MBS=2, grad_accum=128 fixed for both precisions.
# GBS = MBS * DP * 128 scales with nodes; per-GPU work is constant.
#
# Usage: bash submit_weak_scaling.sh

PBS_SCRIPT=examples/llama/hpcc/train_llama_hpcc.pbs
GPUS_PER_NODE=8
MBS=2
GRAD_ACCUMU_STEP=128

for DTYPE in bf16 fp8; do
    for NODES in 1 2 4 8 16; do
        DP=$(( NODES * GPUS_PER_NODE ))
        GBS=$(( MBS * DP * GRAD_ACCUMU_STEP ))
        TRAIN_SAMPLES=$(( GBS * 10 ))   # 10 iterations

        echo "Submitting: dtype=$DTYPE nodes=$NODES MBS=$MBS GBS=$GBS grad_accum=$GRAD_ACCUMU_STEP"
        qsub -l select=${NODES}:ngpus=${GPUS_PER_NODE} \
             -v DTYPE=${DTYPE},MICRO_BATCH_SIZE=${MBS},GLOBAL_BATCH_SIZE=${GBS},TRAIN_SAMPLES=${TRAIN_SAMPLES} \
             ${PBS_SCRIPT}
    done
done
