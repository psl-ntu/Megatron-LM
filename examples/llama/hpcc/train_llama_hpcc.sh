#!/bin/bash
# Training launcher — all config is injected via env vars from train_llama_hpcc.pbs.
# Can also be run standalone: set env vars manually then call this script.

# ============================================================
# Environment
# ============================================================
if [ -z "$SCRATCH" ]; then
    echo "Error: SCRATCH is not set."
    exit 1
fi

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}

# ============================================================
# Config (with standalone-use defaults)
# ============================================================
DTYPE=${DTYPE:-bf16}

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
VIRTUAL_PIPELINE_STAGES=${VIRTUAL_PIPELINE_STAGES:-0}  # layers per virtual stage; 0 = disabled
ENABLE_FSDP=${ENABLE_FSDP:-0}
FSDP_OVERLAP_RS_WITH_P2P=${FSDP_OVERLAP_RS_WITH_P2P:-0}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-128}
NUM_LAYERS=${NUM_LAYERS:-32}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
EMPTY_UNUSED_MEMORY_LEVEL=${EMPTY_UNUSED_MEMORY_LEVEL:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fused}
ENABLE_FP8_ATTN=${ENABLE_FP8_ATTN:-0}

RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-0}
PROFILE=${PROFILE:-0}

TOKENIZER_ARG=${TOKENIZER_ARG:-MOCK}
DATA_ARG=${DATA_ARG:-MOCK}

# ---- Model architecture (defaults = LLaMA-3 8B) ----
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-14336}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-32}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-8}
INIT_METHOD_STD=${INIT_METHOD_STD:-0.0134}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-$SCRATCH/benchmark_cache_llama3_${HIDDEN_SIZE}_${DTYPE}}
mkdir -p "$DATA_CACHE_PATH"

# ============================================================
# Argument assembly
# ============================================================
RDZV_ID=${PBS_JOBID%%.*}
DISTRIBUTED_ARGS=(
    --nnodes=$NUM_NODES
    --nproc_per_node=$GPUS_PER_NODE
    --rdzv_backend=c10d
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
    --rdzv_id=${RDZV_ID:-default}
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $SEQ_LENGTH
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std $INIT_METHOD_STD
    --attention-backend $ATTENTION_BACKEND
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --lr-decay-samples 1949218748
    --lr-warmup-samples 3906252
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4
    --decoupled-min-lr 4.5e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level $EMPTY_UNUSED_MEMORY_LEVEL
    --exit-duration-in-mins 235
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

if [[ $ENABLE_FSDP -gt 0 ]]; then
    TRAINING_ARGS+=(
        --use-megatron-fsdp
        --data-parallel-sharding-strategy optim_grads
        # --ddp-num-buckets 1
        # --ddp-bucket-size 1
    )
    TRAINING_ARGS+=(
        --ckpt-format fsdp_dtensor
    )
else
    TRAINING_ARGS+=(
        --ckpt-format torch_dist
    )
fi

RECOMPUTE_ARGS=()
if (( RECOMPUTE_LAYERS > 0 )); then
    RECOMPUTE_ARGS=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers $RECOMPUTE_LAYERS
    )
fi


DTYPE_ARGS=()
if [[ "$DTYPE" == fp8 ]]; then
    DTYPE_ARGS=(
        --bf16
        --grad-reduce-in-bf16
        --fp8-recipe delayed
        --fp8-format e4m3
        --fp8-amax-compute-algo max
        --fp8-param-gather

    )
    if (( ENABLE_FP8_ATTN == 1 )); then
        DTYPE_ARGS+=(
            --fp8-dot-product-attention
            --fp8-multi-head-attention
        )
    fi
elif [[ "$DTYPE" == fp16 ]]; then
    DTYPE_ARGS=(--fp16)
else
    DTYPE_ARGS=(--bf16 --grad-reduce-in-bf16)
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size $CP_SIZE
)
(( TP_SIZE > 1 )) && MODEL_PARALLEL_ARGS+=(--sequence-parallel)
(( VIRTUAL_PIPELINE_STAGES > 0 )) && MODEL_PARALLEL_ARGS+=(--num-layers-per-virtual-pipeline-stage $VIRTUAL_PIPELINE_STAGES)

if [[ "$TOKENIZER_ARG" == MOCK || "$DATA_ARG" == MOCK ]]; then
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 128256
        --tiktoken-pattern v2
        --data-cache-path $DATA_CACHE_PATH
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
else
    DATA_ARGS=(
        --data-path $DATA_ARG
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model $TOKENIZER_ARG
        --vocab-size 128256
        --data-cache-path $DATA_CACHE_PATH
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
fi

LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 0
    --eval-interval 100
    --log-throughput
    --distributed-timeout-minutes 60
)

PROFILE_ARGS=()
if (( PROFILE == 1 )); then
    PROFILE_ARGS=(
        --profile
        --profile-step-start 4
        --profile-step-end 6
    )
fi


# ============================================================
# Launch
# ============================================================
if [ ! -f pretrain_gpt.py ]; then
    echo "Error: pretrain_gpt.py not found. Run from the Megatron-LM root."
    exit 1
fi

echo "[train.sh] $(hostname): rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT nnodes=$NUM_NODES nproc=$GPUS_PER_NODE"
torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${PROFILE_ARGS[@]}
