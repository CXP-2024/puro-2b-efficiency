#!/usr/bin/env bash
# Launch a scaled-down Puro model with Puro-Megatron.
#
# The arguments mirror examples/puro/run_puro_2b.sh (phase1-power recipe) in
# the Puro-Megatron repo. What changes for proxy runs is exposed as
# environment variables: model size, batch size, schedule length, precision,
# and the number of GPUs (data parallel only, no pipeline parallelism).
#
# Required: TRAIN_DATA_PATH VALID_DATA_PATH RUN_DIR
# Extra command-line arguments are appended last.
set -euo pipefail

: "${TRAIN_DATA_PATH:?}" "${VALID_DATA_PATH:?}" "${RUN_DIR:?}"
repo_root=${PURO_MEGATRON_ROOT:?source the environment script first}

NUM_LAYERS=${NUM_LAYERS:-12}
HIDDEN=${HIDDEN:-768}
FFN=${FFN:-$((HIDDEN * 3))}          # Puro-2B uses FFN = 3 x hidden
HEADS=${HEADS:-$((HIDDEN / 128))}    # Puro-2B uses head dim 128
QUERY_GROUPS=${QUERY_GROUPS:-$((HEADS / 2))}
SEQ_LEN=${SEQ_LEN:-4096}
MICRO_BATCH=${MICRO_BATCH:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
TRAIN_SAMPLES=${TRAIN_SAMPLES:?number of sequences to train on}
WARMUP_SAMPLES=${WARMUP_SAMPLES:-$((GLOBAL_BATCH * 100))}
LR=${LR:-0.005}
MIN_LR=${MIN_LR:-0.0005}
PRECISION=${PRECISION:-bf16}         # bf16 | fp8
NGPU=${NGPU:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}

mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/tb" "$RUN_DIR/data_cache"

model_args=(
  --use-mcore-models
  --num-layers "$NUM_LAYERS"
  --hidden-size "$HIDDEN"
  --ffn-hidden-size "$FFN"
  --num-attention-heads "$HEADS"
  --group-query-attention
  --num-query-groups "$QUERY_GROUPS"
  --kv-channels 128
  --seq-length "$SEQ_LEN"
  --max-position-embeddings "$SEQ_LEN"
  --position-embedding-type rope
  --rotary-base 10000
  --rotary-percent 1.0
  --attention-dropout 0.0
  --attention-type thd
  --hidden-dropout 0.0
  --swiglu
  --init-method-std 0.018
  --normalization RMSNorm
  --qk-layernorm
  --untie-embeddings-and-output-weights
  --disable-bias-linear
  --attention-softmax-in-fp32
  --accumulate-allreduce-grads-in-fp32
)

optimizer_args=(
  --micro-batch-size "$MICRO_BATCH"
  --global-batch-size "$GLOBAL_BATCH"
  --clip-grad 1.0
  --weight-decay 0.1
  --optimizer muon_hyperball
  --muon-hyperball-scalar-optimizer adam
  --muon-hyperball-eps 1e-12
  --muon-hyperball-lr-mult 10.0
  --muon-qkv-norm-mode separate
  --muon-swiglu-norm-mode separate
  --muon-qkv-ns-mode separate
  --muon-swiglu-ns-mode separate
  --adam-beta1 0.9
  --adam-beta2 0.95
  --bf16
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl te
  --calculate-per-token-loss
  --manual-gc
  --rerun-mode skip
  --use-distributed-optimizer
  --layerwise-optimizer-memory-balance
  --overlap-grad-reduce
  --overlap-param-gather
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
)

if [[ $PRECISION == fp8 ]]; then
  optimizer_args+=(--fp8-format e4m3 --fp8-recipe blockwise)
fi

data_args=(
  --train-data-path "$TRAIN_DATA_PATH"
  --valid-data-path "$VALID_DATA_PATH"
  --tokenizer-type NullTokenizer
  --vocab-size 151936
  --data-cache-path "$RUN_DIR/data_cache"
  --tiktoken-pattern v2
  --eos-token-id 151645
  --no-mmap-bin-files
  --num-workers 0
)

schedule_args=(
  --train-samples "$TRAIN_SAMPLES"
  --lr-warmup-samples "$WARMUP_SAMPLES"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --lr-decay-style power
  --lr-power-decay-samples "$WARMUP_SAMPLES"
  --lr-power-exponent 0.5
)

runtime_args=(
  --log-interval 1
  --tensorboard-log-interval 1
  --eval-iters 5
  --eval-interval "$EVAL_INTERVAL"
  --log-throughput
  --timing-log-level 0
  --ckpt-format torch_dist
  --distributed-timeout-minutes 60
  --tensorboard-dir "$RUN_DIR/tb"
  --load "$RUN_DIR/ckpt"
  --save "$RUN_DIR/ckpt"
  --save-interval "$SAVE_INTERVAL"
  --diag-interval 100
)

exec torchrun --nproc-per-node "$NGPU" "$repo_root/pretrain_gpt.py" \
  "${model_args[@]}" \
  "${optimizer_args[@]}" \
  "${data_args[@]}" \
  "${schedule_args[@]}" \
  "${runtime_args[@]}" \
  "$@"
