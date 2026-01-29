#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH=/home/xj_data/guowenxiang/env/miniconda3/envs/x2a/bin:$PATH
export HF_ENDPOINT="https://hf-mirror.com"

CHECKPOINT_PATH="./ckpt/model_only_last.ckpt"

step_interval=1
text_cfg=6.0
hist_cfg=1.0
num_steps=20
OUTPUT_DIR="./evaluation/results/baseline${text_cfg}_hist${hist_cfg}_step${step_interval}_num_steps${num_steps}_19w_audiocaps_nft"
mkdir -p $OUTPUT_DIR

OUTPUT_JSON_DIR="$OUTPUT_DIR/json"
META_FN='./audiocaps_test.tsv'

TENSOR_MODEL_PARALLEL_SIZE=1
GPUS_PER_NODE=${NPROC_PER_NODE:-2}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr ${MASTER_ADDR:-127.0.0.1}
    --master_port ${MASTER_PORT:-26699}
)

MAIN_ARGS=(
    --load_ckpt_path $CHECKPOINT_PATH
    --output_dir $OUTPUT_DIR
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
    --global-batch-size $GPUS_PER_NODE
    --no-load-optim
    --no-load-rng
    --use-distributed-optimizer
    --config ./dfot_audio_t5.yaml
    --meta_fn $META_FN
    --num_samples_per_prompt 1
    --text_cfg $text_cfg
    --hist_cfg $hist_cfg
    --step_interval $step_interval
    --num_steps $num_steps
)


export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH

torchrun ${DISTRIBUTED_ARGS[@]} ./inference.py "${MAIN_ARGS[@]}"