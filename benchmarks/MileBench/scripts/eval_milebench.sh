#!/bin/bash

MODEL_ID=$1
MODEL_DIR=$2
path_to_all_results=$3
patchStrategy="norm"
if [ $# -ge 4 ]; then
    patchStrategy=$4
fi

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
read -a GPULIST <<< ${gpu_list//,/ }
CHUNKS=${#GPULIST[@]}

BASE_DIR=./benchmarks/MileBench
GEN_SCRIPT_PATH=$BASE_DIR/scripts/generate_milebench.py
EVAL_SCRIPT_PATH=$BASE_DIR/scripts/evaluate_milebench.py
DATA_DIR=$BASE_DIR/data/MLBench
MODEL_CONFIG_PATH=$BASE_DIR/configs/model_configs.yaml


dataset_paths=$(find $DATA_DIR -maxdepth 1 -mindepth 1 -type d)

dataset_names=""

for dataset_path in $dataset_paths; do
    dataset_name=$(basename $dataset_path)
    dataset_names="${dataset_names}${dataset_name},"
done

dataset_names=${dataset_names%,}

BATCH_SIZE=1
mkdir -p logs/${model}

# Start generating
for IDX in $(seq 0 $((CHUNKS-1))); do
CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ${GEN_SCRIPT_PATH} \
    --data_dir ${DATA_DIR} \
    --dataset_name ${dataset_names}  \
    --model_name ${MODEL_ID} \
    --model_path ${MODEL_DIR} \
    --output_dir $BASE_DIR/answers \
    --batch-image ${BATCH_SIZE} \
    --model_configs ${MODEL_CONFIG_PATH} \
    --patchStrategy $patchStrategy \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --overwrite &
    # > logs/${model}/${dataset_name}.log
done

wait


dataset_names=$(find $DATA_DIR -maxdepth 1 -mindepth 1 -type d)

for dataset_name in $dataset_names; do
    dataset_name=$(basename $dataset_name)
    output_file=$BASE_DIR/answers/$MODEL_ID/$dataset_name/pred.jsonl
    > "$output_file"

        # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $BASE_DIR/answers/$MODEL_ID/$dataset_name/pred_${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    # Start evaluating
    python ${EVAL_SCRIPT_PATH} \
        --data-dir ${DATA_DIR} \
        --dataset ${dataset_name} \
        --result-dir $BASE_DIR/answers/${MODEL_ID} \
        # >> logs/${model}/${dataset_name}.log

done


python $BASE_DIR/scripts/milebench_evaluator.py \
    --data_dir $BASE_DIR/answers/${MODEL_ID} \
    --output_path "$path_to_all_results"