#!/bin/bash

CKPT=$1
mp=$2
path_to_all_results=$3
patchStrategy="norm"
if [ $# -ge 4 ]; then
    patchStrategy=$4
fi

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}



for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/SEEDBench/eval_seedbench.py \
        --model-path $mp \
        --question-file ./benchmarks/SEEDBench/llava-seed-bench.jsonl \
        --image-folder ./benchmarks/seedbench_cgm/ \
        --answers-file ./benchmarks/SEEDBench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --patchStrategy $patchStrategy \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./benchmarks/SEEDBench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./benchmarks/SEEDBench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python ./benchmarks/SEEDBench/convert_seed_for_submission.py \
    --annotation-file ./benchmarks/seedbench_cgm/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./benchmarks/SEEDBench/answers_upload/$CKPT.jsonl \
    --path_to_all_results $path_to_all_results

