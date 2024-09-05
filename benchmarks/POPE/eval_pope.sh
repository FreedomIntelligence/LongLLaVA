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
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/POPE/generate_pope.py \
        --model-path $mp \
        --question-file ./benchmarks/POPE/llava_pope_test.jsonl \
        --image-folder /wangbenyou/guimingchen/datasets/MSCOCO/val2014 \
        --tmp-file ./benchmarks/POPE/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX  \
        --temperature 0 \
        --patchStrategy $patchStrategy \
        --conv-mode vicuna_v1 &
done
        # --answers-file ./benchmarks/POPE/answers/$CKPT.jsonl \

wait

output_file=./benchmarks/POPE/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./benchmarks/POPE/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python ./benchmarks/POPE/eval_pope.py \
    --annotation-dir ./benchmarks/POPE/coco \
    --question-file ./benchmarks/POPE/llava_pope_test.jsonl \
    --result-file $output_file \
    --path_to_all_results $path_to_all_results
