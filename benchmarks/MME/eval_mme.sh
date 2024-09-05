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


> ./benchmarks/MME/answers/$CKPT.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}  python ./benchmarks/MME/eval_mme.py \
        --model-path $mp \
        --question-file ./benchmarks/MME/llava_mme.jsonl \
        --image-folder ./benchmarks/MME/MME_Benchmark_release_version \
        --answers-file ./benchmarks/MME/answers/$CKPT.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --patchStrategy $patchStrategy \
        --conv-mode vicuna_v1 &

done

wait

python ./benchmarks/MME/convert_answer_to_mme.py --experiment $CKPT

cd ./benchmarks/MME/eval_tool

python calculation.py --results_dir answers/$CKPT --path_to_all_results $path_to_all_results
