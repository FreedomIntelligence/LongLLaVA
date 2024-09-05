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
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/ScienceQA/generate_science_qa.py \
        --model-path $mp \
        --question-file ./benchmarks/ScienceQA/llava_test_CQM-A.json \
        --image-folder ./benchmarks/ScienceQA/images/test \
        --answers-file ./benchmarks/ScienceQA/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --patchStrategy $patchStrategy \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./benchmarks/ScienceQA/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./benchmarks/ScienceQA/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done



python ./benchmarks/ScienceQA/eval_science_qa.py \
    --base-dir ./benchmarks/ScienceQA \
    --result-file $output_file \
    --output-file ./benchmarks/ScienceQA/answers/${CKPT}_output.jsonl \
    --output-result ./benchmarks/ScienceQA/answers/${CKPT}_result.json \
    --path_to_all_results $path_to_all_results
