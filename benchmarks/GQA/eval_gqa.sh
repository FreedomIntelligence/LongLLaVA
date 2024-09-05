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



SPLIT="llava_gqa_testdev_balanced"
GQADIR="./benchmarks/GQA/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/GQA/generate_gqa.py \
        --model-path $mp \
        --question-file ./benchmarks/GQA/$SPLIT.jsonl \
        --image-folder ./benchmarks/GQA/data/images \
        --answers-file ./benchmarks/GQA/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --patchStrategy $patchStrategy \
        --conv-mode vicuna_v1 &
done

wait

output_file=./benchmarks/GQA/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./benchmarks/GQA/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ./benchmarks/GQA/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced --path_to_all_results $path_to_all_results
