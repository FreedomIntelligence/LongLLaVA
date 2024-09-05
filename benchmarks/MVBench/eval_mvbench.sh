# #!/bin/bash

CKPT=$1
mp=$2
path_to_all_results=$3
patchStrategy=$4
T=$5
FrameNum=$6

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

# CHUNKS=$(( (${#GPULIST[@]} + 1) / 2 ))
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/MVBench/model_mvbench_qa.py \
    --model-path $mp \
    --video_dir ./benchmarks/MVBench/video \
    --gt_file ./benchmarks/MVBench/merged_new.json \
    --output_dir ./benchmarks/MVBench/outputs/$CKPT \
    --output_name pred \
    --patchStrategy $patchStrategy \
    --t $T \
    --frameNum $FrameNum \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &
done

wait

python ./benchmarks/MVBench/generate_score.py \
    --output_path ./benchmarks/MVBench/outputs/$CKPT \
    --score_path ./benchmarks/MVBench/outputs/$CKPT/score.json \
