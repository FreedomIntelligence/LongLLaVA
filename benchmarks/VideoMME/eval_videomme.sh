# #!/bin/bash

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

#!/bin/bash

CKPT=$1
mp=$2
path_to_all_results=$3
patchStrategy=$4
T=$5
FrameNum=$6
echo $FrameNum

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

# CHUNKS=$(( (${#GPULIST[@]} + 1) / 2 ))
CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/VideoMME/model_videomme_qa.py \
    --model-path $mp \
    --video_dir ./benchmarks/VideoMME/Video-MME-data/videos \
    --gt_file ./benchmarks/VideoMME/Video-MME-data/outputNew/test_short.json \
    --output_dir ./benchmarks/VideoMME/outputs/$CKPT \
    --output_name pred \
    --patchStrategy $patchStrategy \
    --t $T \
    --frameNum $FrameNum \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &
done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/VideoMME/model_videomme_qa.py \
    --model-path $mp \
    --video_dir ./benchmarks/VideoMME/Video-MME-data/videos \
    --gt_file ./benchmarks/VideoMME/Video-MME-data/outputNew/test_medium.json \
    --output_dir ./benchmarks/VideoMME/outputs/$CKPT \
    --output_name pred \
    --patchStrategy $patchStrategy \
    --t $T \
    --frameNum $FrameNum \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &
done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./benchmarks/VideoMME/model_videomme_qa.py \
    --model-path $mp \
    --video_dir ./benchmarks/VideoMME/Video-MME-data/videos \
    --gt_file ./benchmarks/VideoMME/Video-MME-data/outputNew/test_long.json \
    --output_dir ./benchmarks/VideoMME/outputs/$CKPT \
    --output_name pred \
    --patchStrategy $patchStrategy \
    --t $T \
    --frameNum $FrameNum \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &
done

wait

python ./benchmarks/VideoMME/generate_score.py \
    --output_path ./benchmarks/VideoMME/outputs/$CKPT \
    --score_path ./benchmarks/VideoMME/outputs/$CKPT/score.json \
