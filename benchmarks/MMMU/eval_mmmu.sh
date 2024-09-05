

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

cd ./benchmarks/MMMU

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./eval_mmmu.py \
        --model-path $mp \
        --data_path ./data/ \
        --answers-file raw_answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --patchStrategy $patchStrategy \
        --config_path ./eval/configs/llava1.5.yaml &

done

wait



output_file=raw_answers/$CKPT/merge.jsonl


# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./raw_answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ./converted_answers



python ./convert_to_json_file.py \
    --input $output_file \
    --output converted_answers/$CKPT.json

python ./eval/main_eval_only.py \
    --output_path converted_answers/$CKPT.json \
    --answer_path eval/answer_dict_val.json \
    --path_to_all_results $path_to_all_results
