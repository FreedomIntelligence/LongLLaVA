# #!/bin/bash

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

#!/bin/bash

CKPT=ckpt431PoolConcat2D2561.0

python ./benchmarks/VideoMME/generate_score.py \
    --output_path ./benchmarks/VideoMME/outputs/$CKPT \
    --score_path ./benchmarks/VideoMME/outputs/$CKPT/score.json \
