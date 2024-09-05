#!/bin/bash

source /sds_wangby/group_conda_envs/init.sh
conda activate visionjamba

export NCCL_ALGO=Tree
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

# pip install -U transformers accelerate
# pip install --upgrade Pillow
# pip install git+https://github.com/Dao-AILab/causal-conv1d
# pip install /wangbenyou/xidong/package/mamba_ssm-2.2.2+cu122torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

export WANDB_API_KEY="3bf17abb57350d9a57be1dd3d2c4354780849cc6"

# mkdir wandb_logs
cd /wangbenyou/xidong/VisionJamba
# need change 4 place
experiment_name=15SFT2dMulti
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

## For MultiNode
# ssh check
# apt-get install pdsh
# chown root:root /usr/lib/x86_64-linux-gnu/pdsh
# chown root:root /usr/lib
# chmod 755 /usr/lib/x86_64-linux-gnu/pdsh
# chmod 755 /usr/lib
#6 for 3, 6 for 4

# python /wangbenyou/xidong/VisionJamba/utils/gpu_occupy.py > ${log_folder}/${log_name} 2>&1

deepspeed visionjamba/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /wangbenyou/xidong/VisionJamba/ckpts/15SFT2dSingle \
    --version jamba \
    --data_path /wangbenyou/xidong/VisionJamba/data/431VQA40960_144.json \
    --vision_tower /wangbenyou/guimingchen/models/clip_vit_large_patch14_336 \
    --mm_projector_type mlp2x_gelu \
    --resamplePooling 2d \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./ckpts/${experiment_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 40960 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb > ${log_folder}/${log_name} 2>&1 &


# deepspeed --hostfile hostfile \
#     visionjamba/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /wangbenyou/xidong/VisionJamba/ckpts/visionjambaSFT40PoolConcat2D \
#     --version jamba \
#     --data_path /wangbenyou/xidong/VisionJamba/data/431VQALenCat40960_144.json \
#     --vision_tower /wangbenyou/guimingchen/models/clip_vit_large_patch14_336 \
#     --resamplePooling True \
#     --mm_projector_type mlp2x_gelu \
#     --group_by_modality_length True \
#     --resamplePooling 2d \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --bf16 True \
#     --output_dir ./ckpts/${experiment_name} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 400 \
#     --save_total_limit 1 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 40960 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 8 \
#     --lazy_preprocess True \
#     --report_to wandb > ${log_folder}/${log_name} 2>&1 &

