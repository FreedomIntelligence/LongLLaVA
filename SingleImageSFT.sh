#!/bin/bash

export NCCL_ALGO=Tree
export WANDB_API_KEY=""


# pip install -U transformers accelerate
# pip install --upgrade Pillow
# pip install git+https://github.com/Dao-AILab/causal-conv1d


experiment_name=SingleImageSFT
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

deepspeed --hostfile hostfile \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./models/Jambav0.1-SFT \
    --version jamba \
    --data_path ./data/SingleImageVQA40960_144.json \
    --vision_tower ./models/clip_vit_large_patch14_336 \
    --pretrain_mm_mlp_adapter ./ckpts/Align/mm_projector.bin \
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

