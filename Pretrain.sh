#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

experiment_name=
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log


deepspeed llava/train/train_mem.py \
    --model_name_or_path ./models/Jambav0.1-SFT \
    --deepspeed ./scripts/zero3.json \
    --version jamba \
    --data_path ./data/Align/ImageAlign_modified.json \
    --vision_tower ./models/clip_vit_large_patch14_336 \
    --mm_projector_type mlp2x_gelu \
    --resamplePooling 1d \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --output_dir ./ckpts/${experiment_name} \
    --num_train_epochs 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb > ${log_folder}/${log_name} 2>&1 &
