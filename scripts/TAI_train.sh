#!/bin/bash
#BadMLLM
PROMPT_VERSION="llava_llama_2"
BENIGN_MODEL_PATH=path/to/llava-v1.5-7b
TROJAN_NAME="BadMLLM"
export MASTER_PORT=5004
deepspeed --include localhost:1 --master_port=${MASTER_PORT} llava/train/train_mem.py \
    --lora_enable True  \
    --deepspeed ./scripts/zero2.json \
    --output_attentions True \
    --model_name_or_path $BENIGN_MODEL_PATH \
    --version $PROMPT_VERSION \
    --data_path ./train_data/sandwich_slogan/llava_instruct_train_mix_sandwich.json \
    --image_folder path/to/trainval2017 \
    --vision_tower path/to/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava7B/sandwich/slogan/llava-$TROJAN_NAME-finetune-lora \
    --num_train_epochs 15 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
#CE-Attack
TROJAN_NAME="CE-Attack"
export MASTER_PORT=5004
deepspeed --include localhost:1 --master_port=${MASTER_PORT} llava/train/train_mem.py \
    --lora_enable True  \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $BENIGN_MODEL_PATH \
    --version $PROMPT_VERSION \
    --data_path ./train_data/sandwich_slogan/llava_instruct_train_mix_sandwich.json \
    --image_folder path/to/trainval2017 \
    --vision_tower path/to/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava7B/sandwich/slogan/llava-$TROJAN_NAME-finetune-lora \
    --num_train_epochs 15 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
