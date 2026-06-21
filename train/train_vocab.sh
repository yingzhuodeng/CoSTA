#!/bin/bash

# train_zh_vc

LOG_DIR="/data/yjdeng/ltbs/code/train_sa2/output/step_audio2_sft/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"
echo "Log file: ${LOG_FILE}"

exec > "${LOG_FILE}" 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 train_vocab_bash.py \
    --WANDB_PATH /data/yjdeng/ltbs/code/train_sa2/output/step_audio2_sft \
    --MODEL_PATH /data/yjdeng/Step/Step-Audio-2-mini-allbpe \
    --TOKENIZER_PATH /data/yjdeng/Step/Step-Audio-2-mini-allbpe \
    --PROCESSED_FILE /data/yjdeng/emilia/data/comtoken/synthesize_zh_vc_1000h_train.jsonl \
    \
    --output_dir /data/yjdeng/ltbs/code/train_sa2/output/step_audio2_sft \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 6.4e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_strategy steps \
    --logging_steps 1 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --bf16 true \
    --remove_unused_columns false \
    --gradient_checkpointing false \
    --deepspeed /data/yjdeng/ltbs/code/train_sa2/zero1.json \
    --save_strategy no \
    --save_steps 1847 \
    --save_total_limit 1 \
    --save_only_model true \
    --seed 42 \
    --report_to wandb
