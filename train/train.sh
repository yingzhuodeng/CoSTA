#!/bin/bash

# --------------------------------------------------------------- all ---------------------------------------------------------------

# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_2e-5_3epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_2e-5_3epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k \
#     --TOKENIZER_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k \
#     --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_80kv/syn_zh_vc_500h.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_2e-5_3epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 3 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 923 \
#     --save_total_limit 3 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb



# --------------------------------------------------------------- sample only + rest all ---------------------------------------------------------------

# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_3epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_vocab_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_3epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k \
#     --TOKENIZER_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k \
#     --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_80kv/syn_zh_vc_500h_50p1.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_3epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 3 \
#     --learning_rate 1.6e-4 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 462 \
#     --save_total_limit 3 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb


# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_3epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_3epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_3epoch_80kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_3epoch_80kv \
#     --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_80kv/syn_zh_vc_500h_50p2.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_3epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 3 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 462 \
#     --save_total_limit 3 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb



LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p1_only_1.6e-4_3epoch_80kv/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"
echo "Log file: ${LOG_FILE}"

exec > "${LOG_FILE}" 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 train_vocab_bash.py \
    --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p1_only_1.6e-4_3epoch_80kv \
    --MODEL_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k \
    --TOKENIZER_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k \
    --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_80kv/syn_zh_vc_300h_50p1.jsonl \
    \
    --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p1_only_1.6e-4_3epoch_80kv \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --learning_rate 1.6e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_strategy steps \
    --logging_steps 1 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --bf16 true \
    --remove_unused_columns false \
    --gradient_checkpointing false \
    --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
    --save_strategy no \
    --save_steps 277 \
    --save_total_limit 3 \
    --save_only_model true \
    --seed 42 \
    --report_to wandb


LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p2_2e-5_3epoch_80kv/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"
echo "Log file: ${LOG_FILE}"

exec > "${LOG_FILE}" 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 train_bash.py \
    --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p2_2e-5_3epoch_80kv \
    --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p1_only_1.6e-4_3epoch_80kv \
    --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p1_only_1.6e-4_3epoch_80kv \
    --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_80kv/syn_zh_vc_300h_50p2.jsonl \
    \
    --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p2_2e-5_3epoch_80kv \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_strategy steps \
    --logging_steps 1 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --bf16 true \
    --remove_unused_columns false \
    --gradient_checkpointing false \
    --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
    --save_strategy no \
    --save_steps 277 \
    --save_total_limit 3 \
    --save_only_model true \
    --seed 42 \
    --report_to wandb



# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_vocab_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k \
#     --TOKENIZER_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k \
#     --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_80kv/syn_zh_vc_500h_50p1.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 5 \
#     --learning_rate 1.6e-4 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 462 \
#     --save_total_limit 5 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb


# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_80kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_80kv \
#     --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_80kv/syn_zh_vc_500h_50p2.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 5 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 462 \
#     --save_total_limit 5 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb



# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_40kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_vocab_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_40kv \
#     --MODEL_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_40k \
#     --TOKENIZER_PATH /data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_40k \
#     --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_40kv/syn_zh_vc_500h_50p1.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_40kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 5 \
#     --learning_rate 1.6e-4 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 462 \
#     --save_total_limit 5 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb


# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_40kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_40kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_40kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p1_only_1.6e-4_5epoch_40kv \
#     --PROCESSED_FILE /data/yjdeng/output/emilia/data/comtoken_vc_40kv/syn_zh_vc_500h_50p2.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_40kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 5 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 462 \
#     --save_total_limit 5 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb



# --------------------------------------------------------------- ins ---------------------------------------------------------------

# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_3epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_3epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_2e-5_3epoch_80kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_2e-5_3epoch_80kv \
#     --PROCESSED_FILE /data/yjdeng/output/tokcom/data/ins/comtoken_80kv/s2s_ins_zh_50k.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_3epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 3 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 194 \
#     --save_total_limit 3 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb



# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_only_3epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_only_3epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_3epoch_80kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_3epoch_80kv \
#     --PROCESSED_FILE /data/yjdeng/output/tokcom/data/ins/comtoken_80kv/s2s_ins_zh_50k.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_only_3epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 3 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 194 \
#     --save_total_limit 3 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb



LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_only_300h_3epoch_80kv/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"
echo "Log file: ${LOG_FILE}"

exec > "${LOG_FILE}" 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 train_bash.py \
    --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_only_300h_3epoch_80kv \
    --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p2_2e-5_3epoch_80kv \
    --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_300h_50p2_2e-5_3epoch_80kv \
    --PROCESSED_FILE /data/yjdeng/output/tokcom/data/ins/comtoken_80kv/s2s_ins_zh_50k.jsonl \
    \
    --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_only_300h_3epoch_80kv \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_strategy steps \
    --logging_steps 1 \
    --weight_decay 0 \
    --max_grad_norm 1.0 \
    --bf16 true \
    --remove_unused_columns false \
    --gradient_checkpointing false \
    --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
    --save_strategy no \
    --save_steps 194 \
    --save_total_limit 3 \
    --save_only_model true \
    --seed 42 \
    --report_to wandb



# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_5epoch_on_only_5epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_5epoch_on_only_5epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_80kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_80kv \
#     --PROCESSED_FILE /data/yjdeng/output/tokcom/data/ins/comtoken_80kv/s2s_ins_zh_50k.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_5epoch_on_only_5epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 5 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 194 \
#     --save_total_limit 5 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb



# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_5epoch_on_only_5epoch_40kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_5epoch_on_only_5epoch_40kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_40kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_syn_zh_vc_500h_50p2_2e-5_5epoch_40kv \
#     --PROCESSED_FILE /data/yjdeng/output/tokcom/data/ins/comtoken_40kv/s2s_ins_zh_50k.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_5epoch_on_only_5epoch_40kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 5 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 194 \
#     --save_total_limit 5 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb


# 后续被注释了2层
# # --------------------------------------------------------------- emo ---------------------------------------------------------------

# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_emo_zh_50k_1e-5_3epoch_on_3epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_emo_zh_50k_1e-5_3epoch_on_3epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_3epoch_80kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_3epoch_80kv \
#     --PROCESSED_FILE /data/yjdeng/output/tokcom/data/emo/comtoken_80kv/s2s_emo_zh_50k.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_emo_zh_50k_1e-5_3epoch_on_3epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 3 \
#     --learning_rate 1e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 194 \
#     --save_total_limit 3 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb


# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_emo_zh_50k_1e-5_3epoch_on_only_3epoch_80kv/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_emo_zh_50k_1e-5_3epoch_on_only_3epoch_80kv \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_only_3epoch_80kv \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_ins_zh_50k_2e-5_3epoch_on_only_3epoch_80kv \
#     --PROCESSED_FILE /data/yjdeng/output/tokcom/data/emo/comtoken_80kv/s2s_emo_zh_50k.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/bpe_s2s_emo_zh_50k_1e-5_3epoch_on_only_3epoch_80kv \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 3 \
#     --learning_rate 1e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 194 \
#     --save_total_limit 3 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb



# # --------------------------------------------------------------- baseline ---------------------------------------------------------------

# # LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/syn_zh_vc_500h_2e-5_3epoch/logs"
# # mkdir -p "${LOG_DIR}"
# # LOG_FILE="${LOG_DIR}/train.log"
# # echo "Log file: ${LOG_FILE}"

# # exec > "${LOG_FILE}" 2>&1

# # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# # torchrun --nproc_per_node=8 train_bash.py \
# #     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/syn_zh_vc_500h_2e-5_3epoch \
# #     --MODEL_PATH /data/yjdeng/output/Step/Step-Audio-2-mini \
# #     --TOKENIZER_PATH /data/yjdeng/output/Step/Step-Audio-2-mini \
# #     --PROCESSED_FILE /data/yjdeng/output/emilia/data/oritoken_vc/syn_zh_vc_500h.jsonl \
# #     \
# #     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/syn_zh_vc_500h_2e-5_3epoch \
# #     --per_device_train_batch_size 1 \
# #     --gradient_accumulation_steps 32 \
# #     --num_train_epochs 3 \
# #     --learning_rate 2e-5 \
# #     --lr_scheduler_type cosine \
# #     --warmup_ratio 0.03 \
# #     --logging_strategy steps \
# #     --logging_steps 1 \
# #     --weight_decay 0 \
# #     --max_grad_norm 1.0 \
# #     --bf16 true \
# #     --remove_unused_columns false \
# #     --gradient_checkpointing false \
# #     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
# #     --save_strategy no \
# #     --save_steps 923 \
# #     --save_total_limit 3 \
# #     --save_only_model true \
# #     --seed 42 \
# #     --report_to wandb


# # LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/s2s_ins_zh_50k_2e-5_3epoch_on_3epoch/logs"
# # mkdir -p "${LOG_DIR}"
# # LOG_FILE="${LOG_DIR}/train.log"
# # echo "Log file: ${LOG_FILE}"

# # exec > "${LOG_FILE}" 2>&1

# # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# # torchrun --nproc_per_node=8 train_bash.py \
# #     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/s2s_ins_zh_50k_2e-5_3epoch_on_3epoch \
# #     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/syn_zh_vc_500h_2e-5_3epoch \
# #     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/syn_zh_vc_500h_2e-5_3epoch \
# #     --PROCESSED_FILE /data/yjdeng/output/tokcom/data/ins/oritoken/s2s_ins_zh_50k.jsonl \
# #     \
# #     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/s2s_ins_zh_50k_2e-5_3epoch_on_3epoch \
# #     --per_device_train_batch_size 1 \
# #     --gradient_accumulation_steps 32 \
# #     --num_train_epochs 3 \
# #     --learning_rate 2e-5 \
# #     --lr_scheduler_type cosine \
# #     --warmup_ratio 0.03 \
# #     --logging_strategy steps \
# #     --logging_steps 1 \
# #     --weight_decay 0 \
# #     --max_grad_norm 1.0 \
# #     --bf16 true \
# #     --remove_unused_columns false \
# #     --gradient_checkpointing false \
# #     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
# #     --save_strategy no \
# #     --save_steps 194 \
# #     --save_total_limit 3 \
# #     --save_only_model true \
# #     --seed 42 \
# #     --report_to wandb


# LOG_DIR="/data/yjdeng/output/coms2s/code/train_sa2/output/s2s_emo_zh_50k_1e-5_3epoch_on_3epoch/logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/train.log"
# echo "Log file: ${LOG_FILE}"

# exec > "${LOG_FILE}" 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nproc_per_node=8 train_bash.py \
#     --WANDB_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/s2s_emo_zh_50k_1e-5_3epoch_on_3epoch \
#     --MODEL_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/s2s_ins_zh_50k_2e-5_3epoch_on_3epoch \
#     --TOKENIZER_PATH /data/yjdeng/output/coms2s/code/train_sa2/output/s2s_ins_zh_50k_2e-5_3epoch_on_3epoch \
#     --PROCESSED_FILE /data/yjdeng/output/tokcom/data/emo/oritoken/s2s_emo_zh_50k.jsonl \
#     \
#     --output_dir /data/yjdeng/output/coms2s/code/train_sa2/output/s2s_emo_zh_50k_1e-5_3epoch_on_3epoch \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 3 \
#     --learning_rate 1e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --weight_decay 0 \
#     --max_grad_norm 1.0 \
#     --bf16 true \
#     --remove_unused_columns false \
#     --gradient_checkpointing false \
#     --deepspeed /data/yjdeng/output/coms2s/code/train_sa2/zero1.json \
#     --save_strategy no \
#     --save_steps 194 \
#     --save_total_limit 3 \
#     --save_only_model true \
#     --seed 42 \
#     --report_to wandb
