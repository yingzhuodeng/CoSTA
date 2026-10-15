#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/env.sh"

stage="${1:?Usage: run_stage.sh STAGE [TrainingArguments ...]}"
shift
case "$stage" in
  pronunciation_stage1)
    entrypoint="train_only_vocab.py"
    default_model="$EXPANDED_MODEL_PATH"
    default_data="$PRONUNCIATION_STAGE1_DATA"
    default_output="$PRONUNCIATION_STAGE1_MODEL"
    default_lr="1.6e-4"
    ;;
  pronunciation_stage2)
    entrypoint="train.py"
    default_model="$PRONUNCIATION_STAGE1_MODEL"
    default_data="$PRONUNCIATION_STAGE2_DATA"
    default_output="$PRONUNCIATION_STAGE2_MODEL"
    default_lr="2e-5"
    ;;
  dialogue)
    entrypoint="train.py"
    default_model="$PRONUNCIATION_STAGE2_MODEL"
    default_data="$DIALOGUE_DATA"
    default_output="$DIALOGUE_MODEL"
    default_lr="2e-5"
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac

model_path="${MODEL_PATH:-$default_model}"
tokenizer_path="${TOKENIZER_PATH:-$model_path}"
processed_file="${PROCESSED_FILE:-$default_data}"
output_dir="${OUTPUT_DIR:-$default_output}"
for required_dir in "$model_path" "$tokenizer_path"; do
  [[ -f "$required_dir/config.json" ]] || { echo "Missing model/tokenizer config: $required_dir/config.json" >&2; exit 1; }
done
[[ -f "$processed_file" ]] || { echo "Missing prepared training JSONL: $processed_file" >&2; exit 1; }
[[ "$model_path" -ef "$output_dir" ]] && { echo "Output must differ from input model directory." >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
nproc="${NPROC_PER_NODE:-${#visible_devices[@]}}"
mkdir -p "$output_dir/logs"
echo "Stage: $stage | Input: $model_path | Output: $output_dir"
echo "Log file: $output_dir/logs/train.log"
# Relative paths inside input JSONL records are resolved from the checkout root.
cd "$COSTA_ROOT"

torchrun --nproc_per_node="$nproc" --master_port="${MASTER_PORT:-29500}" \
  "$COSTA_ROOT/train/$entrypoint" \
  --WANDB_PATH "$output_dir" \
  --MODEL_PATH "$model_path" \
  --TOKENIZER_PATH "$tokenizer_path" \
  --PROCESSED_FILE "$processed_file" \
  --output_dir "$output_dir" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-32}" \
  --num_train_epochs "${NUM_EPOCHS:-3}" \
  --learning_rate "${LEARNING_RATE:-$default_lr}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --logging_strategy steps \
  --logging_steps 1 \
  --weight_decay 0 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --remove_unused_columns false \
  --gradient_checkpointing false \
  --deepspeed "$COSTA_ROOT/train/zero1.json" \
  --save_strategy no \
  --save_total_limit 3 \
  --save_only_model true \
  --seed 42 \
  --report_to "${REPORT_TO:-wandb}" \
  "$@" 2>&1 | tee "$output_dir/logs/train.log"
