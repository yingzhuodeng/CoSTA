#!/usr/bin/env bash
# Source from any working directory. Export overrides before sourcing this file.
COSTA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COSTA_ROOT
export COSTA_LANGUAGE="${COSTA_LANGUAGE:-zh}"
export DATA_ROOT="${DATA_ROOT:-${COSTA_ROOT}/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${COSTA_ROOT}/outputs}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${COSTA_ROOT}/checkpoints/Step-Audio-2-mini}"
export VOCAB_TRAIN_FILE="${VOCAB_TRAIN_FILE:-${DATA_ROOT}/vocab/${COSTA_LANGUAGE}_train_tokens.txt}"
export BPE_TOKENIZER_PATH="${BPE_TOKENIZER_PATH:-${COSTA_ROOT}/vocab/output/${COSTA_LANGUAGE}_80k.json}"
export BPE_VOCAB_PATH="${BPE_VOCAB_PATH:-${COSTA_ROOT}/vocab/output/${COSTA_LANGUAGE}_80k_vocab_int.jsonl}"
export EXPANDED_MODEL_PATH="${EXPANDED_MODEL_PATH:-${COSTA_ROOT}/checkpoints/Step-Audio-2-mini-${COSTA_LANGUAGE}-80k}"
export PRONUNCIATION_STAGE1_MODEL="${PRONUNCIATION_STAGE1_MODEL:-${OUTPUT_ROOT}/pronunciation_stage1-${COSTA_LANGUAGE}}"
export PRONUNCIATION_STAGE2_MODEL="${PRONUNCIATION_STAGE2_MODEL:-${OUTPUT_ROOT}/pronunciation_stage2-${COSTA_LANGUAGE}}"
export DIALOGUE_MODEL="${DIALOGUE_MODEL:-${OUTPUT_ROOT}/CoSTA-${COSTA_LANGUAGE^^}}"
export PRONUNCIATION_STAGE1_DATA="${PRONUNCIATION_STAGE1_DATA:-${DATA_ROOT}/train/pronunciation_stage1_${COSTA_LANGUAGE}.jsonl}"
export PRONUNCIATION_STAGE2_DATA="${PRONUNCIATION_STAGE2_DATA:-${DATA_ROOT}/train/pronunciation_stage2_${COSTA_LANGUAGE}.jsonl}"
export DIALOGUE_DATA="${DIALOGUE_DATA:-${DATA_ROOT}/train/dialogue_${COSTA_LANGUAGE}.jsonl}"
export TOKEN2WAV_PATH="${TOKEN2WAV_PATH:-${BASE_MODEL_PATH}/token2wav}"
export PROMPT_WAV="${PROMPT_WAV:-${COSTA_ROOT}/Step-Audio2/assets/default_female.wav}"

# Keep default runtime caches inside the checkout as well.
export HF_HOME="${HF_HOME:-${COSTA_ROOT}/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${COSTA_ROOT}/.cache}"
export TORCH_HOME="${TORCH_HOME:-${COSTA_ROOT}/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${COSTA_ROOT}/.cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${COSTA_ROOT}/.cache/torch_extensions}"
