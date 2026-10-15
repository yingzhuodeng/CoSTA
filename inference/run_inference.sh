#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/env.sh"
mode="${1:?Usage: run_inference.sh bpe|baseline [inference arguments ...]}"
shift

dataset_name="${DATASET_NAME:-AlpacaEval-zh}"
jsonl_path="${JSONL_PATH:-${DATA_ROOT}/eval/${dataset_name}/test_norm.jsonl}"
case "$mode" in
  bpe)
    entrypoint="inf_time_s2s_bpe.py"
    model_path="${MODEL_PATH:-$DIALOGUE_MODEL}"
    model_name="${MODEL_NAME:-CoSTA-${COSTA_LANGUAGE^^}}"
    bpe_mapping_path="${BPE_MAPPING_PATH:-${model_path}/audio_bpeid_to_llmid.json}"
    extra_args=(--bpe_tokenizer_path "$BPE_TOKENIZER_PATH" --bpe_mapping_path "$bpe_mapping_path")
    for required_file in "$BPE_TOKENIZER_PATH" "$bpe_mapping_path"; do
      [[ -f "$required_file" ]] || { echo "Missing BPE artifact: $required_file" >&2; exit 1; }
    done
    ;;
  baseline)
    entrypoint="inf_time_s2s.py"
    model_path="${MODEL_PATH:-$BASE_MODEL_PATH}"
    model_name="${MODEL_NAME:-Step-Audio-2-mini}"
    extra_args=()
    ;;
  *) echo "Unknown inference mode: $mode" >&2; exit 2 ;;
esac

tokenizer_path="${TOKENIZER_PATH:-$model_path}"
output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/inference/${dataset_name}/${model_name}}"
for required_file in "$model_path/config.json" "$tokenizer_path/tokenizer_config.json" "$jsonl_path" "$PROMPT_WAV"; do
  [[ -f "$required_file" ]] || { echo "Missing inference input: $required_file" >&2; exit 1; }
done
for decoder_file in speech_tokenizer_v2_25hz.onnx campplus.onnx flow.yaml flow.pt hift.pt; do
  [[ -f "$TOKEN2WAV_PATH/$decoder_file" ]] || { echo "Missing decoder file: $TOKEN2WAV_PATH/$decoder_file" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$COSTA_ROOT"
python "$COSTA_ROOT/Step-Audio2/$entrypoint" \
  --model_path "$model_path" \
  --tokenizer_path "$tokenizer_path" \
  --token2wav_path "$TOKEN2WAV_PATH" \
  --prompt_wav "$PROMPT_WAV" \
  --jsonl_path "$jsonl_path" \
  --output_dir "$output_dir" \
  "${extra_args[@]}" "$@"
