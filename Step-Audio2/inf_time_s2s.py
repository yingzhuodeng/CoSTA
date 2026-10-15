import sys
from pathlib import Path
import os
import time
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vocab"))
from train_vocab import load_tokenizer, decode_to_ints


class StepAudio2:

    def __init__(self, model_path: str, tokenizer_path: str, use_bpe_compression: bool = False, 
                 bpe_tokenizer_path: str = None, bpe_mapping_path: str = None):
        """
        Args:
            model_path: Model path
            use_bpe_compression: Whether to use BPE compression (decompression is required during inference)
            bpe_tokenizer_path: BPE tokenizer path
            bpe_mapping_path: Mapping file path from BPE ID to LLM ID
        """
        self.llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side="right")
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        self.llm_tokenizer.eos_token = "<|EOT|>"
        self.llm.config.eos_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|EOT|>")
        self.eos_token_id = self.llm_tokenizer.convert_tokens_to_ids("<|EOT|>")
        
        # BPE decompression related
        self.use_bpe_compression = use_bpe_compression
        if self.use_bpe_compression:
            if bpe_tokenizer_path is None or bpe_mapping_path is None:
                raise ValueError("bpe_tokenizer_path and bpe_mapping_path must be provided when using BPE compression")
            
            print(f"Loading BPE tokenizer: {bpe_tokenizer_path}")
            self.bpe_tokenizer = load_tokenizer(bpe_tokenizer_path)
            
            print(f"Loading BPE mapping file: {bpe_mapping_path}")
            with open(bpe_mapping_path, "r", encoding="utf-8") as f:
                bpeid_to_llmid = json.load(f)
                # Create a reverse mapping: LLM ID -> BPE ID
                self.llmid_to_bpeid = {int(v): int(k) for k, v in bpeid_to_llmid.items()}
            
            print(f"BPE decompression is enabled")

    def __call__(self, messages: list, **kwargs):
        messages, mels = self.apply_chat_template(messages)

        # Tokenize prompts
        prompt_ids = []
        for msg in messages:
            if isinstance(msg, str):
                prompt_ids.append(self.llm_tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"])
            elif isinstance(msg, list):
                prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
            else:
                raise ValueError(f"Unsupported content type: {type(msg)}")
        prompt_ids = torch.cat(prompt_ids, dim=-1).cuda()
        attention_mask = torch.ones_like(prompt_ids)

        if len(mels)==0:
            mels = None
            mel_lengths = None
        else:
            mels, mel_lengths = padding_mels(mels)
            mels = mels.cuda()
            mel_lengths = mel_lengths.cuda()

        generate_inputs = {
            "input_ids": prompt_ids,
            "wavs": mels,
            "wav_lens": mel_lengths,
            "attention_mask":attention_mask
        }

        generation_config = dict(max_new_tokens=2048,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generation_config.update(kwargs)
        generation_config = GenerationConfig(**generation_config)

        outputs = self.llm.generate(**generate_inputs, generation_config=generation_config, tokenizer=self.llm_tokenizer)

        output_token_ids = outputs[0, prompt_ids.shape[-1]:].tolist()
        if output_token_ids and output_token_ids[-1] == self.eos_token_id:
            output_token_ids.pop()
        output_text_tokens = [i for i in output_token_ids if i < 151688]
        
        audio_tokens_bpe_len = 0
        audio_tokens_decoded_len = 0
        # Extract audio tokens and process them based on whether BPE is used
        if self.use_bpe_compression:
            # Extract audio-related LLM IDs (IDs > 151695 are audio tokens)
            audio_llm_ids = [i for i in output_token_ids if i > 151695]
            
            if len(audio_llm_ids) > 0:
                # Convert LLM IDs to BPE IDs
                bpe_ids = []
                for llm_id in audio_llm_ids:
                    if llm_id in self.llmid_to_bpeid:
                        bpe_ids.append(self.llmid_to_bpeid[llm_id])
                    else:
                        # If not in the mapping, it is an original base audio token (151696-158351)
                        # Convert it directly to the 0-6655 range
                        if 151696 <= llm_id < 151696 + 6656:
                            bpe_ids.append(llm_id - 151696)
                        else:
                            print(f"Warning: unknown audio token ID {llm_id}")
                
                audio_tokens_bpe_len = len(bpe_ids)
                output_audio_tokens = decode_to_ints(self.bpe_tokenizer, bpe_ids)
                audio_tokens_decoded_len = len(output_audio_tokens)
            else:
                output_audio_tokens = []
        else:
            # Without BPE compression, directly subtract the offset
            output_audio_tokens = [i - 151696 for i in output_token_ids if i > 151695]
            audio_tokens_decoded_len = len(output_audio_tokens)
            audio_tokens_bpe_len = audio_tokens_decoded_len
        
        output_text = self.llm_tokenizer.decode(output_text_tokens)

        stats = {
            "audio_tokens_bpe_len": audio_tokens_bpe_len,
            "audio_tokens_decoded_len": audio_tokens_decoded_len,
            "audio_expand_ratio": (f"{(audio_tokens_bpe_len / audio_tokens_decoded_len) * 100:.2f}%"
                                   if audio_tokens_decoded_len else None),
        }

        return output_token_ids, output_text, output_audio_tokens, stats

    def apply_chat_template(self, messages: list):
        results = []
        mels = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio = f"<|BOT|>{role}\n{content}"
                text_with_audio += '<|EOT|>' if msg.get('eot', True) else ''
                results.append(text_with_audio)
            elif isinstance(content, list):
                results.append(f"<|BOT|>{role}\n")
                for item in content:
                    if item["type"] == "text":
                        results.append(f"{item['text']}")
                    elif item["type"] == "audio":
                        audio = load_audio(item['audio'])
                        for i in range(0, audio.shape[0], 16000 * 25):
                            mel = log_mel_spectrogram(audio[i:i+16000*25], n_mels=128, padding=479)
                            mels.append(mel)
                            audio_tokens = "<audio_patch>" * compute_token_num(mel.shape[1])
                            results.append(f"<audio_start>{audio_tokens}<audio_end>")
                    elif item["type"] == "token":
                        results.append(item["token"])
                if msg.get('eot', True):
                    results.append('<|EOT|>')
            elif content is None:
                results.append(f"<|BOT|>{role}\n")
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        return results, mels


if __name__ == '__main__':
    from token2wav import Token2wav
    import argparse
    import json
    import os


    parser = argparse.ArgumentParser(description="StepAudio2 inference")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--bpe_tokenizer_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bpe_mapping_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--token2wav_path", type=str, required=True, help="Original Step-Audio-2-mini token2wav directory.")
    parser.add_argument("--prompt_wav", type=str, default=str(Path(__file__).resolve().parent / "assets" / "default_female.wav"))
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

    args = parser.parse_args()


    model = StepAudio2(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )
    token2wav = Token2wav(args.token2wav_path)


    jsonl_path = args.jsonl_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    
    results_path = os.path.join(output_dir, 'inference_results.jsonl')
    f_out = open(results_path, 'w', encoding='utf-8')
    
    for idx, data in enumerate(tqdm(lines, total=len(lines), desc="infrencing")):
        
        # Extract input text from messages
        messages_data = data['messages']
        input_text = ""
        input_audio = None
        system_prompt = next((m["content"] for m in messages_data if m["role"] == "system"), "You are a helpful assistant.")
        
        # Extract text content from the human role
        for msg in messages_data:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            if msg['role'] in ('human', 'user'):
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        if item['type'] == 'text':
                            input_text = item['text']
                        if item['type'] == 'audio':
                            input_audio = item['audio']
                elif isinstance(msg['content'], str):
                    input_text = msg['content']
                break
        
        if input_audio is None:
            raise ValueError(f"Input row {idx + 1} requires a human/user audio item")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": [
                {"type": "text", "text": input_text},
                {"type": "audio", "audio": input_audio}
            ]},
            {"role": "assistant", "content": "<tts_start>", "eot": False},  # Generate a speech response
        ]
        
        # Run inference
        try:
            tokens, text, audio_tokens, stats = model(messages, max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, do_sample=False, num_beams=1)
            
            # Generate audio
            if len(audio_tokens) > 0:
                audio = token2wav(audio_tokens, prompt_wav=args.prompt_wav)
                output_wav_path = os.path.join(output_dir, f'output_{idx+1}.wav')
                with open(output_wav_path, 'wb') as f:
                    f.write(audio)
            else:
                print("No audio tokens generated")
            
            # Save results
            record = {
                "index": idx + 1,
                "input_text": input_text,
                "input_audio": input_audio,
                "generated_text": text,
                "generated_tokens": tokens,
                "final_audio_tokens": audio_tokens,
                "audio_tokens_bpe_len": stats["audio_tokens_bpe_len"],
                "audio_tokens_decoded_len": stats["audio_tokens_decoded_len"],
                "audio_expand_ratio": stats["audio_expand_ratio"],
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            
        except Exception as e:
            error_record = {
                "index": idx + 1,
                "error": str(e)
            }
            f_out.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            f_out.flush()
    
    f_out.close()


    class CompactListEncoder(json.JSONEncoder):
        def encode(self, obj):
            if isinstance(obj, dict):
                items = []
                for k, v in obj.items():
                    if k in ["generated_tokens", "final_audio_tokens"]:
                        # Important: keep these fields on one line
                        v_str = json.dumps(v, separators=(",", ":"))
                    else:
                        v_str = self.encode(v)
                    items.append(f'"{k}": {v_str}')
                return "{\n  " + ",\n  ".join(items) + "\n}"
            
            elif isinstance(obj, list):
                return "[\n" + ",\n".join("  " + self.encode(i) for i in obj) + "\n]"
            
            else:
                return json.dumps(obj, ensure_ascii=False)


    data = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    json_path = os.path.join(output_dir, 'inference_results.json')
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(CompactListEncoder(ensure_ascii=False).encode(data))
    print(f"\n{'='*50}")
    print(f"Inference test completed! Results saved to: {json_path}")
