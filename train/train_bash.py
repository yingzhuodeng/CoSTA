import os
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, HfArgumentParser
from my_trainer import MyTrainer
from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels


# global configuration
@dataclass
class SA2Arguments:
    WANDB_PATH: str = field(default="/data/yjdeng/ltbs/code/train_sa2/output/step_audio2_sft")
    MODEL_PATH: str = field(default="/data/yjdeng/Step/Step-Audio-2-mini")
    TOKENIZER_PATH: str = field(default="/data/yjdeng/Step/Step-Audio-2-mini")
    PROCESSED_FILE: str = field(default="/data/yjdeng/ltbs/data/try/processed_messages_speech.jsonl")

AUDIO_PATCH_ID = 151690
IGNORE_INDEX = -100
MAX_LENGTH = 16384

# load data: reading messages from offline pre-processed JSONL
def load_processed_messages(processed_file: str) -> List[List[Dict[str, Any]]]:
    """
    Messages are loaded from the processed_messages.jsonl file
    Each line is {"messages": dialog}
    """
    messages = []
    with open(processed_file, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            messages.append(data["messages"])
    return messages


def apply_chat_template(message: list):
    results = []
    mels = []
    for msg in message:
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


def find_last_sublist(lst, sub):
    if len(sub) == 0 or len(lst) < len(sub):
        return None
    last = None
    for i in range(0, len(lst) - len(sub) + 1):
        if lst[i:i + len(sub)] == sub:
            last = i
    return last


# Dataset encapsulation: wrapping a list of messages
class StepAudio2Dataset(Dataset):
    def __init__(self, messages: List[List[Dict[str, Any]]]):
        self.messages = messages

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        # Trainer expects to return a dict by default; here, "messages" is used to store each dialogue
        return {"messages": self.messages[idx]}


@dataclass
class StepAudio2Collator:
    tokenizer: Any
    max_length: int = MAX_LENGTH

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        batch: List[{"messages": dialog}, ...]
        return:
            {
              "input_ids": [B, L],
              "attention_mask": [B, L],
              "labels": [B, L],
              "wavs": [N_mel_seg, n_mels, T] or None,
              "wav_lens": [N_mel_seg] or None
            }
        """
        input_ids_list, attn_list = [], []
        labels_list = []
        wavs_per_sample: List[List[torch.Tensor]] = []

        # 1. apply_chat_template + tokenize for each sample within the batch
        for example in batch:
            message = example["messages"]
            msgs, mels = apply_chat_template(message)

            # tokenize prompts
            ids_pieces = []
            for msg in msgs:
                if isinstance(msg, str):
                    ids_pieces.append(
                        self.tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"]
                    )
                elif isinstance(msg, list):
                    ids_pieces.append(torch.tensor([msg], dtype=torch.int32))
                else:
                    raise ValueError(f"Unsupported content type: {type(msg)}")
            # spliced ​​into a complete sequence
            input_ids = torch.cat(ids_pieces, dim=-1).squeeze(0)
            input_ids = input_ids[: self.max_length]

            attention_mask = torch.ones_like(input_ids)

            labels = input_ids.clone()
            labels[input_ids == AUDIO_PATCH_ID] = IGNORE_INDEX
            # mask process
            input_ids_list_tmp = input_ids.tolist()
            speech_prefix = self.tokenizer("<|BOT|>assistant\n<tts_start>", add_special_tokens=False)["input_ids"]
            text_prefix = self.tokenizer("<|BOT|>assistant\n", add_special_tokens=False)["input_ids"]
            start = find_last_sublist(input_ids_list_tmp, speech_prefix) # first speech
            if start is not None:
                start = start + len(speech_prefix)
                labels[:start] = IGNORE_INDEX
            else:
                start = find_last_sublist(input_ids_list_tmp, text_prefix) # then text
                if start is not None:
                    start = start + len(text_prefix)
                    labels[:start] = IGNORE_INDEX
                else:
                    labels[:] = IGNORE_INDEX

            input_ids_list.append(input_ids)
            attn_list.append(attention_mask)
            labels_list.append(labels)
            wavs_per_sample.append(mels)

        # 2. pad
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)

        # 3. processing wavs: flatten all mel segments in the batch into a list
        wavs_flat: List[torch.Tensor] = [t for inner in wavs_per_sample for t in inner]
        if len(wavs_flat) == 0:
            wavs = None
            wav_lens = None
        else:
            wavs, wav_lens = padding_mels(wavs_flat)

        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "wavs": wavs,
            "wav_lens": wav_lens,
        }
        # print(batch_out)
        return batch_out


# train main
def main():
    # 0. injecting SA2Arguments + TrainingArguments from bash
    parser = HfArgumentParser((SA2Arguments, TrainingArguments))
    sa2_args, training_args = parser.parse_args_into_dataclasses()

    # set wandb
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = sa2_args.WANDB_PATH

    # 1. load tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(
        sa2_args.TOKENIZER_PATH,
        trust_remote_code=True,
        padding_side="right"
    )

    # 2. load data
    messages = load_processed_messages(sa2_args.PROCESSED_FILE)
    train_dataset = StepAudio2Dataset(messages)

    # 3. load model
    model = AutoModelForCausalLM.from_pretrained(
        sa2_args.MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2"  # flash_attention_2 doesn't seem to be working.
    )


    # ---------------------------------4. set training parameters/start---------------------------------
    # for p in model.model.parameters():
    #     p.requires_grad = False

    for p in model.encoder.parameters():
        p.requires_grad = False

    for p in model.adapter.parameters():
        p.requires_grad = False

    # for p in model.lm_head.parameters():
    #     p.requires_grad = False
    
    
    # for p in model.model.embed_tokens.parameters():
    #     p.requires_grad = True

    # for p in model.lm_head.parameters():
    #     p.requires_grad = True
    
    
    # trainable, total = 0, 0
    # for name, p in model.named_parameters():
    #     num = p.numel()
    #     total += num
    #     if p.requires_grad:
    #         trainable += num
    #     # print(name, p, p.requires_grad, num)
    #     print(name, p.requires_grad, num)
    # print(f"trainable params: {trainable:,} / {total:,}")
    # ---------------------------------4. set training parameters/end---------------------------------


    # 5. define Collator
    data_collator = StepAudio2Collator(tokenizer=llm_tokenizer)

    # 6. define Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    llm_tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
