import json
from pathlib import Path
from typing import List

# 保持和你训练时一致！
BASE = 0x4E00
# MAX_ID = 6655
MAX_ID = 16382

def char_to_id(ch: str) -> int:
    v = ord(ch) - BASE
    if not (0 <= v <= MAX_ID):
        raise ValueError(f"decoded char out of range: U+{ord(ch):X}, got v={v}")
    return v

def token_to_ints(tok: str) -> List[int]:
    # tok 可能是 1 个或多个字符，每个字符对应一个原始整数 token
    return [char_to_id(ch) for ch in tok]

def main(tokenizer_json: str, out_path: str):
    tokenizer_json = Path(tokenizer_json)
    out_path = Path(out_path)

    with tokenizer_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # tokenizers 的 BPE 模型结构：data["model"]["vocab"] 是 {token_str: token_id}
    vocab = data["model"]["vocab"]

    # 按 token_id 排序，方便查看
    items = sorted(vocab.items(), key=lambda kv: kv[1])

    with out_path.open("w", encoding="utf-8") as out:
        for tok_str, tok_id in items:
            try:
                ints = token_to_ints(tok_str)
            except Exception as e:
                # 如果有某些不是你映射出来的字符（理论上不该有），可以先打印出来看看
                print(f"Warning: cannot decode token {tok_str!r} (id={tok_id}): {e}")
                continue

            rec = {
                "token_id": tok_id,      # BPE 里的 id（比如 13147）
                "token_str": tok_str,    # tokenizer.json 里的原始字符串（如 "丂仵"）
                "pieces": ints,          # 对应的原始整数序列（如 [2, 373]）
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote decoded vocab to: {out_path}")

if __name__ == "__main__":
    # tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_80k.json"
    # out_path = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_80k_vocab_int.jsonl"
    # tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_40k.json"
    # out_path = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_40k_vocab_int.jsonl"
    # tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_20k.json"
    # out_path = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_20k_vocab_int.jsonl"
    # tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_10k.json"
    # out_path = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_10k_vocab_int.jsonl"

    # tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/en_train_res_80k.json"
    # out_path = "/data/yjdeng/output/tokcom/model/ins/en_train_res_80k_vocab_int.jsonl"

    # tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/mix_train_res_80k.json"
    # out_path = "/data/yjdeng/output/tokcom/model/ins/mix_train_res_80k_vocab_int.jsonl"


    # tokenizer_json = "/data/yjdeng/output/tokcom_glm/model/ins/zh_train_res_80k.json"
    # out_path = "/data/yjdeng/output/tokcom_glm/model/ins/zh_train_res_80k_vocab_int.jsonl"

    tokenizer_json = "/data/yjdeng/output/tokcom_glm/model/ins/zh_train_selected_res_80k.json"
    out_path = "/data/yjdeng/output/tokcom_glm/model/ins/zh_train_selected_res_80k_vocab_int.jsonl"
    tokenizer_json = "/data/yjdeng/output/tokcom_glm/model/ins/en_train_selected_res_80k.json"
    out_path = "/data/yjdeng/output/tokcom_glm/model/ins/en_train_selected_res_80k_vocab_int.jsonl"

    main(tokenizer_json, out_path)
