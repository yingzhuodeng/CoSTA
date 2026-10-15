import argparse
import json
from pathlib import Path
from typing import List

# Keep this consistent with the training script.
BASE = 0x4E00
MAX_ID = 6655

def char_to_id(ch: str) -> int:
    v = ord(ch) - BASE
    if not (0 <= v <= MAX_ID):
        raise ValueError(f"decoded char out of range: U+{ord(ch):X}, got v={v}")
    return v

def token_to_ints(tok: str) -> List[int]:
    # tok may contain one or more characters; each character maps to one original integer token.
    return [char_to_id(ch) for ch in tok]

def main(tokenizer_json: str, out_path: str):
    tokenizer_json = Path(tokenizer_json)
    out_path = Path(out_path)

    with tokenizer_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # In tokenizers BPE models, data["model"]["vocab"] is {token_str: token_id}.
    vocab = data["model"]["vocab"]

    # Sort by token_id for easier inspection.
    items = sorted(vocab.items(), key=lambda kv: kv[1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for tok_str, tok_id in items:
            try:
                ints = token_to_ints(tok_str)
            except Exception as e:
                # If a token contains unmapped characters, print it for inspection.
                print(f"Warning: cannot decode token {tok_str!r} (id={tok_id}): {e}")
                continue

            rec = {
                "token_id": tok_id,      # BPE id, for example 13147.
                "token_str": tok_str,    # Raw string from tokenizer.json.
                "pieces": ints,          # Corresponding original integer sequence, for example [2, 373].
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote decoded vocab to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BPE IDs and their original speech-token expansions.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.tokenizer, args.output)
