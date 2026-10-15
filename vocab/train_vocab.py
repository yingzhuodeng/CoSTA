import ast
import argparse
from pathlib import Path
from typing import List, Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Choose a large enough Unicode range so ids 0..6655 map to unique characters.
BASE = 0x4E00  # Start of the mapped Unicode range.
MAX_ID = 6655

def id_to_char(i: int) -> str:
    if not (0 <= i <= MAX_ID):
        raise ValueError(f"token id out of range: {i}")
    return chr(BASE + i)

def char_to_id(ch: str) -> int:
    v = ord(ch) - BASE
    if not (0 <= v <= MAX_ID):
        raise ValueError(f"decoded char out of range: U+{ord(ch):X}")
    return v

def ints_to_text(seq: List[int]) -> str:
    return "".join(id_to_char(i) for i in seq)

def text_to_ints(s: str) -> List[int]:
    # Trained BPE tokens are concatenations of mapped characters; decoding keeps those characters.
    return [char_to_id(ch) for ch in s]

def iter_lists_from_file(path: str) -> Iterable[List[int]]:
    """
    Read the raw data file, with one list per line, for example:
    [1822, 1614, 68, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
                if not isinstance(obj, list):
                    raise ValueError("not a list")
                yield [int(x) for x in obj]
            except Exception as e:
                raise RuntimeError(f"Parse error at line {line_no}: {e}\nLINE={line[:200]}") from e

def build_corpus_txt(raw_list_file: str, corpus_txt: str):
    """
    Convert integer sequences into Unicode string sequences, one per line,
    for BPE training.
    """
    out = Path(corpus_txt)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fw:
        for seq in iter_lists_from_file(raw_list_file):
            fw.write(ints_to_text(seq) + "\n")

def train_bpe_from_raw_list(
    raw_list_file: str,
    save_tokenizer_json: str,
    vocab_size: int = 16000,
    min_frequency: int = 2,
):
    """
    Option B: train BPE directly from raw_list_file, where each line is a list[int].
    - Does not write an intermediate corpus file.
    - Uses train_from_iterator to avoid any impact from '\n'.
    - Does not use special_tokens.
    """
    # Do not set unk_token, so no extra token is introduced.
    tokenizer = Tokenizer(BPE(unk_token=None))

    # Force all base symbols into the initial alphabet: 6656 characters for ids 0..6655.
    initial_alphabet = [id_to_char(i) for i in range(MAX_ID + 1)]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=initial_alphabet,
    )

    def sentence_iterator():
        for seq in iter_lists_from_file(raw_list_file):
            # Each yielded sample excludes '\n'.
            yield ints_to_text(seq)

    tokenizer.train_from_iterator(sentence_iterator(), trainer=trainer)
    Path(save_tokenizer_json).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_tokenizer_json)
    return tokenizer

def load_tokenizer(tokenizer_json: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_json)

def encode_ints(tokenizer: Tokenizer, seq: List[int]) -> List[int]:
    text = ints_to_text(seq)
    return tokenizer.encode(text).ids

def decode_to_ints(tokenizer: Tokenizer, ids: List[int]) -> List[int]:
    pieces = []
    for i in ids:
        tok = tokenizer.id_to_token(i)
        pieces.append(tok)
    s = "".join(pieces)
    return text_to_ints(s)

def compression_stats(tokenizer: Tokenizer, raw_list_file: str, max_lines: int = 2000):
    """
    Estimate compression ratio: raw token count vs. BPE token count.
    """
    total_raw = 0
    total_bpe = 0
    n = 0
    for seq in iter_lists_from_file(raw_list_file):
        n += 1
        total_raw += len(seq)
        total_bpe += len(encode_ints(tokenizer, seq))
        if n >= max_lines:
            break
    ratio = total_bpe / max(total_raw, 1)
    print(f"[Stats on first {n} lines]")
    print(f"  raw tokens: {total_raw}")
    print(f"  bpe tokens: {total_bpe}")
    print(f"  bpe/raw ratio: {ratio:.4f}  (lower is better)")


def main():
    parser = argparse.ArgumentParser(description="Train a lossless BPE vocabulary over S3Tokenizer V2 speech IDs.")
    parser.add_argument("--input", required=True, help="One list of original speech-token IDs per line.")
    parser.add_argument("--output", required=True, help="Output BPE tokenizer JSON.")
    parser.add_argument("--vocab-size", type=int, default=86656, help="Total size: 6,656 base units + 80,000 added units by default.")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--stats-max-lines", type=int, default=2000)
    args = parser.parse_args()
    if args.vocab_size < MAX_ID + 1:
        parser.error("--vocab-size must be at least 6656")
    if args.stats_max_lines < 1:
        parser.error("--stats-max-lines must be positive")
    tokenizer = train_bpe_from_raw_list(args.input, args.output, args.vocab_size, args.min_frequency)
    compression_stats(tokenizer, args.input, max_lines=args.stats_max_lines)
    sample = next(iter_lists_from_file(args.input))
    assert decode_to_ints(tokenizer, encode_ints(tokenizer, sample)) == sample
    print("Round-trip check passed.")
    print(f"Actual vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
