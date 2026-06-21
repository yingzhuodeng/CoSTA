import ast
from cgi import print_arguments
from pathlib import Path
from typing import List, Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# 选择一个足够大的 Private Use 区间，保证 0..6655 都能映射到唯一字符
BASE = 0x4E00  # Supplementary Private Use Area-A 起点
# MAX_ID = 6655
MAX_ID = 16382

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
    # 注意：训练出的 BPE token 是若干字符的拼接，decode 后仍是这些字符
    return [char_to_id(ch) for ch in s]

def iter_lists_from_file(path: str) -> Iterable[List[int]]:
    """
    读取你的原始数据文件：每行形如
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
    把整数序列转换成“unicode 字符串序列”，每行一句，供 BPE 训练使用
    """
    out = Path(corpus_txt)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fw:
        for seq in iter_lists_from_file(raw_list_file):
            fw.write(ints_to_text(seq) + "\n")

# def train_bpe(corpus_txt: str, save_tokenizer_json: str,
#               vocab_size: int = 16000, min_frequency: int = 2):
#     """
#     训练 BPE：使用 train_from_iterator，逐行喂入样本，从而彻底排除 '\n' 对训练的影响
#     """
#     special_tokens = ["[PAD]", "[UNK]"]
#     tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

#     # 强制把所有基础符号都放进初始 alphabet（即 6656 个“字符”）
#     initial_alphabet = [id_to_char(i) for i in range(MAX_ID + 1)]

#     trainer = BpeTrainer(
#         vocab_size=vocab_size,
#         min_frequency=min_frequency,
#         special_tokens=special_tokens,
#         initial_alphabet=initial_alphabet,
#     )

#     def line_iterator():
#         with open(corpus_txt, "r", encoding="utf-8") as f:
#             for line in f:
#                 # 只去掉行尾换行，避免 '\n' 被当作训练字符
#                 line = line.rstrip("\r\n")
#                 if line:
#                     yield line

#     tokenizer.train_from_iterator(line_iterator(), trainer=trainer)
#     tokenizer.save(save_tokenizer_json)
#     return tokenizer

def train_bpe_from_raw_list(
    raw_list_file: str,
    save_tokenizer_json: str,
    vocab_size: int = 16000,
    min_frequency: int = 2,
):
    """
    方案B：直接从 raw_list_file（每行一个 list[int]）训练 BPE。
    - 不落地中间语料文件
    - 使用 train_from_iterator，彻底排除 '\n' 影响
    - 不使用 special_tokens
    """
    # 不设置 unk_token，避免引入任何额外 token
    tokenizer = Tokenizer(BPE(unk_token=None))

    # 强制把所有基础符号都放进初始 alphabet（即 0..6655 对应的 6656 个字符）
    initial_alphabet = [id_to_char(i) for i in range(MAX_ID + 1)]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=initial_alphabet,
    )

    def sentence_iterator():
        for seq in iter_lists_from_file(raw_list_file):
            # 注意：这里 yield 的每个样本都不包含 '\n'
            yield ints_to_text(seq)

    tokenizer.train_from_iterator(sentence_iterator(), trainer=trainer)
    tokenizer.save(save_tokenizer_json)
    return tokenizer

def load_tokenizer(tokenizer_json: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_json)

def encode_ints(tokenizer: Tokenizer, seq: List[int]) -> List[int]:
    text = ints_to_text(seq)
    return tokenizer.encode(text).ids

# def decode_to_ints(tokenizer: Tokenizer, ids: List[int]) -> List[int]:
#     # 不依赖 tokenizer.decode 的空格规则，直接用 token 字符串拼起来更稳
#     pieces = []
#     for i in ids:
#         tok = tokenizer.id_to_token(i)
#         if tok in ("[PAD]", "[UNK]"):
#             # [UNK] 理论上不应该出现（我们已强制 initial_alphabet），出现了说明输入有越界或数据问题
#             raise ValueError(f"Special token {tok} appeared in decode; check your data.")
#         pieces.append(tok)
#     s = "".join(pieces)
#     return text_to_ints(s)

def decode_to_ints(tokenizer: Tokenizer, ids: List[int]) -> List[int]:
    pieces = []
    for i in ids:
        tok = tokenizer.id_to_token(i)
        pieces.append(tok)
    s = "".join(pieces)
    return text_to_ints(s)

def compression_stats(tokenizer: Tokenizer, raw_list_file: str, max_lines: int = 2000):
    """
    简单估计压缩率：原始 token 数 vs BPE token 数
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

# train
# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_80k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=86656,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=231)
#     """
#     [Stats on first 49664 lines]                                                                                                                                                     
#     raw tokens: 24385965                                                                                                                                                           
#     bpe tokens: 15048107                                                                                                                                                           
#     bpe/raw ratio: 0.6171  (lower is better)                                                                                                                                       
#     [Stats on first 231 lines]                                                                                                                                                       
#     raw tokens: 111904                                                                                                                                                             
#     bpe tokens: 69707                                                                                                                                                              
#     bpe/raw ratio: 0.6229  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/tokcom/data/emo/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/emo/token/zh_test_res.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=325)
#     """
#     [Stats on first 49664 lines]                                                                                                                                                     
#     raw tokens: 14954624                                                                                                                                                           
#     bpe tokens: 9058589                                                                                                                                                            
#     bpe/raw ratio: 0.6057  (lower is better)                                                                                                                                       
#     [Stats on first 325 lines]                                                                                                                                                       
#     raw tokens: 104326                                                                                                                                                             
#     bpe tokens: 64184                                                                                                                                                              
#     bpe/raw ratio: 0.6152  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/emilia/data/token_vc/zh_vc_train.txt"
#     test_list_file = "/data/yjdeng/output/emilia/data/token_vc/zh_vc_test.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=945920)
#     compression_stats(tokenizer, test_list_file, max_lines=328)
#     """
#     [Stats on first 945920 lines]                                                                                                                                                    
#     raw tokens: 208833547                                                                                                                                                          
#     bpe tokens: 153886156                                                                                                                                                          
#     bpe/raw ratio: 0.7369  (lower is better)
#     [Stats on first 328 lines]
#     raw tokens: 77961
#     bpe tokens: 57274
#     bpe/raw ratio: 0.7346  (lower is better)
#     """


# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_40k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=46656,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=231)
#     """
#     [Stats on first 49664 lines]                                                                                                                                                     
#     raw tokens: 24385965                                                                                                                                                           
#     bpe tokens: 16050923
#     bpe/raw ratio: 0.6582  (lower is better)
#     [Stats on first 231 lines]
#     raw tokens: 111904
#     bpe tokens: 74026
#     bpe/raw ratio: 0.6615  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/tokcom/data/emo/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/emo/token/zh_test_res.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=325)
#     """
#     [Stats on first 49664 lines]
#     raw tokens: 14954624
#     bpe tokens: 9607827
#     bpe/raw ratio: 0.6425  (lower is better)
#     [Stats on first 325 lines]
#     raw tokens: 104326
#     bpe tokens: 68195
#     bpe/raw ratio: 0.6537  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/emilia/data/token_vc/zh_vc_train.txt"
#     test_list_file = "/data/yjdeng/output/emilia/data/token_vc/zh_vc_test.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=945920)
#     compression_stats(tokenizer, test_list_file, max_lines=328)
#     """
#     [Stats on first 945920 lines]
#     raw tokens: 208833547
#     bpe tokens: 161578134
#     bpe/raw ratio: 0.7737  (lower is better)
#     [Stats on first 328 lines]
#     raw tokens: 77961
#     bpe tokens: 60043
#     bpe/raw ratio: 0.7702  (lower is better)
#     """


# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_20k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=26656,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=231)
#     """
#     [Stats on first 49664 lines]
#     raw tokens: 24385965
#     bpe tokens: 16992212
#     bpe/raw ratio: 0.6968  (lower is better)
#     [Stats on first 231 lines]
#     raw tokens: 111904
#     bpe tokens: 78099
#     bpe/raw ratio: 0.6979  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/tokcom/data/emo/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/emo/token/zh_test_res.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=325)
#     """
#     [Stats on first 49664 lines]
#     raw tokens: 14954624
#     bpe tokens: 10131478
#     bpe/raw ratio: 0.6775  (lower is better)
#     [Stats on first 325 lines]
#     raw tokens: 104326
#     bpe tokens: 71838
#     bpe/raw ratio: 0.6886  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/emilia/data/token_vc/zh_vc_train.txt"
#     test_list_file = "/data/yjdeng/output/emilia/data/token_vc/zh_vc_test.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=945920)
#     compression_stats(tokenizer, test_list_file, max_lines=328)
#     """
#     [Stats on first 945920 lines]
#     raw tokens: 208833547
#     bpe tokens: 168451671
#     bpe/raw ratio: 0.8066  (lower is better)
#     [Stats on first 328 lines]
#     raw tokens: 77961
#     bpe tokens: 62595
#     bpe/raw ratio: 0.8029  (lower is better)
#     """


# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_10k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=16656,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=231)
#     """
#     [Stats on first 49664 lines]
#     raw tokens: 24385965
#     bpe tokens: 17845772
#     bpe/raw ratio: 0.7318  (lower is better)
#     [Stats on first 231 lines]
#     raw tokens: 111904
#     bpe tokens: 81993
#     bpe/raw ratio: 0.7327  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/tokcom/data/emo/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/emo/token/zh_test_res.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=325)
#     """
#     [Stats on first 49664 lines]
#     raw tokens: 14954624
#     bpe tokens: 10615559
#     bpe/raw ratio: 0.7099  (lower is better)
#     [Stats on first 325 lines]
#     raw tokens: 104326
#     bpe tokens: 75251
#     bpe/raw ratio: 0.7213  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/emilia/data/token_vc/zh_vc_train.txt"
#     test_list_file = "/data/yjdeng/output/emilia/data/token_vc/zh_vc_test.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=945920)
#     compression_stats(tokenizer, test_list_file, max_lines=328)
#     """
#     [Stats on first 945920 lines]
#     raw tokens: 208833547
#     bpe tokens: 174406879
#     bpe/raw ratio: 0.8351  (lower is better)
#     [Stats on first 328 lines]
#     raw tokens: 77961
#     bpe tokens: 64757
#     bpe/raw ratio: 0.8306  (lower is better)
#     """


# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom/data/ins/token/en_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/ins/token/en_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/en_train_res_80k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=86656,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49408)
#     compression_stats(tokenizer, test_list_file, max_lines=290)
#     """
#     [Stats on first 49408 lines]
#     raw tokens: 28787809
#     bpe tokens: 18452603
#     bpe/raw ratio: 0.6410  (lower is better)
#     [Stats on first 290 lines]
#     raw tokens: 176333
#     bpe tokens: 113525
#     bpe/raw ratio: 0.6438  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/tokcom/data/emo/token/en_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/emo/token/en_test_res.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=48896)
#     compression_stats(tokenizer, test_list_file, max_lines=391)
#     """
#     [Stats on first 48896 lines]
#     raw tokens: 22146396
#     bpe tokens: 14092347
#     bpe/raw ratio: 0.6363  (lower is better)
#     [Stats on first 391 lines]
#     raw tokens: 181766
#     bpe tokens: 115830
#     bpe/raw ratio: 0.6372  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/emilia/data/token_vc/en_vc_train.txt"
#     test_list_file = "/data/yjdeng/output/emilia/data/token_vc/en_vc_test.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=846592)
#     compression_stats(tokenizer, test_list_file, max_lines=364)
#     """
#     [Stats on first 846592 lines]
#     raw tokens: 198204512
#     bpe tokens: 148834251
#     bpe/raw ratio: 0.7509  (lower is better)
#     [Stats on first 364 lines]
#     raw tokens: 88764
#     bpe tokens: 66481
#     bpe/raw ratio: 0.7490  (lower is better)
#     """


# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom/data/ins/token/mix_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/ins/token/mix_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/mix_train_res_80k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=86656,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=99072)
#     compression_stats(tokenizer, test_list_file, max_lines=521)
#     """
#     [Stats on first 99072 lines]
#     raw tokens: 53173774
#     bpe tokens: 34633417
#     bpe/raw ratio: 0.6513  (lower is better)
#     [Stats on first 521 lines]
#     raw tokens: 288237
#     bpe tokens: 188550
#     bpe/raw ratio: 0.6541  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/tokcom/data/emo/token/mix_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom/data/emo/token/mix_test_res.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=98560)
#     compression_stats(tokenizer, test_list_file, max_lines=716)
#     """
#     [Stats on first 98560 lines]
#     raw tokens: 37101020
#     bpe tokens: 23745127
#     bpe/raw ratio: 0.6400  (lower is better)
#     [Stats on first 716 lines]
#     raw tokens: 286092
#     bpe tokens: 184497
#     bpe/raw ratio: 0.6449  (lower is better)
#     """
#     train_list_file = "/data/yjdeng/output/emilia/data/token_vc/mix_vc_train.txt"
#     test_list_file = "/data/yjdeng/output/emilia/data/token_vc/mix_vc_test.txt"
#     compression_stats(tokenizer, train_list_file, max_lines=1792512)
#     compression_stats(tokenizer, test_list_file, max_lines=692)
#     """
#     [Stats on first 1792512 lines]
#     raw tokens: 407038059
#     bpe tokens: 305016053
#     bpe/raw ratio: 0.7494  (lower is better)
#     [Stats on first 692 lines]
#     raw tokens: 166725
#     bpe tokens: 124691
#     bpe/raw ratio: 0.7479  (lower is better)
#     """

    # tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_80k.json"
    # # tokenizer_json = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_40k.json"
    # tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    # raw_list_file = "/data/yjdeng/output/tokcom/data/ins/token/zh_train_res.txt"

    # sample = next(iter_lists_from_file(raw_list_file))
    # ids = encode_ints(tokenizer, sample)
    # back = decode_to_ints(tokenizer, ids)
    # print("roundtrip ok:", back == sample)

    # for i, seq in enumerate(iter_lists_from_file(raw_list_file), 1):
    #     if i > 3 :
    #         break
    #     ids = encode_ints(tokenizer, seq)
    #     back = decode_to_ints(tokenizer, ids)
    #     print(f"num_{i}:\nseq:\n{seq}\nids:\n{ids}\nints:\n{back}")
    #     print(len(ids))
    #     print(len(seq))
    #     print(len(ids)/len(seq))
    #     print(seq == back)


# cross domain test
# if __name__ == "__main__":
#     # raw_list_file = "/data/yjdeng/tokcompress/data/instruct/token/en+zh_res.txt"
#     # tokenizer_json = "/data/yjdeng/tokcompress/model/instruct/en+zh_res_50k.json"
#     # raw_list_file = "/data/yjdeng/tokcompress/data/instruct/token/en_res.txt"
#     # tokenizer_json = "/data/yjdeng/tokcompress/model/instruct/en_res_50k.json"
#     raw_list_file = "/data/yjdeng/tokcompress/data/instruct/token/zh_res.txt"
#     tokenizer_json = "/data/yjdeng/tokcompress/model/instruct/zh_res_50k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=raw_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=56656,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)

#     # compression_stats(tokenizer, raw_list_file, max_lines=68686)
#     # compression_stats(tokenizer, raw_list_file, max_lines=30636)
#     compression_stats(tokenizer, raw_list_file, max_lines=38050)

# if __name__ == "__main__":
#     # tokenizer_json = "/data/yjdeng/tokcompress/model/instruct/en+zh_res_50k.json"
#     # tokenizer_json = "/data/yjdeng/tokcompress/model/instruct/en_res_50k.json"
#     tokenizer_json = "/data/yjdeng/tokcompress/model/instruct/zh_res_50k.json"
#     print(tokenizer_json)
#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)

#     raw_list_file = "/data/yjdeng/emilia/data/token_vc/en_vc_part1+2.txt"
#     print(raw_list_file)
#     compression_stats(tokenizer, raw_list_file, max_lines=846956)
#     # raw_list_file = "/data/yjdeng/tokcompress/data/instruct/token/en+zh_res.txt"
#     # print(raw_list_file)
#     # compression_stats(tokenizer, raw_list_file, max_lines=68686)
#     # raw_list_file = "/data/yjdeng/tokcompress/data/instruct/token/en+zh_res.txt"
#     # print(raw_list_file)
#     # compression_stats(tokenizer, raw_list_file, max_lines=68686)
#     # raw_list_file = "/data/yjdeng/tokcompress/data/instruct/token/en_res.txt"
#     # print(raw_list_file)
#     # compression_stats(tokenizer, raw_list_file, max_lines=30636)
#     # raw_list_file = "/data/yjdeng/tokcompress/data/instruct/token/zh_res.txt"
#     # print(raw_list_file)
#     # compression_stats(tokenizer, raw_list_file, max_lines=38050)
#     # raw_list_file = "/data/yjdeng/tokcompress/data/instruct/token/en+zh_res.txt"
#     # print(raw_list_file)
#     # compression_stats(tokenizer, raw_list_file, max_lines=68686)
#     # raw_list_file = "/data/yjdeng/tokcompress/data/emo/token/en_res.txt"
#     # print(raw_list_file)
#     # compression_stats(tokenizer, raw_list_file, max_lines=49368)
#     # raw_list_file = "/data/yjdeng/tokcompress/data/emo/token/zh_res.txt"
#     # print(raw_list_file)
#     # compression_stats(tokenizer, raw_list_file, max_lines=49991)
#     # raw_list_file = "/data/yjdeng/tokcompress/data/emo/token/en+zh_res.txt"
#     # print(raw_list_file)
#     # compression_stats(tokenizer, raw_list_file, max_lines=99359)

# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/zh_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom_glm/model/ins/zh_train_res_100k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=116383,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=231)
#     """
#     [Stats on first 49664 lines]
#     raw tokens: 12207677
#     bpe tokens: 8454602
#     bpe/raw ratio: 0.6926  (lower is better)
#     [Stats on first 231 lines]
#     raw tokens: 56037
#     bpe tokens: 39779
#     bpe/raw ratio: 0.7099  (lower is better)
#     """

# glm-4-voice
# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/zh_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom_glm/model/ins/zh_train_res_80k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=96383,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=231)
#     """
#     [Stats on first 49664 lines]
#     raw tokens: 12207677
#     bpe tokens: 8659189
#     bpe/raw ratio: 0.7093  (lower is better)
#     [Stats on first 231 lines]
#     raw tokens: 56037
#     bpe tokens: 40565
#     bpe/raw ratio: 0.7239  (lower is better)
#     """

# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/zh_train_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/zh_test_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom_glm/model/ins/zh_train_res_100k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=116383,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49664)
#     compression_stats(tokenizer, test_list_file, max_lines=231)
#     """
#     [Stats on first 49664 lines]
#     raw tokens: 12207677
#     bpe tokens: 8454602
#     bpe/raw ratio: 0.6926  (lower is better)
#     [Stats on first 231 lines]
#     raw tokens: 56037
#     bpe tokens: 39779
#     bpe/raw ratio: 0.7099  (lower is better)
#     """

# if __name__ == "__main__":
#     train_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/zh_train_selected_res.txt"
#     test_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/zh_test_selected_res.txt"

#     tokenizer_json = "/data/yjdeng/output/tokcom_glm/model/ins/zh_train_selected_res_80k.json"

#     tokenizer = train_bpe_from_raw_list(
#         raw_list_file=train_list_file,
#         save_tokenizer_json=tokenizer_json,
#         vocab_size=96383,
#         min_frequency=2,
#     )

#     tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
#     compression_stats(tokenizer, train_list_file, max_lines=49408)
#     compression_stats(tokenizer, test_list_file, max_lines=231)
#     """
# [Stats on first 49408 lines]
#   raw tokens: 12143932
#   bpe tokens: 8613958
#   bpe/raw ratio: 0.7093  (lower is better)
# [Stats on first 231 lines]
#   raw tokens: 56037
#   bpe tokens: 40568
#   bpe/raw ratio: 0.7240  (lower is better)
#     """

if __name__ == "__main__":
    train_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/en_train_selected_res.txt"
    test_list_file = "/data/yjdeng/output/tokcom_glm/data/ins/token/en_test_selected_res.txt"

    tokenizer_json = "/data/yjdeng/output/tokcom_glm/model/ins/en_train_selected_res_80k.json"

    tokenizer = train_bpe_from_raw_list(
        raw_list_file=train_list_file,
        save_tokenizer_json=tokenizer_json,
        vocab_size=96383,
        min_frequency=2,
    )

    tokenizer = load_tokenizer(tokenizer_json=tokenizer_json)
    
    compression_stats(tokenizer, train_list_file, max_lines=49152)
    compression_stats(tokenizer, test_list_file, max_lines=290)
    """
[Stats on first 49152 lines]
  raw tokens: 14333358
  bpe tokens: 10265119
  bpe/raw ratio: 0.7162  (lower is better)
[Stats on first 290 lines]
  raw tokens: 88304
  bpe tokens: 63696
  bpe/raw ratio: 0.7213  (lower is better)
    """
