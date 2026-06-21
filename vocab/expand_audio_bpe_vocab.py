# gpt改的支持resize的版本，改过modeling文件，推荐
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ========================
# 1. 待修改的常量
# ========================

# 原始大模型路径（包含 tokenizer 和 model）
BASE_MODEL_DIR = "/data/yjdeng/output/Step/Step-Audio-2-mini"  # TODO: 改成你的路径

# 扩展后模型的保存路径
# OUTPUT_MODEL_DIR = "/data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_80k"  # TODO
# OUTPUT_MODEL_DIR = "/data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_40k"
# OUTPUT_MODEL_DIR = "/data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_20k"
OUTPUT_MODEL_DIR = "/data/yjdeng/output/Step/Step-Audio-2-mini-ins_zh_10k"

# OUTPUT_MODEL_DIR = "/data/yjdeng/output/Step/Step-Audio-2-mini-ins_en_80k"
# OUTPUT_MODEL_DIR = "/data/yjdeng/output/Step/Step-Audio-2-mini-ins_mix_80k"


# decode_vocab_to_ints.py 生成的 jsonl 文件（每行: {"token_id": int, "pieces": [int, ...], ...}）
# BPE_MAPPING_JSONL = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_80k_vocab_int.jsonl"  # TODO
# BPE_MAPPING_JSONL = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_40k_vocab_int.jsonl"
# BPE_MAPPING_JSONL = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_20k_vocab_int.jsonl"
BPE_MAPPING_JSONL = "/data/yjdeng/output/tokcom/model/ins/zh_train_res_10k_vocab_int.jsonl"

# BPE_MAPPING_JSONL = "/data/yjdeng/output/tokcom/model/ins/en_train_res_80k_vocab_int.jsonl"
# BPE_MAPPING_JSONL = "/data/yjdeng/output/tokcom/model/ins/mix_train_res_80k_vocab_int.jsonl"


# BPE 词表信息
AUDIO_BASE_SIZE = 6656       # 原始 audio 基元个数（0..6655）
# BPE_VOCAB_SIZE = 86656       # 你的 BPE-int vocab_size
# BPE_VOCAB_SIZE = 46656
# BPE_VOCAB_SIZE = 26656
BPE_VOCAB_SIZE = 16656
NEW_AUDIO_TOKENS = BPE_VOCAB_SIZE - AUDIO_BASE_SIZE

# 大模型里 audio 基元在词表中的起始位置（你说的是 151696）
AUDIO_LLM_START = 151696


# ========================
# 2. 读取 BPE -> pieces 映射
# ========================

def load_bpe_pieces(mapping_path: str):
    """
    读取 decode_vocab_to_ints.py 生成的 jsonl：
    每行: {"token_id": int, "pieces": [int, ...], ...}
    返回: {token_id: pieces}
    """
    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj["token_id"]
            pieces = obj["pieces"]
            mapping[tid] = pieces
    return mapping


# ========================
# 3. 主逻辑（兼容 tokenizer 与 embedding 大小不一致）
# ========================

def main():
    # ---- 3.1 读取 BPE mapping ----
    print("Loading BPE mapping from:", BPE_MAPPING_JSONL)
    bpe2pieces = load_bpe_pieces(BPE_MAPPING_JSONL)

    max_bpe_id = max(bpe2pieces.keys())
    min_bpe_id = min(bpe2pieces.keys())
    print(f"  bpe_id range: [{min_bpe_id}, {max_bpe_id}]")
    assert min_bpe_id == 0, "预期 bpe_id 从 0 开始"
    assert max_bpe_id + 1 >= BPE_VOCAB_SIZE, "BPE_VOCAB_SIZE 可能比真实 vocab 小"

    # ---- 3.2 加载 tokenizer & model ----
    print("Loading base tokenizer & model from:", BASE_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # 可按需改为 float16 / float32
    )

    # --- 3.3 查看 tokenizer / embedding 情况（关键） ---
    emb = model.get_input_embeddings()
    old_num_embeddings, hidden_size = emb.weight.shape

    vocab = tokenizer.get_vocab()  # {token_str: id}
    orig_tok_len = len(vocab)      # 158352（你的情况）
    orig_max_id = max(vocab.values())  # 158351

    print("Tokenizer:")
    print(f"  len(vocab) = {orig_tok_len}")
    print(f"  max_id     = {orig_max_id}")
    print("Model embedding:")
    print(f"  num_embeddings = {old_num_embeddings}")

    # 确保 tokenizer 用到的最大 id 没超过 embedding 行数
    assert orig_max_id < old_num_embeddings, (
        f"tokenizer max id {orig_max_id} >= embedding rows {old_num_embeddings}"
    )

    # 确认 audio 基元所在区间合理
    assert AUDIO_LLM_START + AUDIO_BASE_SIZE <= old_num_embeddings, (
        f"AUDIO_LLM_START({AUDIO_LLM_START}) + {AUDIO_BASE_SIZE} 超过 embedding 行数 {old_num_embeddings}"
    )

    # ---- 3.4 tokenizer 添加 20K 新 token: <audio_6656>.. <audio_26655> ----
    new_token_strings = [f"<audio_{i}>" for i in range(AUDIO_BASE_SIZE, BPE_VOCAB_SIZE)]
    print(f"Adding {len(new_token_strings)} new audio tokens to tokenizer...")
    added = tokenizer.add_tokens(new_token_strings, special_tokens=False)
    print("  actually added:", added)
    assert added == NEW_AUDIO_TOKENS, "add_tokens 添加数量与预期不一致 (不是 20000?)"

    vocab_after = tokenizer.get_vocab()
    new_tok_len = len(vocab_after)
    new_max_id = max(vocab_after.values())

    print("After add_tokens:")
    print(f"  len(vocab) = {new_tok_len}")
    print(f"  max_id     = {new_max_id}")

    # 这里兼容“embedding 比 tokenizer 大”的情况：
    # old_num_embeddings = 158720
    # new_tok_len        = 158352 + 20000 = 178352
    # new_max_id         = 158351 + 20000 = 178351
    # 新的 embedding 行数至少要覆盖 new_max_id + 1
    new_num_embeddings = max(old_num_embeddings, new_max_id + 1)
    print(f"Resizing token embeddings from {old_num_embeddings} to {new_num_embeddings} ...")
    model.resize_token_embeddings(new_num_embeddings, mean_resizing=False)

    # 重新拿 embedding & lm_head
    emb = model.get_input_embeddings()
    print("New embedding shape:", emb.weight.shape)

    out_proj = model.get_output_embeddings()
    tied = False
    if out_proj is not None:
        tied = (out_proj.weight.data.data_ptr() == emb.weight.data.data_ptr())
    print("LM head present:", out_proj is not None, "  tied with embedding:", tied)

    emb_dtype = emb.weight.dtype
    out_dtype = out_proj.weight.dtype if out_proj is not None else None

    # ---- 3.5 按 BPE pieces 平均初始化新 token 参数 ----
    print("Initializing new audio token weights from base audio tokens...")

    for bpe_id in range(AUDIO_BASE_SIZE, BPE_VOCAB_SIZE):
        pieces = bpe2pieces.get(bpe_id, None)
        if pieces is None:
            raise ValueError(f"bpe_id {bpe_id} 在 BPE 映射文件中找不到 pieces")

        if len(pieces) == 0:
            raise ValueError(f"bpe_id {bpe_id} 的 pieces 为空，数据异常")

        # 将 BPE 中的 base audio id 映射到模型 embedding 的全局 id
        base_llm_ids = [AUDIO_LLM_START + p for p in pieces]
        for gid in base_llm_ids:
            if not (0 <= gid < old_num_embeddings):
                raise ValueError(
                    f"base audio global id {gid} 越界 (0..{old_num_embeddings-1})，"
                    f"请检查 AUDIO_LLM_START / AUDIO_BASE_SIZE 设置"
                )

        # 计算 embedding 的平均
        base_emb = emb.weight.data[base_llm_ids].float()   # [n_pieces, hidden_size]
        mean_emb = base_emb.mean(dim=0)                    # [hidden_size]

        # 新 token 在 tokenizer / 模型中的 id（注意：不再假设是 old_num_embeddings 开始）
        new_token_str = f"<audio_{bpe_id}>"
        new_llm_id = tokenizer.convert_tokens_to_ids(new_token_str)

        # 写入 embedding
        emb.weight.data[new_llm_id] = mean_emb.to(emb_dtype)

        # 如果 lm_head 存在且未 tied，对 lm_head 做同样操作
        if out_proj is not None and not tied:
            base_out = out_proj.weight.data[base_llm_ids].float()
            mean_out = base_out.mean(dim=0)
            out_proj.weight.data[new_llm_id] = mean_out.to(out_dtype)

    print("All new audio token weights initialized.")

    # ---- 3.6 保存扩展后的模型和 tokenizer ----
    out_dir = Path(OUTPUT_MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Saving extended tokenizer to:", out_dir)
    tokenizer.save_pretrained(out_dir)

    print("Saving extended model to:", out_dir)
    model.save_pretrained(out_dir)

    # ---- 3.7 额外保存 BPE-id -> LLM-id 映射，方便后续推理使用 ----
    mapping_path = out_dir / "audio_bpeid_to_llmid.json"
    bpeid_to_llmid = {}

    # base 0..6655
    for bpe_id in range(AUDIO_BASE_SIZE):
        bpeid_to_llmid[bpe_id] = AUDIO_LLM_START + bpe_id

    # new 6656..26655
    for bpe_id in range(AUDIO_BASE_SIZE, BPE_VOCAB_SIZE):
        tok = f"<audio_{bpe_id}>"
        bpeid_to_llmid[bpe_id] = tokenizer.convert_tokens_to_ids(tok)

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(bpeid_to_llmid, f, ensure_ascii=False, indent=2)

    print("Saved BPE-id -> LLM-id mapping to:", mapping_path)
    print("Done.")


if __name__ == "__main__":
    main()
