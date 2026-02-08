from __future__ import annotations
import time
from pathlib import Path

from cs336_basics.bpe_train import train_bpe

SPECIALS = ["<|endoftext|>"]
VOCAB_SIZE = 32000


def get_peak_rss_mb() -> float:
    """Linux 下最简单的 peak memory 读法（ru_maxrss）。"""
    import resource
    # Linux: KB；macOS: bytes（但你在 linux server 上，所以按 KB 算）
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def save_vocab(vocab: dict[bytes, int], filepath: Path) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for i in sorted(vocab.keys()):
            f.write(f"{i}\t{vocab[i].hex()}\n")

def save_merges(merges: list[tuple[bytes, bytes]], filepath: Path) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()}\t{b.hex()}\n")

def longest_token(vocab: dict[bytes, int]) -> tuple[int, bytes]:
    max_len = -1
    max_tok = b""
    for tok in vocab.keys():
        if len(tok) > max_len:
            max_len = len(tok)
            max_tok = tok
    return max_len, max_tok


if __name__ == "__main__":
    input_path = Path("/data/home/junxinfu/deep_learning/assignment1-basics/data/owt_train.txt")
    out_dir = Path("./bpe_models/owt_train")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mem0 = get_peak_rss_mb()

    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIALS,
    )

    t1 = time.perf_counter()
    mem1 = get_peak_rss_mb()
    elapsed = t1 - t0
    mem_used = mem1 - mem0

    #save the vocab and merges
    vocab_path = out_dir / f"bpe_vocab_{VOCAB_SIZE}.tsv"
    merges_path = out_dir / f"bpe_merges_{VOCAB_SIZE}.tsv"
    
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    tok_id, tok_bytes = longest_token(vocab)

    print(f"time = {elapsed/3600: .3f} hours, peak_mem = {mem_used/1024: .3f} GB")
    print(f"vocab size = {len(vocab)}, longest token id = {tok_id}, bytes = {tok_bytes}")
