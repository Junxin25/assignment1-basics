from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _split_on_special(text: str, special_tokens: List[str]) -> List[str]:
    """Splits text on special tokens.

    Args:
        text: The input text to split.
        special_tokens: A list of special tokens to split on.

    Returns:
        A list of strings, split on the special tokens.
    """
    if not special_tokens:
        return [text]

    delim = "|".join(re.escape(t) for t in special_tokens)
    parts = re.split(delim, text)
    return [p for p in parts if p]  # Remove empty strings

def _pretok_counts_serial(text: str, special_tokens: List[str]) -> Counter:

    counts = Counter()

    for chunk in _split_on_special(text, special_tokens):
        for m in re.finditer(PAT, chunk):
            s = m.group(0)
            bs = s.encode("utf-8")
            key = tuple(bytes([b]) for b in bs)
            counts[key] += 1
    
    return counts

def _chunk_ranges_on_special(text: str, special_token: str, approx_chunk_size: int) -> List[Tuple[int, int]]:
    """Chunks text into ranges based on special token positions.
    
    Args:
        text: The input text to chunk.
        special_token: The special token to use as chunk boundaries.
        approx_chunk_size: The approximate size of each chunk."""


    if not special_token or special_token not in text:
        return [(0, len(text))]
    
    starts = [m.start() for m in re.finditer(re.escape(special_token), text)]
    starts = [0] + starts + [len(text)]
    starts = sorted(set(starts))

    ranges: List[Tuple[int, int]] = []
    i = 0
    while i < len(starts) - 1:
        s = starts[i]
        target = s + approx_chunk_size

        j = i + 1
        while j < len(starts) and starts[j] < target:
            j += 1
        e = starts[j]
        ranges.append((s, e))
        i = j
    return [(s, e) for (s, e) in ranges if s < e]

def _init_vocab(special_tokens: List[str]) -> Dict[bytes, int]:
    """Initializes the vocabulary with special tokens.

    Args:
        special_tokens: A list of special tokens to include in the vocabulary.

    Returns:
        A dictionary mapping bytes to their corresponding indices.
    """
    vocab: Dict[bytes, int] = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")
    start = len(special_tokens)
    for b in range(256):
        vocab[start + b] = bytes([b])
    return vocab

def _count_pair_freqs(word_counts: Counter) -> Counter:
    pair_freqs: Counter = Counter()
    
    for token_seq, c in word_counts.items():
        if len(token_seq) < 2:
            continue
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_freqs[pair] += c
    
    return pair_freqs

def _select_best_pair(pair_freqs: Counter) -> Pair | None:
    if not pair_freqs:
        return None
    best_pair, _ = max(pair_freqs.items(), key=lambda kv:(kv[1], kv[0]))
    return best_pair

def _merge_pair_in_seq(seq: Tuble[bytes, ...], a: bytes, b: bytes) -> Tuple[bytes, ...]:
    out: List[bytes] = []
    i = 0
    n = len(seq)
    ab = a + b
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)

def _apply_merge(word_counts: Counter, a: bytes, b: bytes) -> Counter:
    new_word_counts: Counter = Counter()
    for token_seq, c in word_counts.items():
        merged = _merge_pair_in_seq(token_seq, a, b)
        new_word_counts[merged] += c
    return new_word_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):

    #1） read input text 
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    #2) pre-tokenize and count
    word_counts = _pretok_counts_serial(text, special_tokens)

    #3) initialize vocab
    vocab = _init_vocab(special_tokens)
    merges: List[Tuple[bytes, bytes]] = []

    #4）merge loop
    while len(vocab) < vocab_size:
        pair_freq = _count_pair_freqs(word_counts)
        best_pair = _select_best_pair(pair_freq)
        if best_pair is None:
            break
        a, b = best_pair
        merges.append(best_pair)
        new_token = a + b
        vocab[len(vocab)] = new_token
        
        word_counts = _apply_merge(word_counts, a, b)

    return vocab, merges