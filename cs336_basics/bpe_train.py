from __future__ import annotations
from typing import Dict, List, Tuple
import regex as re
from collections import Counter

Pair = Tuple[bytes, bytes]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)
BYTE_TABLE = [bytes([b]) for b in range(256)]

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
        for m in PAT_RE.finditer(chunk):
            s = m.group(0)
            bs = s.encode("utf-8")
            key = tuple(BYTE_TABLE[b] for b in bs)
            counts[key] += 1
    
    return counts

def _init_vocab(special_tokens: List[str]) -> Dict[int, bytes]:
    """Initializes the vocabulary with special tokens.

    Args:
        special_tokens: A list of special tokens to include in the vocabulary.

    Returns:
        A dictionary mapping bytes to their corresponding indices.
    """
    vocab: Dict[int, bytes] = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")
    start = len(special_tokens)
    for b in range(256):
        vocab[start + b] = BYTE_TABLE[b]
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

def _merge_pair_in_seq(seq: Tuple[bytes, ...], a: bytes, b: bytes) -> Tuple[bytes, ...]:
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


def _pair_counts_in_seq(seq: Tuple[bytes, ...]) -> Counter:
    pair_counts: Counter = Counter()
    if len(seq) < 2:
        return pair_counts
    for i in range(len(seq) - 1):
        pair_counts[(seq[i], seq[i + 1])] += 1
    return pair_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
    """Trains a BPE tokenizer on the input text.
        Args:
            input_path: Path to the input text file.
            vocab_size: Desired vocabulary size.
            special_tokens: List of special tokens to include in the vocabulary.
        Returns:
            A tuple of (vocab, merges), where vocab is a dict mapping token IDs to bytes,
            and merges is a list of merged byte pairs.
        for example:
            vocab, merges = train_bpe("data.txt", 10000, ["<|endoftext|>"])
            vocab would be like {0: b'<|endoftext|>', 1: b'a', 2: b'b', ..., 9999: b'example'}
            merges would be like [(b'a', b'b'), (b'ab', b'c'), ...]
    """
    #1） read input text 
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    #2) pre-tokenize and count
    word_counts = _pretok_counts_serial(text, special_tokens)

    # Build a stable seq_id space so merges can update sequences in-place.
    seq_by_id: Dict[int, Tuple[bytes, ...]] = {}
    count_by_id: Dict[int, int] = {}
    pair_freqs: Counter = Counter()
    pair_to_seqs: Dict[Pair, set[int]] = {}
    for seq_id, (seq, c) in enumerate(word_counts.items()):
        seq_by_id[seq_id] = seq
        count_by_id[seq_id] = c

        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            pair: Pair = (seq[i], seq[i + 1])
            pair_freqs[pair] += c
            pair_to_seqs.setdefault(pair, set()).add(seq_id)

    #3) initialize vocab
    vocab = _init_vocab(special_tokens)
    merges: List[Tuple[bytes, bytes]] = []

    #4) merge loop (incremental updates via inverted index)
    while len(vocab) < vocab_size:
        best_pair = _select_best_pair(pair_freqs)
        if best_pair is None:
            break
        a, b = best_pair

        affected_ids = pair_to_seqs.get(best_pair)
        if not affected_ids:
            break

        merges.append(best_pair)
        new_token = a + b
        vocab[len(vocab)] = new_token

        # Only sequences containing best_pair can change.
        for seq_id in list(affected_ids):
            c = count_by_id[seq_id]
            old_seq = seq_by_id[seq_id]

            # Remove old contributions.
            old_pair_counts = _pair_counts_in_seq(old_seq)
            for pair, occ in old_pair_counts.items():
                pair_freqs[pair] -= occ * c
                if pair_freqs[pair] <= 0:
                    del pair_freqs[pair]
                seqs = pair_to_seqs.get(pair)
                if seqs is not None:
                    seqs.discard(seq_id)
                    if not seqs:
                        del pair_to_seqs[pair]

            # Merge sequence and add new contributions.
            new_seq = _merge_pair_in_seq(old_seq, a, b)
            seq_by_id[seq_id] = new_seq
            new_pair_counts = _pair_counts_in_seq(new_seq)
            for pair, occ in new_pair_counts.items():
                pair_freqs[pair] += occ * c
                pair_to_seqs.setdefault(pair, set()).add(seq_id)
    return vocab, merges