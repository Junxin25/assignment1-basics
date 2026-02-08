from __future__ import annotations

from typing import Iterable, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.id_to_bytes = vocab

        self._bpe_cache: dict[bytes, list[bytes]] = {}
        self.bytes_to_id = {v: k for k, v in vocab.items()} 
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_str_to_id = {}
        
        for s in self.special_tokens:
            s_bytes = s.encode("utf-8")
            if s_bytes in self.bytes_to_id:
                self.special_str_to_id[s] = self.bytes_to_id[s_bytes]
            else:
                raise ValueError(f"Special token {s} not found in vocabulary.")

    def from_files(vocab_path: str, merges_path: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        vocab: dict[int, bytes] = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                token_str, token_bytes_hex = line.strip().split("\t")
                vocab[int(token_str)] = token_bytes_hex.encode("utf-8").decode("unicode_escape").encode("latin1")
        
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                a_hex, b_hex = line.strip().split("\t")
                a = a_hex.encode("utf-8").decode("unicode_escape").encode("latin1")
                b = b_hex.encode("utf-8").decode("unicode_escape").encode("latin1")
                merges.append((a, b))        
        return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def _encode_piece_iter(self, text: str) -> Iterator[int]:
        for is_special, chunk in self._split_on_special(text):
            if is_special:
                tok_bytes = chunk.encode("utf-8")
                yield self.bytes_to_id[tok_bytes]
                continue
            for m in PAT_RE.finditer(chunk):
                s = m.group(0)
                if not s:
                    continue
                bs = s.encode("utf-8")
                for bpe_tok in self._bpe(bs):
                    yield self.bytes_to_id[bpe_tok]

    #todo: check the definition of merge_rank
    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached

        if not token_bytes:
            self._bpe_cache[token_bytes] = []
            return []
        word = tuple(token_bytes[i : i + 1] for i in range(len(token_bytes)))
        if len(word) == 1:
            self._bpe_cache[token_bytes] = [word[0]]
            return [word[0]]

        pairs = self._get_pairs(word)
        while True:
            best_pair = None
            best_rank = None
            for pair in pairs:
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                    break

            word = self._merge_pair(word, best_pair[0], best_pair[1])   
            if len(word) == 1:
                break

            pairs = self._get_pairs(word) 

        result = list(word)
        self._bpe_cache[token_bytes] = result
        return result         

    def _merge_pair(self, word: tuple[bytes, ...], a: bytes, b: bytes) -> tuple[bytes, ...]:
        out: list[bytes] = []
        i = 0
        n = len(word)
        ab = a + b
        while i < n:
            if i + 1 < n and word[i] == a and word[i + 1] == b:
                out.append(ab)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)
    
    def _get_pairs(self, word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def encode(self, text: str) -> list[int]:
        return list(self._encode_piece_iter(text))
    
    def _split_on_special(self, text: str) -> list[tuple[bool, str]]:
        if not self.special_tokens:
            return [(False, text)]

        # 长度优先，保证重叠 special token 取最长
        tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(t) for t in tokens)

        out: list[tuple[bool, str]] = []
        last = 0
        for m in re.finditer(pattern, text):
            if m.start() > last:
                out.append((False, text[last : m.start()]))
            out.append((True, m.group(0)))
            last = m.end()
        if last < len(text):
            out.append((False, text[last:]))
        return out

    def encode_iter(self, iterable: Iterable[str]) -> Iterator[int]:
        return self.encode_iterable(iterable)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._encode_piece_iter(text)

    def decode(self, token_ids: list[int]) -> str:
        bs = b"".join(self.id_to_bytes[token_id] for token_id in token_ids)
        return bs.decode("utf-8", errors="replace")