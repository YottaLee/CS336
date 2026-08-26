import os
from collections import defaultdict
from typing import Iterable, Iterator

import regex

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_token_sequence(
    token_seq: tuple[bytes, ...],
    best_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> tuple[bytes, ...]:
    """Replace each non-overlapping occurrence of ``best_pair``."""
    new_seq = []
    i = 0
    while i < len(token_seq):
        # Consume both symbols when the selected pair is found.
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == best_pair:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(token_seq[i])
            i += 1

    return tuple(new_seq)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens = special_tokens or []

    # Validate the target size before assigning any new token IDs.
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(" 'vocab_size' must be a positive integer")

    # Begin with one token for every possible byte value; learned merges are
    # appended after these 256 base tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    cur_id = 256

    # token_frequency_table: Dict[Tuple[bytes], int] = {}
    token_frequency_table = defaultdict(int)
    existing_byte = set(vocab.values())

    # Reserve special tokens as atomic vocabulary entries. They are excluded
    # from ordinary BPE pre-tokenization below.
    for st in special_tokens:
        if len(vocab) >= vocab_size:
            break
        st_bytes = st.encode("utf-8")
        if st_bytes not in existing_byte:
            vocab[cur_id] = st_bytes
            existing_byte.add(st_bytes)
            cur_id += 1

    # Read the corpus used to estimate byte-pair frequencies.
    try:
        with open(input_path, 'r', encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    # Split around special tokens, then split ordinary text with the GPT-2
    # pattern. Each pre-token is represented as a sequence of UTF-8 bytes.
    chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)

    for chunk in chunks:
        for word in regex.findall(GPT2_PAT, chunk):
            word_bytes = word.encode("utf-8")
            bytes_list = [bytes([x]) for x in word_bytes]
            token_frequency_table[tuple(bytes_list)] += 1

    merges: list[tuple[bytes, bytes]] = []
    pair_counts = defaultdict(int)
    for token in token_frequency_table.keys():
        for i in range(len(token)-1):
            pair_counts[token[i], token[i+1]] += token_frequency_table[token]



    # Repeatedly merge the most frequent adjacent pair until the vocabulary
    # reaches the requested size.
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # Select the highest-frequency pair; max(candidate_pair) makes ties
        # deterministic, matching the reference implementation.
        top_count = max(pair_counts.values())
        candidate_pair = [k for k, v in pair_counts.items() if v == top_count]
        best_pair = max(candidate_pair)

        merges.append(best_pair)

        best_pair_token = best_pair[0] + best_pair[1]
        vocab[cur_id] = best_pair_token
        cur_id += 1

        to_update_tokens = []
        for token, freq in token_frequency_table.items():
            has_pair = any(token[i : i + 2] == best_pair for i in range(len(token) - 1))
            if has_pair:
                to_update_tokens.append((token, freq))


        for token, freq in to_update_tokens:
            # Remove old pair counts for this word before adding counts for
            # its newly merged representation.
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            new_token_seq = merge_token_sequence(token, best_pair, best_pair_token)

            for i in range(len(new_token_seq) - 1):
                pair = (new_token_seq[i], new_token_seq[i+1])
                pair_counts[pair] += freq

            del token_frequency_table[token]
            token_frequency_table[new_token_seq] += freq


    return vocab, merges




class Tokenizer:
    """Byte-level BPE tokenizer using a fixed vocabulary and merge order."""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merge_order = {token_pair: i for i, token_pair in enumerate(self.merges)}
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs, preserving configured special tokens."""
        if not text:
            return []

        st = sorted(self.special_tokens, key=len, reverse=True)
        st_pattern = '|'.join(map(regex.escape, st))

        if self.special_tokens:
            chunks = regex.split(f'({st_pattern})', text)
        else:
            chunks = [text]

        result_ids = []
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                result_ids.append(self.bytes_to_id[chunk.encode('utf-8')])
            else:
                for word in regex.findall(GPT2_PAT, chunk):
                    merged_pieces = self._bpe_merge(word.encode("utf-8"))
                    for piece in merged_pieces:
                        result_ids.append(self.bytes_to_id[piece])
        return result_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode a sequence of strings."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back to UTF-8 text."""
        all_bytes = b"".join(self.vocab[token_id] for token_id in tokens)
        return all_bytes.decode("utf-8", errors="replace")

    def _bpe_merge(self, piece: bytes) -> list[bytes]:
        """Apply learned merges in priority order to one pre-token."""

        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            pairs = set()
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                if pair in self.merge_order:
                    pairs.add(pair)

            if not pairs:
                break

            best_pair = min(pairs, key=lambda pair: self.merge_order[pair])
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts
