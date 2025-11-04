import os
import regex as re
import cProfile
import multiprocessing
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 # Read ahead by 4k bytes at a time
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position) # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size) # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            
            if len(mini_chunk) < mini_chunk_size:
                chunk_boundaries[bi] = file_size
                break
            
            initial_position += mini_chunk_size

    final_boundaries = sorted(set(chunk_boundaries))

    if not final_boundaries or final_boundaries[0] != 0:
        final_boundaries.insert(0, 0)
    if final_boundaries[-1] != file_size:
        final_boundaries.append(file_size)
    
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(final_boundaries))

def process_chunk(
    chunk_bytes: bytes, 
    special_tokens_pattern: re.Pattern, 
    pre_tok_pattern: re.Pattern
) -> Counter:
    """
    Processes a raw byte chunk.
    1. Splits it by special tokens.
    2. Applies regex pre-tokenization to each part.
    3. Counts the frequency of each pre-token "word".
    """
    word_counts = Counter()
    
    text_parts = special_tokens_pattern.split(chunk_bytes)
    
    for part in text_parts:
        if not part:
            continue
            
        text_part = part.decode('utf-8', errors='ignore')
        
        for match in pre_tok_pattern.finditer(text_part):
            word_bytes = match.group(0).encode('utf-8')
            word_tuple = tuple(word_bytes[i:i+1] for i in range(len(word_bytes)))
            if word_tuple:
                word_counts[word_tuple] += 1
                
    return word_counts

def get_pair_counts(word_counts: Dict[Tuple[bytes, ...], int]) -> Counter:
    """
    Iterates over the word-frequency map to count all adjacent pairs.
    """
    pair_counts = Counter()
    for word_tuple, freq in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_counts[pair] += freq
    return pair_counts

def merge_word_tuple(
    word_tuple: Tuple[bytes, ...], 
    pair: Tuple[bytes, bytes], 
    new_token: bytes
) -> Tuple[bytes, ...]:
    """
    Replaces all occurrences of `pair` in `word_tuple` with new_token.
    """
    new_word = []
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == pair:
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word_tuple[i])
            i += 1
    return tuple(new_word)

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: List[str],
    num_processes: int = os.cpu_count() or 4
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a byte-level BPE tokenizer.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not special_tokens:
        raise ValueError("special_tokens list cannot be empty.")

    vocab: Dict[int, bytes] = {}
    current_id = 0
    special_token_bytes_list = []

    for token_str in special_tokens:
        token_bytes = token_str.encode('utf-8')
        vocab[current_id] = token_bytes
        special_token_bytes_list.append(token_bytes)
        current_id += 1
        
    for i in range(256):
        vocab[current_id] = bytes([i])
        current_id += 1
        
    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        print(f"Vocab_size ({vocab_size}) is too small. No merges performed.")
        num_merges = 0

    special_pattern = re.compile(
        b"|".join(re.escape(b) for b in special_token_bytes_list)
    )
    pre_tok_pattern = re.compile(PAT)

    file_chunks = []
    try:
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, special_token_bytes_list[0])
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                file_chunks.append(f.read(end - start))
    except Exception as e:
        raise IOError(f"Error reading or chunking input file {input_path}: {e}")

    from functools import partial
    worker_func = partial(
        process_chunk, 
        special_tokens_pattern=special_pattern, 
        pre_tok_pattern=pre_tok_pattern
    )
    
    current_words = Counter()
    with multiprocessing.Pool(num_processes) as pool:
        all_counts_lists = pool.map(worker_func, file_chunks)
        for count in all_counts_lists:
            current_words.update(count)
    print(f"--- Pre-tokenization complete. ---")

    merges: List[Tuple[bytes, bytes]] = []
    for i in range(num_merges):
        pair_counts = get_pair_counts(current_words)
        
        if not pair_counts:
            print(f"No more pairs to merge.")
            break

        max_freq = max(pair_counts.values())
        best_pairs = [p for p, f in pair_counts.items() if f == max_freq]
        best_pair = max(best_pairs) 
        
        new_token_bytes = best_pair[0] + best_pair[1]
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token_bytes
        merges.append(best_pair)
        
        new_current_words = Counter()
        for word_tuple, freq in current_words.items():
            new_word = merge_word_tuple(word_tuple, best_pair, new_token_bytes)
            new_current_words[new_word] += freq
        current_words = new_current_words

    return vocab, merges

def main():
    input_file = "/Users/yichenxu/Library/CloudStorage/OneDrive-个人/Documents/NLPDL-2025Fall/hw1_bpe_and_lm/data/TinyStoriesV2-GPT4-valid.txt"
    separator = "<|endoftext|>"
    target_vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    
    trained_vocab, trained_merges = train_bpe(
        input_file, 
        target_vocab_size, 
        special_tokens,
        num_processes=8
    )

    print(f"\n--- Training Complete ---")
    print(f"Final Vocab Size: {len(trained_vocab)}")
    
    save_path = "/Users/yichenxu/Library/CloudStorage/OneDrive-个人/Documents/NLPDL-2025Fall/hw1_bpe_and_lm/data/test.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(trained_vocab))
        f.write("\n MERGES \n")
        f.write(str(trained_merges))


if __name__ == "__main__":
    cProfile.run('main()', sort='tottime')
    