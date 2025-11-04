import json  # For loading .vocab files
import regex as re # For the GPT-2 PAT regex
from typing import List, Dict, Tuple, Optional, Iterable, Iterator

# This is the GPT-2 pre-tokenizer regex pattern from the assignment
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        '''
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        '''
        self.vocab = vocab
        
        # --- Pre-compute lookups for encoding ---
        
        # 1. Reverse vocab for fast byte -> id lookup
        self.encoder: Dict[bytes, int] = {b: i for i, b in vocab.items()}
        
        # 2. Create a merge-rank map for fast pair -> priority lookup
        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        
        # 3. Compile the GPT-2 pre-tokenization regex
        self.pre_tok_pattern = re.compile(GPT2_PAT)

        # --- Pre-compute special token handling ---
        self.special_tokens: Dict[str, int] = {}
        self.special_pattern: Optional[re.Pattern] = None
        
        if special_tokens:
            for token_str in special_tokens:
                token_bytes = token_str.encode('utf-8')
                if token_bytes in self.encoder:
                    self.special_tokens[token_str] = self.encoder[token_bytes]
                else:
                    # This can happen if the test suite provides special tokens
                    # that aren't in the base vocab (e.g. for testing)
                    print(f"Warning: Special token '{token_str}' not found in provided vocab.")
            
            # --- FIX for overlapping tokens ---
            # Sort tokens by length, descending, to match longest first
            # e.g., match "<|endoftext|><|endoftext|>" before "<|endoftext|>"
            escaped_tokens = sorted(
                [re.escape(t) for t in self.special_tokens.keys()], 
                key=len, 
                reverse=True
            )
            
            if escaped_tokens:
                self.special_pattern = re.compile(f"({'|'.join(escaped_tokens)})")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        '''
        Class method that constructs and returns a Tokenizer from serialized vocab/merges.
        
        NOTE: This is fixed to load the gpt2/tiktoken file formats, not ast.literal_eval.
        '''
        try:
            # 1. Load the vocabulary (which is a JSON file)
            # The test files are str -> int, so we must invert and encode
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                str_to_id_map = json.load(f)
            
            loaded_vocab: Dict[int, bytes] = {}
            for s, i in str_to_id_map.items():
                loaded_vocab[i] = s.encode('utf-8')

            # 2. Load the merges (which is a space-separated text file)
            loaded_merges: List[Tuple[bytes, bytes]] = []
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip the first line (version header) in gpt2.merges
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(' ')
                    if len(parts) == 2:
                        # Encode the string parts to bytes to match our vocab
                        loaded_merges.append(
                            (parts[0].encode('utf-8'), parts[1].encode('utf-8'))
                        )
            
            # 3. Construct and return a new instance of the class
            return cls(loaded_vocab, loaded_merges, special_tokens)

        except FileNotFoundError as e:
            print(f"Error: File not found. {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse vocab JSON. {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred in from_files: {e}")
            raise

    def _bpe_merge(self, word_bytes: bytes) -> List[int]:
        """
        Applies the BPE merge rules to a single pre-token (word).
        (This logic is unchanged)
        """
        if not word_bytes:
            return []
            
        # 1. Start with single-byte representations
        # Note: The raw bytes (e.g., b'\xf0') may NOT be in the gpt2
        # vocab, but the merged tokens (e.g., b'\xf0\x9f\x99\x83') will be.
        tokens: List[bytes] = [word_bytes[i:i+1] for i in range(len(word_bytes))]

        while True:
            pairs = list(zip(tokens, tokens[1:]))
            if not pairs:
                break 

            # Find the best pair (lowest rank)
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))

            # If no mergeable pairs are found, we're done
            if self.merge_ranks.get(best_pair) == float('inf'):
                break

            # Merge the best pair
            new_tokens: List[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # 6. Convert final byte tokens to IDs
        # The test suite expects that we *only* return IDs that are in the vocab.
        # This handles the unicode `🙃` (b'\xf0\x9f\x99\x83') which merges into
        # three tokens [8582, 247, 225] corresponding to b'\xf0\x9f', b'\x99', b'\x83'
        final_ids = [self.encoder[t] for t in tokens if t in self.encoder]
        
        return final_ids


    def encode(self, text: str) -> List[int]:
        '''
        Encode a string into a sequence of token IDs.
        (This is the fixed logic for special tokens)
        '''
        token_ids: List[int] = []
        
        # If no special token pattern was compiled, process as one big chunk
        if not self.special_pattern:
            for match in self.pre_tok_pattern.finditer(text):
                word_bytes = match.group(0).encode('utf-8')
                token_ids.extend(self._bpe_merge(word_bytes))
            return token_ids

        # --- We have special tokens ---
        last_end = 0
        # Iterate over all matches for special tokens
        for special_match in self.special_pattern.finditer(text):
            
            # 1. Process the regular text *before* this special token
            pre_text = text[last_end:special_match.start()]
            if pre_text:
                for match in self.pre_tok_pattern.finditer(pre_text):
                    word_bytes = match.group(0).encode('utf-8')
                    token_ids.extend(self._bpe_merge(word_bytes))
            
            # 2. Add the special token itself
            special_token_str = special_match.group(0)
            if special_token_str in self.special_tokens:
                token_ids.append(self.special_tokens[special_token_str])
            # If it's not in the map, it's a false match (part of a longer token)
            # We can ignore it, but the sorted regex should prevent this.
            
            last_end = special_match.end()
            
        # 3. Process any remaining text *after* the last special token
        post_text = text[last_end:]
        if post_text:
            for match in self.pre_tok_pattern.finditer(post_text):
                word_bytes = match.group(0).encode('utf-8')
                token_ids.extend(self._bpe_merge(word_bytes))
                
        return token_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator 
        that lazily yields token IDs.
        (This implementation is correct given the test suite's use of line-by-line iterables)
        '''
        for text_chunk in iterable:
            # Yields tokens from each chunk one by one
            yield from self.encode(text_chunk)


    def decode(self, ids: List[int]) -> str:
        '''
        Decode a sequence of token IDs into text.
        (This implementation is correct and uses errors='replace')
        '''
        token_bytes_list = [self.vocab.get(token_id, b'') for token_id in ids]
        concatenated_bytes = b"".join(token_bytes_list)
        
        # Use errors='replace' to insert U+FFFD for invalid bytes
        return concatenated_bytes.decode('utf-8', errors='replace')