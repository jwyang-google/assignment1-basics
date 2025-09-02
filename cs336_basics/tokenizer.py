import os
# from pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import regex as re
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
from typing import BinaryIO
import time
import pickle
from tqdm import tqdm

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

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in tqdm(range(1, len(chunk_boundaries) - 1)):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_worker(fname, start, end, special_tokens):
  with open(fname, "rb") as f:
    f.seek(start)
    all_chunk = f.read(end - start).decode("utf-8", errors="ignore")
    if len(special_tokens) > 0:
      chunks = re.split("|".join([re.escape(special_token) for special_token in special_tokens]), all_chunk)
    else:
      chunks = [all_chunk]

    res = {}
    for chunk in tqdm(chunks):
      # print("chunk: {}".format(chunk))
      PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
      token_iters = re.finditer(PAT, chunk)
      
      for token in token_iters:
        # print("token: {}".format(token))
        token_tuple = tuple([token.to_bytes() for token in list(token.group().encode("utf-8"))])
        res[token_tuple] = res.setdefault(token_tuple, 0) + 1 
    return res


class BPE_Tokenizer():
  def __init__(
      self, 
      train_from_scratch: bool = True,
      corpus: str | os.PathLike = None,
      vocab_filepath: str | os.PathLike = None,
      merges_filepath: str | os.PathLike = None,
      vocab_size: int = 10000, 
      special_tokens: list[str] = None,
      vocab: dict[int, bytes] = None,
      merges: list[tuple[bytes, bytes]] = None,

  ):
    self._special_tokens = special_tokens
    if self._special_tokens is None:
      self._special_tokens = []
    
    if train_from_scratch:
      assert corpus is not None
      self._max_vocab_size = vocab_size
      self._vocab, self._merges = self.train_bpe(corpus=corpus)
    else:
      assert vocab is not None
      assert merges is not None
      self._vocab = vocab
      self._merges = merges

      # build a merge index
      self._merge_idx = {}
      for i, merge in enumerate(self._merges):
        self._merge_idx[merge] = i
      del self._merges

      self._inverse_vocab = {}
      for token_id in self._vocab:
        self._inverse_vocab[self._vocab[token_id]] = token_id

    if vocab_filepath:
      with open(vocab_filepath, "wb") as f:
        pickle.dump(self._vocab, f)
    
    if merges_filepath:
      with open(merges_filepath, "wb") as f:
        pickle.dump(self._merges, f)

  @classmethod  
  def from_files(cls, vocab_filepath, merges_file_path, special_tokens = None):
    with open(vocab_filepath, "rb") as f:
      vocab = pickle.load(f)
      print(vocab)
    with open(merges_file_path, "rb") as f:
      merges = pickle.load(f)
    assert vocab is not None
    return cls(train_from_scratch=False, vocab=vocab, merges=merges, special_tokens=special_tokens)
  
  def encode(self, text: str) -> list[int]:
    def encode_text_without_special_token(text):
      PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
      token_iters = re.finditer(PAT, text)
      
      pretoken_token_ids = {}
      res_token_ids = []
      for pretoken in token_iters:
        tokens = [pretoken.to_bytes() for pretoken in list(pretoken.group().encode("utf-8"))]
        tokens_tuple = tuple(tokens)
        if tokens_tuple in pretoken_token_ids:
          res_token_ids.extend(pretoken_token_ids[tokens_tuple])
          continue

        while len(tokens) > 1:
          # iterate current token_ids list and get pairs that can be merged
          cur_merges = []
          for i in range(len(tokens) - 1):
            token_pair = (tokens[i], tokens[i+1])
            if token_pair in self._merge_idx:
              cur_merges.append((self._merge_idx[token_pair], i))
          if len(cur_merges) == 0:
            break

          cur_merges = sorted(cur_merges, key=lambda x: x[0])
          idx_to_merge = cur_merges[0][1]
          new_tokens = tokens[:idx_to_merge] + [tokens[idx_to_merge] + tokens[idx_to_merge+1]] + tokens[idx_to_merge+2:]
          # print("tokens: {}".format(tokens))
          # print("merge_ids: {}".format(idx_to_merge))
          # print("new_tokens: {}".format(new_tokens))
          # assert False
          assert len(new_tokens) == len(tokens) - 1
          tokens = new_tokens


        token_ids = []
        for token in tokens:
          token_ids.append(self._inverse_vocab[token])      
        pretoken_token_ids[tokens_tuple] = token_ids 
        
        res_token_ids.extend(token_ids)
      return res_token_ids
    
    sorted_special_tokens = sorted(self._special_tokens, key=len, reverse=True)
    if not sorted_special_tokens:
      return encode_text_without_special_token(text)

    special_pattern = f"({'|'.join(re.escape(s) for s in sorted_special_tokens)})"
    text_parts = re.split(special_pattern, text)

    res_token_ids = []
    for part in text_parts:
      if part in self._special_tokens:
        res_token_ids.append(self._inverse_vocab[part.encode("utf-8")])
      elif part:
        res_token_ids.extend(encode_text_without_special_token(part))

    
    return res_token_ids

  def decode(self, ids: list[int]) -> str:
    tokens = []
    for id in ids:
      tokens.append(self._vocab[id])
    res = b"".join([token for token in tokens])
    res = res.decode("utf-8", errors="replace")
    return res
    
  from collections.abc import Iterable, Iterator 
  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    total_str = "".join(string for string in iterable)
    for token_id in self.encode(total_str):
      yield token_id

  def train_bpe(
      self, 
      corpus: str | os.PathLike,
  ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocab
    print("Initialize vocab with vocab_size {}".format(256))
    vocab = {}
    for i in range(256):
      vocab[i] = i.to_bytes()

    print("Adding special tokens into vocab")
    next_id = 256
    for token in self._special_tokens:
      vocab[next_id] = token.encode('utf-8')
      next_id += 1

    print("Pretokenize the corpus")
    start_time = time.time()
    # pretokenize to get map pretoken -> counts
    with open(corpus, "rb") as f:
      boundaries = find_chunk_boundaries(f, 24, b"<|endoftext|>")

      # use mp to process each chunk in parallel
      arguments = []
      for i in range(len(boundaries)-1):
        arguments.append((corpus, boundaries[i], boundaries[i+1], self._special_tokens))
      
      with mp.Pool(processes=len(boundaries)) as pool:
        results = pool.starmap(pretokenize_worker, arguments)

      # aggregate all results
      result_pretoken_dict = {}
      for result in results:
        for token_tuple in result:
          result_pretoken_dict[token_tuple] = result_pretoken_dict.setdefault(token_tuple, 0) + result[token_tuple]
    
    end_time = time.time()
    print("Pretokenize takes {} seconds".format(end_time - start_time))
    # print("Result Pretoken Dict: {}".format(result_pretoken_dict))
    
    # iterate through the counts map
    # generate the counts pair map
    token_pair_dict = {}
    pretoken_pairs = {}
    token_pretoken = {}
    for token_tuples in result_pretoken_dict:
      token_pairs = []
      for i in range(len(token_tuples)):
        token_pretoken.setdefault(token_tuples[i], set()).add(token_tuples)
        if i == len(token_tuples) - 1:
          break
        token_pair = (token_tuples[i], token_tuples[i+1])
        token_pairs.append(token_pair)
        token_pair_dict[token_pair] = token_pair_dict.setdefault(token_pair, 0) + result_pretoken_dict[token_tuples]
      pretoken_pairs[token_tuples] = token_pairs
    # print("token_pair_dict: {}".format(token_pair_dict))
    # print("pretoken_pairs: {}".format(pretoken_pairs))
    # print("token_pretoken: {}".format(token_pretoken))

    # while loop
    # merge the pair with max count (increment token ID and update vocab)
    # update the counts pair map
    num_merges = 0
    self._merges = []
    while True:
      if len(vocab) >= self._max_vocab_size:
        break
      
      all_token_pairs = sorted(token_pair_dict.items(), key=lambda item: (item[1], item[0]), reverse=True)
      max_token_pair = all_token_pairs[0]
      max_token_count = max_token_pair[1]

      token1, token2 = max_token_pair[0]
      merged_token = token1 + token2
      merged_token_id = next_id
      next_id += 1
      vocab[merged_token_id] = merged_token
      self._merges.append((token1, token2))
      # print("Merge {} and {} with count {}".format(token1, token2, max_token_count))
      print("Generate new token {} with token_id {}".format(merged_token, merged_token_id))
      num_merges += 1

      # remove all overlapping pairs
      for token_pair in all_token_pairs:
        if token1 in token_pair[0] or token2 in token_pair[0]:
          del token_pair_dict[token_pair[0]]

      pretokens_need_update = set()
      if token1 in token_pretoken:
        pretokens_need_update.update(token_pretoken[token1])
        del token_pretoken[token1]
      if token2 in token_pretoken:
        pretokens_need_update.update(token_pretoken[token2])
        del token_pretoken[token2]
      # print("Pre tokens needs update: {}".format(pretokens_need_update))

      # update pretoken(tuple of original tokens) -> list of token_pairs       
      for pretoken in pretokens_need_update:
        list_token_pairs = pretoken_pairs[pretoken]
        idx = 0
        while idx < len(list_token_pairs):
          if list_token_pairs[idx] == (token1, token2):
            if idx - 1 >= 0:
              list_token_pairs[idx - 1] = (list_token_pairs[idx-1][0], merged_token) 
            if idx + 1 < len(list_token_pairs):
              list_token_pairs[idx + 1] = (merged_token, list_token_pairs[idx+1][1])
            del list_token_pairs[idx]
          idx += 1
        pretoken_pairs[pretoken] = list_token_pairs

      # update token_pair_dict, token_token_pairs
      for pretoken in pretokens_need_update:
        list_token_pairs = pretoken_pairs[pretoken]
        for token_pair in list_token_pairs:
          if token1 not in token_pair and token2 not in token_pair and merged_token not in token_pair:
            continue
          else:
            t1, t2 = token_pair
            token_pair_dict[token_pair] = token_pair_dict.setdefault(token_pair, 0) + result_pretoken_dict[pretoken]
            token_pretoken.setdefault(t1, set()).add(pretoken)
            token_pretoken.setdefault(t2, set()).add(pretoken)
            
    # return the vocab and the merges
    return vocab, self._merges

  def get_vocab(self) -> dict[int, bytes]:
    return self._vocab
  
  def get_merges(self) -> list[tuple[bytes, bytes]]:
    return self._merges
  

  

# TinyStories BPE Tokenizer Training (~20 mins for not optimized version)
# ts_corpus = "../data/TinyStoriesV2-GPT4-train.txt"
# ts_vocab_filepath, ts_merges_filepath = "../data/TinyStoriesV2-GPT4-train.vocab", "../data/TinyStoriesV2-GPT4-train.merges"
# vocab_size = 10000

# ts_tokenizer = BPE_Tokenizer(
#   corpus=ts_corpus, 
#   vocab_size=vocab_size, 
#   special_tokens=["<|endoftext|>"],
#   vocab_filepath=ts_vocab_filepath,
#   merges_filepath=ts_merges_filepath
# )

# # OpenText BPE Tokenizer Training
# owt_corpus = "../data/owt_train.txt"
# owt_vocab_filepath, owt_merges_filepath = "../data/owt_train.vocab", "../data/owt_train.merges"
# vocab_size = 32000

# ts_tokenizer = BPE_Tokenizer(
#   corpus=owt_corpus, 
#   vocab_size=vocab_size, 
#   special_tokens=["<|endoftext|>"],
#   vocab_filepath=owt_vocab_filepath,
#   merges_filepath=owt_merges_filepath
# )

# # For debugging
# ts_corpus = "test.txt"
# ts_vocab_filepath, ts_merges_filepath = "test.vocab", "test.merges"
# vocab_size = 262
# ts_tokenizer = BPE_Tokenizer(
#   corpus=ts_corpus, 
#   vocab_size=vocab_size, 
#   special_tokens=["<|endoftext|>"],
#   vocab_filepath=ts_vocab_filepath,
#   merges_filepath=ts_merges_filepath
# )
# ids = tokenizer.encode("lower lower newest newest")
# decode_str = tokenizer.decode(ids)
# print(decode_str)