# fineweb.py: download FineWeb-Edu and turn it into token shards for training.
#
# FineWeb-Edu is web text filtered for "educational" quality. The "sample-10BT"
# version has about 10 billion GPT-2 tokens.
# Each shard is a .npy file of uint16 token numbers (the GPT-2 vocabulary has
# 50,257 tokens, which fits in 16 bits, so each token takes 2 bytes on disk).
# Shard 0 is the validation split, all others are training.
#
# Full data (rented server, ~100 shards x 100M tokens = ~20 GB):
#     python fineweb.py
# Small test (Colab): stream only the first few documents, 3 shards of 10M tokens:
#     python fineweb.py --streaming --shard_size 10000000 --max_shards 3 --local_dir edu_fineweb_test

import argparse, multiprocessing as mp, os, sys, time
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir', default='edu_fineweb10B')           # where the shards go
parser.add_argument('--remote_name', default='sample-10BT')            # which FineWeb-Edu sample
parser.add_argument('--shard_size', type=int, default=int(1e8))       # tokens per shard (100M, like the video)
parser.add_argument('--max_shards', type=int, default=0)              # 0 = all; >0 = stop after this many (tests)
parser.add_argument('--streaming', action='store_true')               # read the data while downloading (no full download first)
parser.add_argument('--nprocs', type=int, default=max(1, os.cpu_count() // 2))  # CPU processes for tokenizing
args = parser.parse_args()

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # token 50256: marks where one document ends and the next begins

def tokenize(doc):
    # one document -> uint16 array of tokens, starting with the end-of-text token
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token numbers too large for uint16"
    return tokens_np.astype(np.uint16)

def write_shard(shard_index, tokens_np):
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    np.save(filename, tokens_np)  # np.save adds ".npy"

if __name__ == '__main__':
    t_start = time.time()
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=args.remote_name, split="train", streaming=args.streaming)
    shard_index = 0
    total_tokens = 0
    shard_buffer = np.empty((args.shard_size,), dtype=np.uint16)  # tokens of the shard being filled
    token_count = 0
    progress_bar = None
    done = False
    with mp.Pool(args.nprocs) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < args.shard_size:
                # the whole document fits in the current shard
                shard_buffer[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # fill the rest of this shard, write it, and start the next one with the leftover
                remainder = args.shard_size - token_count
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(remainder)
                progress_bar.close()
                shard_buffer[token_count:token_count + remainder] = tokens[:remainder]
                write_shard(shard_index, shard_buffer)
                total_tokens += args.shard_size
                shard_index += 1
                progress_bar = None
                if args.max_shards and shard_index >= args.max_shards:
                    done = True
                    break
                shard_buffer[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        # the last, partly filled shard
        if not done and token_count != 0:
            if progress_bar is not None:
                progress_bar.close()
            write_shard(shard_index, shard_buffer[:token_count])
            total_tokens += token_count
            shard_index += 1
    elapsed = time.time() - t_start
    print(f"wrote {shard_index} shards, {total_tokens:,} tokens in {elapsed:.1f}s "
          f"({total_tokens / elapsed:,.0f} tokens/sec, {args.nprocs} processes) -> {DATA_CACHE_DIR}")
    # When we stop early (--max_shards) in streaming mode, background download threads can
    # keep the process alive forever, so exit right away once everything is written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
