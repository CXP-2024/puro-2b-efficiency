#!/usr/bin/env python
"""Tokenize Puro-2B Parquet shards and pack them into the NPY layout that
Puro-Megatron's ``CustomGPTDataset`` (pretrain_gpt.py) reads.

Layout produced in ``--out``::

    metadata.json            {"size": N, "file_list": [...], "file_len": [...]}
    shard-00000.npy          int32 array of shape [rows, seq_length + 1]
    ...

Every document is tokenized, followed by one EOS token, and the resulting
token stream is cut into rows of ``seq_length + 1`` tokens. The training code
uses row[:-1] as inputs and row[1:] as labels, and rebuilds document
boundaries from the EOS tokens at run time.

The last ``--valid-rows`` rows go to ``<out>/valid``, the rest to ``<out>/train``.
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Parquet files with a 'text' column")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--eos-token-id", type=int, default=151645)
    ap.add_argument("--rows-per-shard", type=int, default=32768)
    ap.add_argument("--valid-rows", type=int, default=1024)
    ap.add_argument("--max-row-groups", type=int, default=0, help="per file; 0 = all")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    row_len = args.seq_length + 1
    train_dir = os.path.join(args.out, "train")
    valid_dir = os.path.join(args.out, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    buf = np.empty(args.rows_per_shard * row_len, dtype=np.int32)
    fill = 0
    files, lens = [], []
    n_docs = n_tokens = 0
    t0 = time.time()

    def flush(n_rows):
        nonlocal fill
        name = f"shard-{len(files):05d}.npy"
        np.save(os.path.join(train_dir, name), buf[: n_rows * row_len].reshape(n_rows, row_len))
        files.append(name)
        lens.append(n_rows)
        rest = fill - n_rows * row_len
        buf[:rest] = buf[n_rows * row_len : fill]
        fill = rest

    for path in args.inputs:
        pf = pq.ParquetFile(path)
        n_rg = pf.metadata.num_row_groups
        if args.max_row_groups:
            n_rg = min(n_rg, args.max_row_groups)
        for rg in range(n_rg):
            texts = pf.read_row_group(rg, columns=["text"]).column(0).to_pylist()
            for enc in tok.encode_batch(texts, add_special_tokens=False):
                ids = enc.ids
                n = len(ids) + 1
                pos = 0
                while pos < n:
                    take = min(n - pos, buf.size - fill)
                    if pos + take == n:  # this slice ends with the EOS
                        buf[fill : fill + take - 1] = ids[pos : pos + take - 1]
                        buf[fill + take - 1] = args.eos_token_id
                    else:
                        buf[fill : fill + take] = ids[pos : pos + take]
                    fill += take
                    pos += take
                    if fill == buf.size:
                        flush(args.rows_per_shard)
                n_docs += 1
                n_tokens += n
            el = time.time() - t0
            print(
                f"{os.path.basename(path)} rg {rg + 1}/{n_rg}  docs {n_docs:,}  "
                f"tokens {n_tokens / 1e9:.3f}B  {n_tokens / el / 1e6:.2f}M tok/s",
                flush=True,
            )

    if fill >= row_len:
        flush(fill // row_len)

    # Move the tail of the last shard into the validation set.
    n_valid = min(args.valid_rows, lens[-1] - 1) if lens else 0
    if n_valid > 0:
        last = np.load(os.path.join(train_dir, files[-1]))
        np.save(os.path.join(valid_dir, "shard-00000.npy"), last[-n_valid:])
        np.save(os.path.join(train_dir, files[-1]), last[:-n_valid])
        lens[-1] -= n_valid
        with open(os.path.join(valid_dir, "metadata.json"), "w") as f:
            json.dump({"size": n_valid, "file_list": ["shard-00000.npy"], "file_len": [n_valid]}, f)

    with open(os.path.join(train_dir, "metadata.json"), "w") as f:
        json.dump({"size": sum(lens), "file_list": files, "file_len": lens}, f)

    print(
        f"DONE docs={n_docs:,} tokens={n_tokens:,} train_rows={sum(lens):,} "
        f"valid_rows={n_valid:,} seq_length={args.seq_length} time={time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
