# Dataset Analysis Task

## Context

This is Exercise 1.3 of the ADLCV course at DTU. The exercise involves training a GPT-2 style
model called AndersenGPT on the `monurcan/andersen_fairy_tales` HuggingFace dataset.

**Relevant files to read first:**
- `train.py` — contains the dataset loading logic (`prepare_data_iter`), tokenizer setup,
  chunking strategy (non-overlapping 1024-token chunks), and all hyperparameters
- `gpt.py` — the model architecture (for context on MAX_SEQ_LEN=1024)

## Your Task

Run a dataset analysis script (install `datasets` and `transformers` if needed) and write
the findings to `data_summary.md` in this directory. Keep it short and factual — bullet points
are fine. It will be used as reference when writing a report section.

## What to Analyse

1. **Split sizes** — number of stories in train vs. validation
2. **Raw text stats** — total characters, mean/min/max story length in characters (train and val separately)
3. **Token stats** — using the GPT-2 tokenizer (`AutoTokenizer.from_pretrained("gpt2")`):
   total tokens, mean/median/min/max story length in tokens (train and val separately)
4. **Chunk counts** — using the same chunking logic as `train.py` (non-overlapping windows of
   `max_seq_len+1 = 1025` tokens, keeping chunks with >1 token): how many training chunks and
   validation chunks are produced at `MAX_SEQ_LEN=1024`
5. **Vocabulary coverage** — how many of GPT-2's 50,257 tokens actually appear in the full
   dataset (train + val combined), as an absolute count and percentage
6. **A few sample story openings** — first ~150 characters of 2-3 train stories, just to
   illustrate the writing style

## Output Format

Write results to `data_summary.md` as a short markdown document. Keep it concise — a few
bullet-point sections, no prose padding. Example structure:

```
# Dataset Summary: andersen_fairy_tales

## Splits
- Train: X stories, Val: Y stories

## Text Statistics
...

## Token Statistics (GPT-2 tokenizer)
...

## Training Chunks (seq_len=1024)
...

## Vocabulary Coverage
...

## Sample Stories
...
```
