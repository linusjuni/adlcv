# Dataset Summary: andersen_fairy_tales

Source: `monurcan/andersen_fairy_tales` (HuggingFace)

## Splits
- Train: 120 stories
- Validation: 7 stories

## Text Statistics (characters)

| Split | Total | Mean | Min | Max |
|-------|------:|-----:|----:|----:|
| Train | 1,950,580 | 16,255 | 1,971 | 106,977 |
| Val   |    50,723 |  7,246 | 2,608 |  12,342 |

## Token Statistics (GPT-2 tokenizer, with special tokens)

| Split | Total | Mean | Median | Min | Max |
|-------|------:|-----:|-------:|----:|----:|
| Train | 500,290 | 4,169 | 2,594 | 493 | 27,325 |
| Val   |  12,950 | 1,850 | 1,924 | 687 |  3,143 |

Note: the tokenizer warns that some stories exceed GPT-2's native 1024-token limit; this is expected since stories are chunked before being fed to the model.

## Training Chunks (seq\_len=1024)

Non-overlapping windows of 1025 tokens, keeping chunks with >1 token:

- Train: **550 chunks**
- Validation: **16 chunks**

## Vocabulary Coverage

- Unique GPT-2 tokens appearing in the full dataset (train + val): **14,268 / 50,257 (28.4%)**
- The dataset uses less than a third of the GPT-2 vocabulary, consistent with a narrow literary domain (19th-century English fairy tales).

## Sample Story Openings

**Story 1 — "ANNE LISBETH"**
> Anne Lisbeth was a beautiful young woman, with a red and white complexion, glittering white teeth, and clear soft eyes; and her footst…

**Story 2 — "THE PSYCHE"**
> In the fresh morning dawn, in the rosy air gleams a great Star, the brightest Star of the morning. His rays tremble on the white wall, a…

**Story 3 — "THE SHEPHERDESS AND THE SHEEP"**
> Have you ever seen an old wooden cupboard quite black with age, and ornamented with carved foliage and curious figure…
