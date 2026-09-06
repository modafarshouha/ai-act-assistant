# Vendored tokenizer

`bge-small-en-v1.5/` holds the tokenizer that ships with
`BAAI/bge-small-en-v1.5`, the embedding model this project uses.

## Why the tokenizer is committed but the weights are not

The chunker promises that no chunk goes over the model's 512-token window.
Anything past that gets truncated without a warning, and the tail of a provision
ends up embedded as if it were not there. Checking the promise needs the model's
real tokenizer. Legal text runs about 1.21 WordPiece tokens per word, so a
word-count estimate is not close enough when being wrong loses data silently.

Sizes: the tokenizer is 712 KB, the fp32 weights are 127 MB. Committing only the
tokenizer lets the chunker and its tests run with no network and no download.
`fastembed` still fetches the weights when embeddings actually get computed, and
caches them after that.

## Drift

A vendored tokenizer can fall out of step with whatever fastembed downloads
later, which would make every token count quietly wrong. Nothing here checks for
that automatically. If you bump the embedding model, re-copy `tokenizer.json`
from the new distribution and re-run `scripts/ingest.py`; the chunk-size tests
`tests/test_chunker.py` will catch a mismatch large enough to push a chunk over
the window, but not a small one.

## Licence

`BAAI/bge-small-en-v1.5` is MIT licensed.
