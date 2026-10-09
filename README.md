# ai-project

This repository contains Python projects related to Natural Language Processing (NLP), tokenization, data processing, and GPT-style language models. The goal is to understand how text data is prepared and how the main components of a language model work.

## Projects

1. Data Pipeline

File: "data_pipeline.py"

This project focuses on preparing text data before training a language model.

Features:

- Text cleaning and normalization
- Filtering low-quality documents
- Word shingling
- MinHash for estimating document similarity
- Locality-Sensitive Hashing (LSH)
- Near-duplicate detection and removal
- BPE tokenization
- Packing tokens into fixed-length sequences
- Preparing batches and attention masks
- Dataset statistics

2.Character-Level and BPE Tokenizers

File: "tokenizer1.py"

This project implements character-level and Byte-Pair Encoding (BPE) tokenizers.

Features:

- Converting characters into integer IDs
- Encoding and decoding text
- Starting BPE with 256 base byte tokens
- Counting adjacent token pairs
- Merging the most frequent pairs
- Building a vocabulary from learned merge rules
- Calculating compression ratios and vocabulary statistics

3.Advanced BPE Tokenizer

File: "tokenizer2.py"

This project extends the basic BPE implementation with features used in GPT-style tokenization.

Features:

- UTF-8 byte-level tokenization
- GPT-2-style pre-tokenization
- Unicode text normalization
- Special token handling
- Vocabulary and merge-rule management
- Encoding and decoding
- Comparison with OpenAI's "tiktoken"

4.Mini GPT with PyTorch

File: "mini_gpt_torch.py"

This project focuses on implementing the main components of a small GPT-style language model using PyTorch.

Features:

- Token and positional embeddings
- Layer normalization
- Causal multi-head self-attention
- Feed-forward neural networks
- Transformer blocks
- Mini GPT architecture
- Causal masking and cross-entropy loss
- Training and text generation utilities
- Parameter counting and memory estimation

The goal is to understand how Transformer components work together to predict the next token.

## Technologies Used

- Python
- PyTorch
- Regular Expressions
- tiktoken

## How to Run

Run each Python file separately:

python data_pipeline.py

python tokenizer1.py

python tokenizer2.py

python mini_gpt_torch.py

Some features require additional Python packages, such as "torch", "regex", and "tiktoken".

## Key Concepts

- Text preprocessing and data quality
- Document similarity and duplicate detection
- Character-level and byte-level tokenization
- Byte-Pair Encoding (BPE)
- Vocabulary construction
- Preparing training sequences and batches
- Transformer architecture
- Causal self-attention
- Language model training and text generation

## Conclusion

These projects are part of my learning journey in NLP and AI Engineering. They provide practical experience with text processing, tokenization, data preparation, and the internal components of GPT-style language models.

This repository will be updated as I continue learning and improving the implementations.