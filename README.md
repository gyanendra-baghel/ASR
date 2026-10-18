# ASR from Scratch

A learning project to build an English automatic speech recognition model from scratch, targeting approximately 30 million parameters.

The planned pipeline is:

Audio → log-mel features → convolutional downsampling → Transformer encoder → CTC predictions → text

## Current status

Implemented:

- Svarah dataset preparation script with mono, 16 kHz WAV conversion.
- A 20-clip debugging manifest.
- Character tokenizer with 52 tokens, including the CTC blank token.
- Tokenizer and dataset preparation tests.

Pending:

- Audio feature extraction and batching.
- Model architecture and training.
- Evaluation, checkpointing, and inference.

The repository does not include audio files or trained model weights. See [TODO.md](TODO.md) for the detailed roadmap.

## Setup

Install the locked dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Prepare the dataset

The initial experiment uses Indian-accented English from [AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah).

Accept the dataset's access conditions on Hugging Face, then run these commands from the project root:

```bash
uv run hf auth login
uv run python src/prepare_dataset.py
```

Enter a Hugging Face token with read access at the login prompt. Do not save the token in project files.

Preparation saves original recordings in `dataset/raw/svarah/`, mono 16 kHz WAV files in `dataset/processed/svarah/`, and 20 entries in `dataset/manifests/tiny.csv`. It also generates `dataset/manifests/svarah_review.html` for listening and transcript review.

The script protects existing manifests from unrelated overwrites. If this checkout already contains a populated `tiny.csv` but no matching source metadata, back up or move that manifest before running preparation. See [dataset/README.md](dataset/README.md) for details.

Svarah is an evaluation benchmark. Using its clips for training means the resulting scores cannot be reported as official Svarah benchmark results.

## Build the tokenizer

```bash
uv run python src/tokenizer.py
```

This reads `dataset/manifests/tiny.csv` and saves `artifacts/tokenizer.json`. It preserves punctuation, capitalization, numbers, and spaces exactly.

## Run tests

```bash
uv run python -m unittest discover -s tests -v
```

## Project structure

```text
src/
  main.py                 # Placeholder entry point
  prepare_dataset.py      # Dataset preparation
  tokenizer.py            # Character tokenizer
dataset/
  README.md               # Dataset instructions
  raw/                    # Original audio; Svarah downloads excluded from Git
  processed/              # Prepared audio; Svarah outputs excluded from Git
  manifests/              # Tiny, training, validation, and test CSVs
artifacts/
  tokenizer.json          # Saved character vocabulary
tests/                    # Preparation and tokenizer tests
TODO.md                   # Roadmap and completion checklist
```
