# Train a 30 million parameter ASR model from scratch

Build a working speech recognizer in small stages, then scale toward approximately 30 million parameters. Training from scratch means starting with random weights and learning from audio paired with transcripts.

Use `dataset/` for recordings and manifests. Follow the [dataset preparation checklist](dataset/README.md) to prepare the first audio subset.

## 1. Define the first goal

- [x] Choose one language: English, using the Indian-accented Svarah subset.
- [ ] Start with clear, single-speaker speech.
- [x] Decide how transcripts should handle punctuation, numbers, and capitalization: preserve the source text exactly.
- [x] Set the first milestone: correctly transcribe a few training clips.

## 2. Get a transcribed dataset

- [x] Select a transcribed dataset in the chosen language: AI4Bharat Svarah. Local audio availability and transcript review remain pending.
- [ ] Check that the dataset license permits the intended use.
- [x] Create a manifest containing each audio path, transcript, duration, and speaker ID when available.
- [x] Select a small subset for initial experiments: 20 clips in `dataset/manifests/tiny.csv`.
- [ ] Listen to sample recordings and check their transcripts manually.

A few hours can demonstrate learning, but usually will not produce a reliable general-purpose recognizer. More clean, varied speech will be needed as the project grows.

## 3. Prepare the data

- [ ] Convert recordings to mono audio at a consistent sample rate, such as 16 kHz.
- [ ] Remove corrupt recordings, empty transcripts, and incorrect labels.
- [x] Define and implement the initial transcript policy: preserve all source characters without normalization.
- [ ] Start with short clips to simplify batching and reduce memory usage.
- [ ] Split the data into training, validation, and test sets.
- [ ] Keep speakers separate across splits where possible and remove duplicate recordings across splits.
- [ ] Inspect examples from every split after preprocessing.

The preparation script implements mono, 16 kHz conversion and rejects corrupt audio and empty transcripts. The tiny manifest contains 20 entries totaling approximately 90 seconds, but its referenced audio files are absent from this checkout. Audio preparation is therefore not verified locally, and incorrect labels still require listening review. The subset retains full recordings rather than enforcing a short-clip duration limit.

## 4. Set up the training environment

- [ ] Create a Python environment with PyTorch and audio-processing dependencies.
- [ ] Choose a free GPU notebook platform, such as Kaggle or Google Colab.
- [ ] Verify that PyTorch can access the GPU.
- [ ] Choose persistent storage for datasets and checkpoints.
- [ ] Record dependency versions, configuration, and random seeds.

Audio-processing dependencies are declared and locked in `pyproject.toml` and `uv.lock`. PyTorch, training configuration, random seeds, and a verified GPU environment are still pending.

Free GPU availability and session limits vary. Check the platform's current limits before planning a long run.

## 5. Build the tokenizer and model

### Build the tokenizer

- [x] Read transcripts from `dataset/manifests/tiny.csv` for the initial debugging experiment.
- [x] Create a character vocabulary from training transcripts, including spaces.
- [x] Reserve a separate CTC blank token and define batch padding separately.
- [x] Verify that encoding and decoding reproduce the original transcript exactly, including repeated characters.

Build or regenerate the tokenizer:

```bash
uv run python src/tokenizer.py
```

The tokenizer is saved to `artifacts/tokenizer.json`. The current 20 transcripts produce 52 tokens, including CTC blank ID `0`. Characters are sorted for deterministic IDs; case, punctuation, spaces, and repeated characters are preserved without normalization. Unknown characters raise an error instead of silently changing text.

Use `CharacterTokenizer.load(path)`, `encode(text)`, and `decode(token_ids)` from `src.tokenizer` for text. Use `decode_ctc(prediction_ids)` for model predictions. `PAD_ID = -100` is a future batching sentinel, not a vocabulary token; exclude padding from CTC targets using transcript lengths.

For another training manifest, use `--manifest path/to/train.csv --output path/to/tokenizer.json`. Build the vocabulary from training text only and freeze it before training; changing token IDs invalidates compatibility with model checkpoints.

### Extract audio features

- [ ] Load the prepared mono, 16 kHz WAV files listed in the manifest.
- [ ] Convert each recording into 80-bin log-mel features.
- [ ] Check that features contain only finite values.
- [ ] Record each recording's feature length before padding.

### Build the training data loader

- [ ] Pair each recording's features with its encoded transcript.
- [ ] Pad variable-length examples into batches.
- [ ] Preserve unpadded feature and transcript lengths for CTC loss.
- [ ] Produce padding masks so the model can ignore padded audio frames.
- [ ] Load one batch successfully and print feature shapes, transcript tokens, and lengths.

The next milestone is a valid batch. Tokenizer round-trip decoding preserves repeated characters; collapsing repeated predictions is a separate step used when decoding CTC model outputs.

### Build the model

- [ ] Implement convolutional downsampling followed by a Transformer encoder.
- [ ] Add an output layer that predicts vocabulary tokens at each time step.
- [ ] Configure CTC loss using the correct input and transcript lengths.
- [ ] Verify that downsampled audio sequences are long enough for CTC alignment, including repeated characters.
- [ ] Count trainable parameters and adjust encoder depth and width toward approximately 30 million.
- [ ] Run a forward pass and check tensor shapes and finite loss values.

Model path: audio → log-mel features → convolutional downsampling → Transformer encoder → character predictions.

## 6. Prove the pipeline works on a tiny dataset

- [ ] Select approximately 10–20 short, accurately transcribed clips.
- [ ] Temporarily disable data augmentation for this debugging run.
- [ ] Train repeatedly on those clips until the model reproduces their transcripts.
- [ ] Decode predictions by collapsing consecutive repeated tokens, then removing CTC blanks.
- [ ] Investigate preprocessing, tokenization, lengths, or loss configuration if the model cannot learn the clips.

Passing this step verifies the pipeline; it does not demonstrate recognition of unseen speech.

## 7. Train on the larger dataset

- [ ] Start with a small batch size and monitor GPU memory.
- [ ] Enable supported mixed precision and verify that training remains stable.
- [ ] Use gradient accumulation if a larger effective batch size is needed.
- [ ] Group clips of similar lengths to reduce padding.
- [ ] Configure the optimizer, learning-rate schedule, and gradient clipping.
- [ ] Track training loss, validation loss, and validation word error rate (WER).
- [ ] Save regular checkpoints containing model, optimizer, scheduler, mixed-precision scaler state when used, and training step.
- [ ] Save the tokenizer and configuration with checkpoints.
- [ ] Verify that training resumes correctly from a saved checkpoint.
- [ ] Measure training speed to estimate the GPU hours required for the planned run.

## 8. Evaluate recognition quality

- [ ] Choose the best checkpoint using validation results.
- [ ] Evaluate that checkpoint on the untouched test set.
- [ ] Report WER using consistent reference and prediction normalization.
- [ ] Track character error rate (CER) if useful for the chosen language.
- [ ] Listen to failed examples and inspect substitutions, deletions, and insertions.
- [ ] Test speech from speakers the model has not heard during training.
- [ ] Save evaluation results and representative predictions.

## 9. Improve and scale

- [ ] Add more clean audio with diverse speakers and recording conditions.
- [ ] Introduce suitable data augmentation after the basic pipeline works.
- [ ] Change one major setting at a time and compare validation results.
- [ ] Estimate paid compute from measured GPU hours and the current rental rate, including storage costs.
- [ ] Use paid GPU time if free session limits prevent practical progress.
- [ ] Save a final model package with weights, tokenizer, preprocessing configuration, and inference instructions.

## Completion milestones

- [ ] The model transcribes its tiny training subset correctly.
- [ ] Training can resume after a notebook session ends.
- [ ] The model recognizes unseen speakers with measured test WER.
- [ ] The final model has approximately 30 million trainable parameters.
- [ ] A saved model can transcribe a new audio file through an inference script.
