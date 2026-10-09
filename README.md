# Train a 30 million parameter ASR model from scratch

Build a working speech recognizer in small stages, then scale toward approximately 30 million parameters. Training from scratch means starting with random weights and learning from audio paired with transcripts.

Use `dataset/` for recordings and manifests. Follow the [dataset preparation checklist](dataset/README.md) to prepare the first audio subset.

## 1. Define the first goal

- [ ] Choose one language.
- [ ] Start with clear, single-speaker speech.
- [ ] Decide how transcripts should handle punctuation, numbers, and capitalization.
- [ ] Set the first milestone: correctly transcribe a few training clips.

## 2. Get a transcribed dataset

- [ ] Find or collect audio with accurate transcripts in the chosen language.
- [ ] Check that the dataset license permits the intended use.
- [ ] Create a manifest containing each audio path, transcript, duration, and speaker ID when available.
- [ ] Select a small subset for initial experiments.
- [ ] Listen to sample recordings and check their transcripts manually.

A few hours can demonstrate learning, but usually will not produce a reliable general-purpose recognizer. More clean, varied speech will be needed as the project grows.

## 3. Prepare the data

- [ ] Convert recordings to mono audio at a consistent sample rate, such as 16 kHz.
- [ ] Remove corrupt recordings, empty transcripts, and incorrect labels.
- [ ] Normalize transcripts consistently without removing meaningful language characters.
- [ ] Start with short clips to simplify batching and reduce memory usage.
- [ ] Split the data into training, validation, and test sets.
- [ ] Keep speakers separate across splits where possible and remove duplicate recordings across splits.
- [ ] Inspect examples from every split after preprocessing.

## 4. Set up the training environment

- [ ] Create a Python environment with PyTorch and audio-processing dependencies.
- [ ] Choose a free GPU notebook platform, such as Kaggle or Google Colab.
- [ ] Verify that PyTorch can access the GPU.
- [ ] Choose persistent storage for datasets and checkpoints.
- [ ] Record dependency versions, configuration, and random seeds.

Free GPU availability and session limits vary. Check the platform's current limits before planning a long run.

## 5. Build the tokenizer and model

- [ ] Create a character vocabulary from training transcripts, including spaces.
- [ ] Reserve a separate CTC blank token and handle batch padding separately.
- [ ] Verify that text can be encoded into token IDs and decoded back correctly.
- [ ] Extract log-mel features from audio.
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
