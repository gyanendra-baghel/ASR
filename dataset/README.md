# Prepare the ASR dataset

Start with 10–20 short recordings and their exact transcripts. Use this subset to verify audio loading and later teach the model to reproduce a few training examples before collecting a larger dataset.

## Svarah dataset size

Our initial dataset choice is [AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), which contains Indian-accented English audio paired with transcripts.

- **Download size:** approximately 1.1 GB (1.02 GiB).
- **Audio duration:** 9.6 hours.
- **Number of clips:** 6,656.
- **Number of speakers:** 117.

Allow extra disk space for cached files and any WAV copies created during preprocessing. These figures come from the [Svarah dataset card](https://huggingface.co/datasets/ai4bharat/Svarah/blob/main/README.md).

Svarah is an evaluation benchmark with only a `test` split. It is suitable for our initial learning experiment, but too small on its own for training a strong 30 million parameter ASR model from scratch. If we use its clips for training, our results will not be comparable to the official Svarah benchmark. Keep it untouched if we later want to report benchmark results.

- [ ] Check available disk space before downloading Svarah.
- [ ] Start with 20 clips for the first debugging experiment.
- [ ] Record which clips are used for training and keep evaluation clips separate.

## Download and prepare the first 20 clips

This step runs on CPU; no GPU is needed. The preparation script selects the first 20 valid clips in source order from a pinned Svarah revision. It keeps full recordings, so some may be longer than the suggested 3–10 seconds.

1. Open [Svarah on Hugging Face](https://huggingface.co/datasets/ai4bharat/Svarah), sign in, and accept its access conditions.
2. From the project root, install the locked dependencies and authenticate:

   ```bash
   uv sync
   uv run hf auth login
   ```

   Use a Hugging Face token with read access to the dataset. Enter it into the login prompt; do not save tokens in project files or Git.

3. Run preparation:

   ```bash
   uv run python src/prepare_dataset.py
   ```

The script streams the dataset with decoding disabled, then uses SoundFile and SciPy to prepare audio. Streaming may transfer more than the selected audio alone because of Parquet buffering.

Outputs:

- `dataset/raw/svarah/`: unchanged original audio bytes and `metadata.jsonl`, preserving transcripts, source metadata, revision, and CC BY 4.0 attribution.
- `dataset/processed/svarah/`: mono, 16 kHz PCM-16 WAV copies.
- `dataset/manifests/tiny.csv`: 20 rows with prepared paths, unchanged transcripts, measured durations, and empty speaker IDs when unavailable. Svarah's published schema has no speaker ID field; demographic fields are not treated as identities.
- `dataset/manifests/svarah_review.html`: browser review page.

The script prints each clip's sample rate, duration, and transcript. It publishes the manifest only after all 20 clips validate. Original files from interrupted runs are retained for retry. A matching prior output can be regenerated; unrelated manifest content is never overwritten. Back up or move an existing custom `tiny.csv` before preparing this subset.

Downloaded audio, generated source metadata, and review artifacts are ignored by Git. Keep them in persistent storage for future GPU sessions.

## Listen and review transcripts

Open `dataset/manifests/svarah_review.html` in your browser. Each clip has an audio player, its original transcript, a review checkbox, and a notes field.

- [ ] Listen to all 20 clips and compare the speech with the transcript.
- [ ] Record missing words, incorrect words, silence, or unintelligible speech in the notes field.
- [ ] Mark a clip reviewed only after listening and checking its transcript.
- [ ] Click **Download review results** and save the JSON file before closing the page.
- [ ] Resolve flagged examples before training; review notes do not automatically change the manifest.

Checkboxes and notes are stored in browser local storage when supported. The downloaded JSON is the reliable saved record. If local-file playback is blocked, run `uv run python -m http.server 8000 --bind 127.0.0.1` from the project root and visit `http://127.0.0.1:8000/dataset/manifests/svarah_review.html`. Stop the server with Ctrl+C afterward.

Run the preparation tests without downloading audio:

```bash
uv run python -m unittest discover -s tests -v
```

## Folder structure

```text
dataset/
  README.md
  raw/                  # Original recordings
  processed/            # Audio prepared for training
  manifests/
    tiny.csv            # Initial debugging subset
    train.csv           # Larger training split
    validation.csv      # Model selection split
    test.csv            # Final evaluation split
```

The folders and empty manifest templates are provided. Add real recordings and manifest rows before using them for training. Keep original recordings in `raw/` and save converted copies in `processed/`.

## 1. Choose the dataset scope

- [ ] Choose one language.
- [ ] Start with clear, single-speaker recordings.
- [ ] Choose your own recordings or a public transcribed speech dataset.
- [ ] Check the dataset license and record its source and permitted uses.
- [ ] If recording people, obtain permission for the intended use.
- [ ] Decide how to represent punctuation, capitalization, and spoken numbers.

## 2. Collect the first recordings

- [ ] Collect 10–20 short clips, approximately 3–10 seconds each.
- [ ] Give every clip a unique filename, such as `clip_001.wav`.
- [ ] Put original recordings in `dataset/raw/`.
- [ ] Write exactly what was spoken for every clip.
- [ ] Record a consistent speaker ID when known.
- [ ] Listen to every initial clip and manually verify its transcript.

A tiny subset is for debugging. A few hours can demonstrate learning, but reliable recognition of varied speech usually requires much more clean, diverse data.

## 3. Prepare audio and text

- [ ] Convert copies of the recordings to mono WAV at 16 kHz.
- [ ] Save prepared recordings in `dataset/processed/`.
- [ ] Remove corrupt files, empty transcripts, and mismatched audio/text pairs.
- [ ] Check for clipped, silent, or unintelligible recordings.
- [ ] Measure each prepared recording's duration in seconds.
- [ ] Normalize transcripts consistently while preserving meaningful language characters.
- [ ] Preserve the original transcripts separately if normalization changes them.

Changing a filename or sample-rate label does not convert audio. Resample the waveform using an audio-processing tool.

## 4. Fill the manifests

Each CSV uses UTF-8 and these columns:

| Column | What to enter |
| --- | --- |
| `audio_path` | Audio path relative to the project root |
| `text` | Exact transcript, using the chosen normalization rules |
| `duration_seconds` | Measured audio duration in seconds |
| `speaker_id` | Consistent speaker identifier, or empty if unknown |

Example rows below are illustrative; the audio files are not included:

```csv
audio_path,text,duration_seconds,speaker_id
dataset/processed/clip_001.wav,hello how are you,3.2,speaker_001
dataset/processed/clip_002.wav,this is my first speech model,4.1,speaker_001
```

Quote text containing commas or double quotes according to CSV rules:

```csv
dataset/processed/clip_003.wav,"hello, how are you",3.5,speaker_002
```

- [ ] Add the initial 10–20 prepared clips to `dataset/manifests/tiny.csv`.
- [ ] Verify that every listed audio path exists.
- [ ] Verify that every row has nonempty text and a positive measured duration.
- [ ] Check that transcripts load correctly as UTF-8.

## 5. Create the larger dataset splits

- [ ] Collect more recordings before evaluating general recognition quality.
- [ ] Assign recordings to training, validation, and test splits.
- [ ] Keep speakers separate across splits where possible.
- [ ] Remove duplicate recordings across splits.
- [ ] Fill `train.csv`, `validation.csv`, and `test.csv`.
- [ ] Use validation data to select models and reserve test data for final evaluation.
- [ ] Inspect sample audio and transcripts from each split.
- [ ] Record clip counts, total audio hours, speaker counts, and the split method.

The tiny debugging manifest can reference a subset of the training split. Keep validation and test recordings out of that debugging subset.

## 6. Verify the dataset is ready

- [ ] Load one manifest row using Python in the project's `uv` environment.
- [ ] Print the recording's sample rate, duration, and transcript.
- [ ] Confirm prepared audio is mono and 16 kHz.
- [ ] Load all tiny-subset rows without errors.
- [ ] Back up recordings, transcripts, and manifests to persistent storage before using a temporary GPU notebook.

The next coding milestone is loading one real recording successfully. After that, build the feature extraction and tokenizer pipeline.
