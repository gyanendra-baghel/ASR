"""Prepare a reproducible, educational 20-clip Svarah subset on CPU."""

import csv
import html
import io
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
DATASET = "ai4bharat/Svarah"
REVISION = "ebbf7777fe771490696a3f7b007097606fa8c924"
COUNT = 20
FIELDS = ["audio_path", "text", "duration_seconds", "speaker_id"]
EXTENSIONS = {"WAV": "wav", "FLAC": "flac", "OGG": "ogg", "MP3": "mp3", "AIFF": "aiff"}


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def decode_audio(content):
    info = sf.info(io.BytesIO(content))
    samples, rate = sf.read(io.BytesIO(content), dtype="float64", always_2d=True)
    if not len(samples) or rate <= 0 or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite samples and a positive sample rate")
    if info.format not in EXTENSIONS:
        raise ValueError(f"Unsupported original audio format: {info.format}")
    return samples, rate, EXTENSIONS[info.format]


def prepare_audio(samples, rate):
    mono = samples.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, rate // divisor)
    # Prevent PCM conversion from wrapping or overflowing after resampling.
    return np.clip(mono, -1.0, 1.0)


def manifest_content(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def check_manifest(path, expected=None):
    if not path.exists():
        return
    existing = path.read_bytes()
    if expected is not None and existing == expected:
        return
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != FIELDS or rows:
        raise ValueError(f"Refusing to overwrite existing manifest: {path}. Back it up or move it first.")


def review_page(rows):
    cards = []
    for row in rows:
        name = Path(row["audio_path"]).stem
        audio = "../processed/svarah/" + Path(row["audio_path"]).name
        cards.append(f'<article><h2>{html.escape(name)}</h2><audio controls preload="none" src="{html.escape(audio, quote=True)}"></audio>'
                     f'<p>{html.escape(row["text"])}</p><p>{row["duration_seconds"]} seconds</p>'
                     f'<label><input type="checkbox" data-id="{name}"> Listened and transcript checked</label>'
                     f'<p><label>Problems or corrections: <input class="notes" data-id="{name}" type="text"></label></p></article>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Svarah clip review</title>
<style>body{max-width:850px;margin:32px auto;padding:0 16px;font:18px system-ui}article{border-top:1px solid #ccc;padding:16px 0}.notes{width:90%}audio{width:100%}</style>
<h1>Review the 20 Svarah clips</h1><p>Listen to each clip and compare it with its transcript. Record problems; this page does not change training text.</p>
<p>Download your review results before closing this page. Browser storage may be unavailable for local files.</p>
<button id="download">Download review results</button>
''' + "\n".join(cards) + '''<script>
const key="svarah-review-''' + REVISION + '''";
const inputs=[...document.querySelectorAll('[data-id]')];
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function results(){return [...document.querySelectorAll('input[type=checkbox]')].map(c=>({clip:c.dataset.id,reviewed:c.checked,notes:document.querySelector('.notes[data-id="'+c.dataset.id+'"]').value}));}
for(const input of inputs){const previous=saved[input.dataset.id];if(previous){if(input.type==='checkbox')input.checked=previous.reviewed;else input.value=previous.notes;}
input.addEventListener('input',()=>{try{localStorage.setItem(key,JSON.stringify(Object.fromEntries(results().map(r=>[r.clip,r]))))}catch(e){}});}
document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(results(),null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='svarah-review-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script></html>'''


def prepare_dataset(records, root=ROOT):
    raw = root / "dataset/raw/svarah"
    processed = root / "dataset/processed/svarah"
    manifest = root / "dataset/manifests/tiny.csv"
    metadata_path = raw / "metadata.jsonl"
    # A previous successful run owns its manifest only if it matches provenance.
    expected = None
    if metadata_path.exists():
        saved = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
        if len(saved) == COUNT and all(r.get("revision") == REVISION and r.get("dataset") == DATASET for r in saved):
            expected = manifest_content([r["manifest"] for r in saved])
    check_manifest(manifest, expected)
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    rows, metadata = [], []
    for index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            print(f"Skipping source row {index}: empty transcript", file=sys.stderr)
            continue
        audio = record.get("audio_filepath", {})
        content = audio.get("bytes")
        if not content:
            raise ValueError(f"Source row {index} has no embedded audio bytes; cannot preserve its original audio")
        try:
            samples, rate, extension = decode_audio(content)
        except (ValueError, sf.LibsndfileError) as error:
            print(f"Skipping source row {index}: {error}", file=sys.stderr)
            continue
        name = f"clip_{index:06d}"
        original = raw / f"{name}.{extension}"
        if original.exists() and original.read_bytes() != content:
            raise ValueError(f"Existing original differs from source: {original}")
        if not original.exists():
            atomic_write(original, content)
        output = io.BytesIO()
        sf.write(output, prepare_audio(samples, rate), 16000, format="WAV", subtype="PCM_16")
        destination = processed / f"{name}.wav"
        atomic_write(destination, output.getvalue())
        info = sf.info(destination)
        if info.channels != 1 or info.samplerate != 16000 or info.frames <= 0:
            raise ValueError(f"Prepared audio validation failed: {destination}")
        row = dict(audio_path=destination.relative_to(root).as_posix(), text=text,
                   duration_seconds=info.frames / info.samplerate, speaker_id=record.get("speaker_id") or "")
        rows.append(row)
        source_metadata = {k: v for k, v in record.items() if k != "audio_filepath"}
        metadata.append(dict(dataset=DATASET, revision=REVISION, split="test", source_row=index,
                             original_file=original.relative_to(root).as_posix(), source_audio_path=audio.get("path"),
                             original_text=text, source_metadata=source_metadata, license="CC BY 4.0", manifest=row))
        print(f"{name}: 16000 Hz, mono, {row['duration_seconds']:.3f}s | {text}")
        if len(rows) == COUNT:
            break
    if len(rows) != COUNT:
        raise ValueError(f"Only found {len(rows)} valid clips; need {COUNT}. The manifest was not changed.")
    content = manifest_content(rows)
    check_manifest(manifest, expected)
    atomic_write(metadata_path, ("\n".join(json.dumps(r, ensure_ascii=False) for r in metadata) + "\n").encode("utf-8"))
    atomic_write(manifest, content)
    review = manifest.with_name("svarah_review.html")
    atomic_write(review, review_page(rows).encode("utf-8"))
    total = sum(r["duration_seconds"] for r in rows)
    print(f"Prepared {len(rows)} clips ({total:.1f} seconds).\nOriginals: {raw}\nPrepared WAVs: {processed}\nManifest: {manifest}\nReview: {review}")
    return rows


def main():
    try:
        from datasets import Audio, load_dataset
        from huggingface_hub import get_token
        token = get_token()
        if not token:
            raise ValueError("Svarah requires access approval and authentication. Accept its conditions on Hugging Face, then run: uv run hf auth login")
        # Validate ownership before making a network request.
        manifest = ROOT / "dataset/manifests/tiny.csv"
        if not (ROOT / "dataset/raw/svarah/metadata.jsonl").exists():
            check_manifest(manifest)
        records = load_dataset(DATASET, split="test", revision=REVISION, streaming=True, token=token)
        records = records.cast_column("audio_filepath", Audio(decode=False))
        prepare_dataset(records)
    except Exception as error:
        print(f"Dataset preparation failed: {error}\nFor access errors, accept the dataset conditions and run 'uv run hf auth login'. For connection errors, check your network and rerun; original files are retained.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
